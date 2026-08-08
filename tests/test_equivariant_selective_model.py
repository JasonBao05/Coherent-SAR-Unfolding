from __future__ import annotations

import torch

from coherent_sar.equivariant_selective_model import (
    ContinuousSelectiveGate,
    FixedMaskExchangeEquivariantSelectiveUnfolding256,
)
from coherent_sar.formal_wideband_operator import (
    MatrixFreeChunkedWidebandSAROperator,
    WidebandSARGroundPlaneGeometry,
)


def _operator() -> MatrixFreeChunkedWidebandSAROperator:
    return MatrixFreeChunkedWidebandSAROperator(
        WidebandSARGroundPlaneGeometry(
            image_height=32,
            image_width=32,
            pixel_spacing_ground_range_m=0.5,
            pixel_spacing_cross_range_m=0.5,
            aperture_count=24,
            frequency_count=20,
            aperture_length_m=1.6,
            bandwidth_hz=160.0e6,
            aperture_chunk_size=7,
        )
    )


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    first = torch.zeros(24)
    second = torch.zeros(24)
    first[torch.tensor([0, 2, 5, 8, 11, 14, 17, 20, 23])] = 1.0
    second[torch.tensor([0, 3, 6, 9, 12, 15, 18, 21, 23])] = 1.0
    return first, second


def _model(mask: torch.Tensor, stages: int = 2) -> FixedMaskExchangeEquivariantSelectiveUnfolding256:
    return FixedMaskExchangeEquivariantSelectiveUnfolding256(
        mask,
        aperture_count=24,
        stages=stages,
        hidden_channels=8,
        coherence_window_size=5,
        gradient_checkpointing=False,
    )


def test_selective_gate_has_exact_continuous_fallback_boundaries() -> None:
    gate = ContinuousSelectiveGate()
    shape = (1, 1, 8, 8)
    residual1 = torch.full(shape, 0.1)
    residual2 = torch.full(shape, 0.1)
    assert torch.count_nonzero(gate(torch.zeros(shape), residual1, residual2)) == 0
    maximal_conflict = gate(
        torch.full(shape, 0.9), torch.zeros(shape), torch.ones(shape)
    )
    assert torch.count_nonzero(maximal_conflict) == 0
    low = gate(torch.full(shape, 0.3), residual1, residual2)
    high = gate(torch.full(shape, 0.8), residual1, residual2)
    conflict = gate(
        torch.full(shape, 0.8), torch.zeros(shape), torch.full(shape, 0.8)
    )
    assert torch.all(high > low)
    assert torch.all(conflict < high)


def test_forward_is_strictly_exchange_equivariant_with_swapped_masks() -> None:
    torch.manual_seed(901)
    operator = _operator()
    mask1, mask2 = _masks()
    truth1 = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    truth2 = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    measurement1 = operator.forward_masked(truth1, mask1)
    measurement2 = operator.forward_masked(truth2, mask2)
    model = _model(mask1).eval()
    output1, output2 = model(
        measurement1, measurement2, operator, mask1=mask1, mask2=mask2
    )
    swapped2, swapped1 = model(
        measurement2, measurement1, operator, mask1=mask2, mask2=mask1
    )
    assert torch.equal(output1, swapped1)
    assert torch.equal(output2, swapped2)


def test_identical_inputs_produce_identical_outputs() -> None:
    torch.manual_seed(902)
    operator = _operator()
    mask, _ = _masks()
    truth = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    measurement = operator.forward_masked(truth, mask)
    model = _model(mask).eval()
    output1, output2 = model(measurement, measurement, operator)
    assert torch.equal(output1, output2)


def test_independent_override_removes_all_cross_epoch_dependence() -> None:
    torch.manual_seed(903)
    operator = _operator()
    mask, _ = _masks()
    truth1 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    truth2a = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    truth2b = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    measurement1 = operator.forward_masked(truth1, mask)
    measurement2a = operator.forward_masked(truth2a, mask)
    measurement2b = operator.forward_masked(truth2b, mask)
    model = _model(mask).eval()
    first_a, _ = model(
        measurement1, measurement2a, operator, coupling_override="independent"
    )
    first_b, _ = model(
        measurement1, measurement2b, operator, coupling_override="independent"
    )
    assert torch.equal(first_a, first_b)


def test_forward_backward_keeps_fixed_mask_and_trains_selective_gate() -> None:
    torch.manual_seed(904)
    operator = _operator()
    mask, _ = _masks()
    truth1 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    truth2 = torch.complex(torch.randn(1, 32, 32), torch.randn(1, 32, 32))
    measurement1 = operator.forward_masked(truth1, mask)
    measurement2 = operator.forward_masked(truth2, mask)
    model = _model(mask, stages=3).train()
    initial_mask = model.fixed_mask.detach().clone()
    estimate1, estimate2, diagnostics = model(
        measurement1, measurement2, operator, return_diagnostics=True
    )
    regularization = model.gate_regularization()
    loss = (estimate1 - truth1).abs().square().mean()
    loss = loss + (estimate2 - truth2).abs().square().mean()
    loss = loss + regularization["smoothness"] + regularization["anti_collapse"]
    loss.backward()
    assert torch.equal(initial_mask, model.fixed_mask)
    assert model.fixed_mask.requires_grad is False and model.fixed_mask.grad is None
    assert all(torch.isfinite(stage["gate"]).all() for stage in diagnostics["stages"])
    gate_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("gates.1")
    ]
    assert gate_gradients and all(value is not None for value in gate_gradients)
    assert sum(float(value.abs().sum()) for value in gate_gradients) > 0.0
