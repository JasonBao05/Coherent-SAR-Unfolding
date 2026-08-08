"""Exchange-equivariant dual-epoch unfolding with selective coupling.

The architecture in this module enforces the two central DP-JMRNet
properties by construction:

* exchanging the two observations (and their masks) exchanges the outputs;
* cross-epoch information is multiplied by a continuous, physics-facing gate,
  so a zero gate is exactly the independent-reconstruction path.

No differential-phase image is supplied to the learned prior or gate.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .components import (
    ComplexStageDenoiser,
    FeatureBranch,
    local_coherence_magnitude,
    normalized_dc_residual,
)
from .formal_wideband_operator import MatrixFreeChunkedWidebandSAROperator
from .torch_complex import channels_to_complex, complex_to_channels


CouplingOverride = Literal["selective", "independent", "always_on"]


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus initialization must be positive")
    return math.log(math.expm1(value))


class IndependentMaskConditioner(nn.Module):
    """Shared per-epoch conditioner that depends only on that epoch's mask."""

    def __init__(self, aperture_count: int, stages: int, channels: int) -> None:
        super().__init__()
        self.stages = int(stages)
        self.channels = int(channels)
        output_count = stages * (2 * channels + 3) + 1
        self.network = nn.Sequential(
            nn.Linear(aperture_count + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, output_count),
        )

    def forward(self, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        density = mask.mean(dim=-1, keepdim=True)
        output = self.network(torch.cat((mask, density), dim=-1))
        cursor = 0
        film_count = self.stages * self.channels
        gamma = output[:, cursor : cursor + film_count].reshape(
            -1, self.stages, self.channels
        )
        cursor += film_count
        beta = output[:, cursor : cursor + film_count].reshape(
            -1, self.stages, self.channels
        )
        cursor += film_count
        pre_step_delta = output[:, cursor : cursor + self.stages]
        cursor += self.stages
        post_step_delta = output[:, cursor : cursor + self.stages]
        cursor += self.stages
        regularization_delta = output[:, cursor : cursor + self.stages]
        cursor += self.stages
        return {
            "gamma": 0.2 * torch.tanh(gamma),
            "beta": 0.2 * torch.tanh(beta),
            "pre_step_delta": 0.25 * torch.tanh(pre_step_delta),
            "post_step_delta": 0.25 * torch.tanh(post_step_delta),
            "regularization_delta": 0.25 * torch.tanh(regularization_delta),
            "phase_scale": torch.tanh(output[:, cursor]),
        }


class ContinuousSelectiveGate(nn.Module):
    """Symmetric gate with exact low-coherence/high-conflict fallback.

    ``gamma ** p`` makes the gate exactly zero at zero coherence, while
    ``(1-conflict) ** q`` makes it exactly zero at maximal normalized residual
    conflict.  Positive learned exponents preserve the intended monotonicity.
    The detached evidence prevents the reconstructor from manipulating its own
    gate inputs.
    """

    def __init__(self, *, analytical_anchor: bool = False) -> None:
        super().__init__()
        self.analytical_anchor = bool(analytical_anchor)
        if not self.analytical_anchor:
            self.raw_coherence_power = nn.Parameter(
                torch.tensor(_inverse_softplus(1.0))
            )
            self.raw_conflict_power = nn.Parameter(
                torch.tensor(_inverse_softplus(1.5))
            )
            self.raw_mean_residual_scale = nn.Parameter(
                torch.tensor(_inverse_softplus(0.35))
            )

    def positive_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.analytical_anchor:
            raise RuntimeError("the analytical anchor has no learned parameters")
        return (
            F.softplus(self.raw_coherence_power).clamp_min(1.0e-4),
            F.softplus(self.raw_conflict_power).clamp_min(1.0e-4),
            F.softplus(self.raw_mean_residual_scale),
        )

    def forward(
        self,
        gamma: torch.Tensor,
        residual1: torch.Tensor,
        residual2: torch.Tensor,
    ) -> torch.Tensor:
        coherence = gamma.detach().clamp(0.0, 1.0)
        mean_residual = (0.5 * (residual1 + residual2)).detach().clamp(0.0, 1.0)
        conflict = (residual1 - residual2).abs().detach().clamp(0.0, 1.0)
        if self.analytical_anchor:
            coherence_power = coherence.new_tensor(1.0)
            conflict_power = coherence.new_tensor(1.5)
            mean_scale = coherence.new_tensor(0.35)
        else:
            coherence_power, conflict_power, mean_scale = self.positive_parameters()
        return (
            coherence.pow(coherence_power)
            * (1.0 - conflict).pow(conflict_power)
            * torch.exp(-mean_scale * mean_residual)
        ).clamp(0.0, 1.0)

    def summary(self) -> dict[str, float | bool]:
        if self.analytical_anchor:
            return {
                "analytical_anchor": True,
                "coherence_power": 1.0,
                "conflict_power": 1.5,
                "mean_residual_scale": 0.35,
            }
        coherence, conflict, mean = self.positive_parameters()
        return {
            "analytical_anchor": False,
            "coherence_power": float(coherence.detach().cpu()),
            "conflict_power": float(conflict.detach().cpu()),
            "mean_residual_scale": float(mean.detach().cpu()),
        }


class EquivariantInteraction(nn.Module):
    """One shared endpoint decoder applied in both argument orders."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.symmetric = FeatureBranch(4, hidden_channels)
        self.endpoint = FeatureBranch(2, hidden_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(3 * hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude1 = first.abs()
        amplitude2 = second.abs()
        symmetric_input = torch.stack(
            (
                0.5 * (amplitude1 + amplitude2),
                (amplitude1 - amplitude2).abs(),
                torch.sqrt((amplitude1 * amplitude2).clamp_min(0.0)),
                torch.maximum(amplitude1, amplitude2),
            ),
            dim=1,
        )
        shared = self.symmetric(symmetric_input)
        endpoint1 = self.endpoint(complex_to_channels(first, channel_dim=1))
        endpoint2 = self.endpoint(complex_to_channels(second, channel_dim=1))
        correction1 = self.decoder(torch.cat((shared, endpoint1, endpoint2), dim=1))
        correction2 = self.decoder(torch.cat((shared, endpoint2, endpoint1), dim=1))
        return (
            channels_to_complex(correction1, channel_dim=1),
            channels_to_complex(correction2, channel_dim=1),
        )


class FixedMaskExchangeEquivariantSelectiveUnfolding256(nn.Module):
    """DP-JMRNet: exact exchange equivariance and selective fallback.

    Five stages are used by the paper configuration.  Each stage contains a
    pre-prior data-consistency update and a post-coupling data-consistency
    update.  This strengthens the physical constraint without copying the
    ten-step CG block of J-MoDL-SAR.
    """

    def __init__(
        self,
        fixed_mask: torch.Tensor,
        *,
        aperture_count: int = 256,
        stages: int = 5,
        hidden_channels: int = 16,
        coherence_window_size: int = 7,
        coherence_eps: float = 1.0e-8,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        hard_mask = torch.as_tensor(fixed_mask, dtype=torch.float32).reshape(-1)
        if hard_mask.numel() != aperture_count:
            raise ValueError("fixed mask length does not match aperture count")
        if not torch.all((hard_mask == 0.0) | (hard_mask == 1.0)):
            raise ValueError("fixed mask must be binary")
        self.register_buffer("fixed_mask", hard_mask.clone(), persistent=True)
        self.stages = int(stages)
        self.hidden_channels = int(hidden_channels)
        self.coherence_window_size = int(coherence_window_size)
        self.coherence_eps = float(coherence_eps)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.raw_pre_steps = nn.Parameter(torch.full((stages,), -1.05))
        self.raw_post_steps = nn.Parameter(torch.full((stages,), -1.35))
        self.raw_regularization = nn.Parameter(torch.full((stages,), -1.8))
        self.raw_interaction_scale = nn.Parameter(torch.full((stages,), -2.2))
        self.conditioner = IndependentMaskConditioner(
            aperture_count, stages, hidden_channels
        )
        self.denoisers = nn.ModuleList(
            ComplexStageDenoiser(hidden_channels) for _ in range(stages)
        )
        self.interactions = nn.ModuleList(
            EquivariantInteraction(hidden_channels) for _ in range(stages)
        )
        self.gates = nn.ModuleList(
            ContinuousSelectiveGate(analytical_anchor=(stage == 0))
            for stage in range(stages)
        )
        self.phase_refiner = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        self._last_gate_regularization: dict[str, torch.Tensor] = {}

    def _mask_batch(
        self, mask: torch.Tensor | None, batch: int, name: str
    ) -> torch.Tensor:
        value = self.fixed_mask if mask is None else mask
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch, -1)
        if tuple(value.shape) != (batch, self.fixed_mask.numel()):
            raise ValueError(f"{name} shape is incompatible with the frozen model")
        if not torch.all((value == 0.0) | (value == 1.0)):
            raise ValueError(f"{name} must be binary")
        return value

    def _denoise(
        self,
        stage: int,
        value: torch.Tensor,
        strength: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                lambda a, b, c, d: self.denoisers[stage](a, b, c, d),
                value,
                strength,
                gamma,
                beta,
                use_reentrant=False,
            )
        return self.denoisers[stage](value, strength, gamma, beta)

    def _interaction(
        self, stage: int, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.gradient_checkpointing:
            return checkpoint(
                self.interactions[stage], first, second, use_reentrant=False
            )
        return self.interactions[stage](first, second)

    @staticmethod
    def _gate_penalties(
        gates: list[torch.Tensor], evidence: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        smoothness = torch.stack(
            [
                0.5
                * (
                    gate[..., 1:, :].sub(gate[..., :-1, :]).square().mean()
                    + gate[..., :, 1:].sub(gate[..., :, :-1]).square().mean()
                )
                for gate in gates
            ]
        ).mean()
        activation_terms = []
        for gate, (gamma, residual1, residual2) in zip(gates, evidence):
            mean_residual = 0.5 * (residual1 + residual2)
            conflict = (residual1 - residual2).abs()
            eligible = (
                torch.sigmoid(20.0 * (gamma.detach() - 0.70))
                * torch.sigmoid(20.0 * (0.30 - mean_residual.detach()))
                * torch.sigmoid(20.0 * (0.30 - conflict.detach()))
            )
            eligible_mean = eligible.flatten(1).sum(dim=1).clamp_min(1.0)
            active_mean = (gate * eligible).flatten(1).sum(dim=1) / eligible_mean
            activation_terms.append(F.relu(gate.new_tensor(0.35) - active_mean).square().mean())
        return {
            "smoothness": smoothness,
            "anti_collapse": torch.stack(activation_terms).mean(),
        }

    def gate_regularization(self) -> dict[str, torch.Tensor]:
        if not self._last_gate_regularization:
            zero = self.raw_pre_steps.sum() * 0.0
            return {"smoothness": zero, "anti_collapse": zero}
        return self._last_gate_regularization

    def selective_gate_summary(self) -> list[dict[str, float | bool]]:
        return [gate.summary() for gate in self.gates]

    @staticmethod
    def _data_consistency(
        current: torch.Tensor,
        measurements: torch.Tensor,
        mask: torch.Tensor,
        operator: MatrixFreeChunkedWidebandSAROperator,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = operator.forward_masked(current, mask) - measurements
        gradient = operator.adjoint_masked(residual, mask)
        return current - step[:, None, None] * gradient, gradient

    def forward(
        self,
        measurements1: torch.Tensor,
        measurements2: torch.Tensor,
        operator: MatrixFreeChunkedWidebandSAROperator,
        *,
        mask1: torch.Tensor | None = None,
        mask2: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        coupling_override: CouplingOverride = "selective",
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        if coupling_override not in ("selective", "independent", "always_on"):
            raise ValueError(f"unsupported coupling override: {coupling_override}")
        batch = measurements1.shape[0]
        first_mask = self._mask_batch(mask1, batch, "mask1")
        second_mask = self._mask_batch(mask2, batch, "mask2")
        first = operator.adjoint_masked(measurements1, first_mask)
        second = operator.adjoint_masked(measurements2, second_mask)
        conditioning1 = self.conditioner(first_mask)
        conditioning2 = self.conditioner(second_mask)
        stage_diagnostics: list[dict[str, torch.Tensor]] = []
        gates: list[torch.Tensor] = []
        evidence: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for stage in range(self.stages):
            base_pre_step = 0.35 * F.softplus(self.raw_pre_steps[stage])
            pre_step1 = base_pre_step * (
                1.0 + conditioning1["pre_step_delta"][:, stage]
            )
            pre_step2 = base_pre_step * (
                1.0 + conditioning2["pre_step_delta"][:, stage]
            )
            first_dc, dc_gradient1 = self._data_consistency(
                first, measurements1, first_mask, operator, pre_step1
            )
            second_dc, dc_gradient2 = self._data_consistency(
                second, measurements2, second_mask, operator, pre_step2
            )

            gamma = local_coherence_magnitude(
                first_dc,
                second_dc,
                window_size=self.coherence_window_size,
                eps=self.coherence_eps,
            )
            residual1 = normalized_dc_residual(dc_gradient1, self.coherence_eps)
            residual2 = normalized_dc_residual(dc_gradient2, self.coherence_eps)
            learned_gate = self.gates[stage](gamma, residual1, residual2)
            if coupling_override == "independent":
                gate = torch.zeros_like(learned_gate)
            elif coupling_override == "always_on":
                gate = torch.ones_like(learned_gate)
            else:
                gate = learned_gate
            gates.append(gate)
            evidence.append((gamma, residual1, residual2))

            base_reg = 0.12 * torch.sigmoid(self.raw_regularization[stage])
            reg1 = base_reg * (
                1.0 + conditioning1["regularization_delta"][:, stage]
            )
            reg2 = base_reg * (
                1.0 + conditioning2["regularization_delta"][:, stage]
            )
            first_prior = self._denoise(
                stage,
                first_dc,
                reg1,
                conditioning1["gamma"][:, stage],
                conditioning1["beta"][:, stage],
            )
            second_prior = self._denoise(
                stage,
                second_dc,
                reg2,
                conditioning2["gamma"][:, stage],
                conditioning2["beta"][:, stage],
            )
            correction1, correction2 = self._interaction(
                stage, first_prior, second_prior
            )
            interaction = 0.08 * torch.sigmoid(self.raw_interaction_scale[stage])
            scale = 0.5 * (
                first_prior.abs().mean((-2, -1))
                + second_prior.abs().mean((-2, -1))
            ).clamp_min(1.0e-4)
            first_coupled = first_prior + (
                interaction * scale[:, None, None] * gate[:, 0] * correction1
            )
            second_coupled = second_prior + (
                interaction * scale[:, None, None] * gate[:, 0] * correction2
            )

            base_post_step = 0.35 * F.softplus(self.raw_post_steps[stage])
            post_step1 = base_post_step * (
                1.0 + conditioning1["post_step_delta"][:, stage]
            )
            post_step2 = base_post_step * (
                1.0 + conditioning2["post_step_delta"][:, stage]
            )
            first, post_gradient1 = self._data_consistency(
                first_coupled, measurements1, first_mask, operator, post_step1
            )
            second, post_gradient2 = self._data_consistency(
                second_coupled, measurements2, second_mask, operator, post_step2
            )
            if return_diagnostics:
                stage_diagnostics.append(
                    {
                        "gamma": gamma,
                        "gate": gate,
                        "learned_gate": learned_gate,
                        "dc_residual_epoch1": residual1,
                        "dc_residual_epoch2": residual2,
                        "post_dc_gradient_epoch1": post_gradient1,
                        "post_dc_gradient_epoch2": post_gradient2,
                    }
                )

        self._last_gate_regularization = self._gate_penalties(gates, evidence)
        outputs: list[torch.Tensor] = []
        for value, conditioning in ((first, conditioning1), (second, conditioning2)):
            phase_map = torch.tanh(
                self.phase_refiner(complex_to_channels(value, channel_dim=1))[:, 0]
            )
            delta = 0.08 * conditioning["phase_scale"][:, None, None] * phase_map
            outputs.append(
                value * torch.exp(torch.complex(torch.zeros_like(delta), delta))
            )
        if return_diagnostics:
            return outputs[0], outputs[1], {
                "stages": stage_diagnostics,
                "coupling_override": coupling_override,
            }
        return outputs[0], outputs[1]
