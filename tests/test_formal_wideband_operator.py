"""Numerical release gates for the final matrix-free wideband SAR operator."""

from __future__ import annotations

import math

import numpy as np
import torch

from coherent_sar.formal_wideband_operator import (
    MatrixFreeChunkedWidebandSAROperator,
    WidebandSARGroundPlaneGeometry,
    simulate_wideband_observation,
)


def _complex_random(shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    return torch.complex(
        torch.randn(shape, generator=generator, dtype=real_dtype),
        torch.randn(shape, generator=generator, dtype=real_dtype),
    ) / math.sqrt(2.0)


def test_formal_wideband_adjoint_complex128() -> None:
    geometry = WidebandSARGroundPlaneGeometry(
        image_height=16,
        image_width=16,
        pixel_spacing_ground_range_m=0.5,
        pixel_spacing_cross_range_m=0.5,
        aperture_count=12,
        frequency_count=10,
        aperture_length_m=0.8,
        bandwidth_hz=80.0e6,
        aperture_chunk_size=5,
    )
    operator = MatrixFreeChunkedWidebandSAROperator(geometry)
    image = _complex_random((2, 16, 16), torch.complex128, 17)
    history = _complex_random((2, 12, 10), torch.complex128, 19)
    left = (operator.forward_full(image).conj() * history).sum()
    right = (image.conj() * operator.adjoint_full(history)).sum()
    relative = float((left - right).abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1.0e-20))
    assert relative < 5.0e-13


def test_formal_wideband_masked_adjoint_and_gradients() -> None:
    geometry = WidebandSARGroundPlaneGeometry(
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
    operator = MatrixFreeChunkedWidebandSAROperator(geometry)
    image = _complex_random((1, 32, 32), torch.complex64, 23).requires_grad_(True)
    history = _complex_random((1, 24, 20), torch.complex64, 29)
    mask = torch.zeros(24, dtype=torch.float32)
    mask[torch.tensor([0, 2, 5, 8, 11, 14, 17, 20, 23])] = 1.0
    left = (operator.forward_masked(image, mask).conj() * history).sum()
    right = (image.conj() * operator.adjoint_masked(history, mask)).sum()
    relative = float((left - right).abs() / torch.maximum(left.abs(), right.abs()).clamp_min(1.0e-12))
    assert relative < 2.0e-5
    loss = operator.forward_masked(image, mask).abs().square().mean()
    loss.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad.real).all() and torch.isfinite(image.grad.imag).all()
    assert float(image.grad.abs().sum()) > 0.0


def test_formal_wideband_shape_stability_focus_and_snr() -> None:
    operator = MatrixFreeChunkedWidebandSAROperator()
    image = torch.zeros((2, 256, 256), dtype=torch.complex64)
    image[:, 128, 128] = torch.tensor(1.0 + 0.5j)
    mask = torch.zeros(256, dtype=torch.float32)
    mask[torch.linspace(0, 255, 104).round().long()] = 1.0
    noise = _complex_random((2, 256, 256), torch.complex64, 31)
    history, realized = simulate_wideband_observation(operator, image, mask, noise, 20.0)
    reconstruction = operator.adjoint_masked(history, mask)
    assert history.shape == (2, 256, 256)
    assert reconstruction.shape == image.shape
    assert torch.isfinite(history.real).all() and torch.isfinite(history.imag).all()
    assert torch.isfinite(reconstruction.real).all() and torch.isfinite(reconstruction.imag).all()
    assert np.allclose(realized.detach().cpu().numpy(), 20.0, atol=2.0e-4)
    peak = torch.argmax(reconstruction[0].abs()).item()
    assert np.unravel_index(peak, (256, 256)) == (128, 128)


def test_geometry_stays_inside_nyquist_and_has_no_dense_matrix() -> None:
    operator = MatrixFreeChunkedWidebandSAROperator()
    summary = operator.geometry_summary()
    nyquist = math.pi / 0.5
    assert max(abs(v) for v in summary["q_cross_range_rad_m"]) < nyquist
    assert max(abs(v) for v in summary["q_ground_range_rad_m"]) < nyquist
    assert summary["dense_system_matrix"] is False
    dense_elements = 256 * 256 * 256 * 256
    assert operator.grid_indices.numel() == 256 * 256 * 4
    assert operator.grid_indices.numel() < dense_elements // 1000
