"""Geometry-defined matrix-free wideband SAR operator for 256 x 256 experiments.

The production operator implements the monostatic far-field phase-history model
after reference-range demodulation.  Platform positions and stepped frequencies
are converted to two-dimensional spatial frequencies, and the resulting
non-Cartesian Fourier samples are evaluated with an explicitly adjoint bilinear
gridding pair.  No dense system matrix is constructed.

For a ground-plane reflectivity ``x(r)`` the discretized forward model is

    y[m, f] = S_x(q[m, f]),
    q[m, f] = 4 pi / c * (f u_m - f0 u_0),

where ``u_m`` is the unit vector from platform position ``m`` to the scene
centre, restricted to the scene plane.  ``S_x`` is the centred spatial Fourier
transform of the reflectivity.  This is the standard planar-wave/far-field
form of the monostatic wideband phase history; it is not a near-field spherical
wave model.  The approximation and all geometry parameters are intentionally
exposed in :class:`WidebandSARGroundPlaneGeometry`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class WidebandSARGroundPlaneGeometry:
    """Frozen geometry for the final simulated-paper acquisition protocol."""

    image_height: int = 256
    image_width: int = 256
    pixel_spacing_ground_range_m: float = 0.5
    pixel_spacing_cross_range_m: float = 0.5
    aperture_count: int = 256
    frequency_count: int = 256
    center_frequency_hz: float = 10.0e9
    bandwidth_hz: float = 400.0e6
    aperture_length_m: float = 4.0
    ground_range_to_scene_center_m: float = 100.0
    platform_altitude_m: float = 100.0
    propagation_speed_m_per_s: float = 299_792_458.0
    aperture_chunk_size: int = 32
    interpolation: str = "bilinear-adjoint-pair"
    propagation_model: str = "monostatic-far-field-reference-demodulated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    if complex_dtype == torch.complex128:
        return torch.float64
    if complex_dtype == torch.complex64:
        return torch.float32
    raise TypeError(f"expected complex64 or complex128, received {complex_dtype}")


class MatrixFreeChunkedWidebandSAROperator(nn.Module):
    """Matrix-free non-Cartesian wideband forward/adjoint pair.

    Bilinear interpolation weights and four neighbouring FFT-grid indices are
    precomputed as small geometry buffers.  Image-dependent phase histories are
    evaluated in aperture chunks, so the implementation never materializes an
    ``(N_aperture * N_frequency) x (H * W)`` system matrix.
    """

    def __init__(self, geometry: WidebandSARGroundPlaneGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or WidebandSARGroundPlaneGeometry()
        geometry = self.geometry
        if min(geometry.image_height, geometry.image_width) < 2:
            raise ValueError("image dimensions must be at least two")
        if geometry.aperture_chunk_size < 1:
            raise ValueError("aperture_chunk_size must be positive")

        frequency = torch.linspace(
            geometry.center_frequency_hz - 0.5 * geometry.bandwidth_hz,
            geometry.center_frequency_hz + 0.5 * geometry.bandwidth_hz,
            geometry.frequency_count,
            dtype=torch.float64,
        )
        aperture_x = torch.linspace(
            -0.5 * geometry.aperture_length_m,
            0.5 * geometry.aperture_length_m,
            geometry.aperture_count,
            dtype=torch.float64,
        )

        # Platform is [cross-range x, -ground-range, altitude].  The scene
        # centre is the origin of the z=0 ground plane.
        ground_range = torch.full_like(aperture_x, geometry.ground_range_to_scene_center_m)
        altitude = torch.full_like(aperture_x, -geometry.platform_altitude_m)
        slant = torch.sqrt(aperture_x.square() + ground_range.square() + altitude.square())
        look_cross = -aperture_x / slant
        look_ground = ground_range / slant
        reference_slant = math.hypot(
            geometry.ground_range_to_scene_center_m, geometry.platform_altitude_m
        )
        reference_look_ground = geometry.ground_range_to_scene_center_m / reference_slant

        scale = 4.0 * math.pi / geometry.propagation_speed_m_per_s
        q_cross = scale * frequency[None, :] * look_cross[:, None]
        q_ground = scale * (
            frequency[None, :] * look_ground[:, None]
            - geometry.center_frequency_hz * reference_look_ground
        )

        delta_q_cross = 2.0 * math.pi / (
            geometry.image_width * geometry.pixel_spacing_cross_range_m
        )
        delta_q_ground = 2.0 * math.pi / (
            geometry.image_height * geometry.pixel_spacing_ground_range_m
        )
        column_coordinate = q_cross / delta_q_cross + geometry.image_width // 2
        row_coordinate = q_ground / delta_q_ground + geometry.image_height // 2
        tolerance = 1.0e-7
        if (
            float(column_coordinate.min()) < -tolerance
            or float(column_coordinate.max()) > geometry.image_width - 1 + tolerance
            or float(row_coordinate.min()) < -tolerance
            or float(row_coordinate.max()) > geometry.image_height - 1 + tolerance
        ):
            raise ValueError(
                "acquisition spatial frequencies exceed the image Nyquist grid: "
                f"column=[{float(column_coordinate.min()):.3f}, "
                f"{float(column_coordinate.max()):.3f}], "
                f"row=[{float(row_coordinate.min()):.3f}, "
                f"{float(row_coordinate.max()):.3f}]"
            )
        column_coordinate = column_coordinate.clamp(0.0, geometry.image_width - 1 - 1.0e-7)
        row_coordinate = row_coordinate.clamp(0.0, geometry.image_height - 1 - 1.0e-7)

        column0 = torch.floor(column_coordinate).to(torch.int64)
        row0 = torch.floor(row_coordinate).to(torch.int64)
        column1 = column0 + 1
        row1 = row0 + 1
        fraction_column = column_coordinate - column0
        fraction_row = row_coordinate - row0
        indices = torch.stack(
            (
                row0 * geometry.image_width + column0,
                row0 * geometry.image_width + column1,
                row1 * geometry.image_width + column0,
                row1 * geometry.image_width + column1,
            ),
            dim=-1,
        )
        weights = torch.stack(
            (
                (1.0 - fraction_row) * (1.0 - fraction_column),
                (1.0 - fraction_row) * fraction_column,
                fraction_row * (1.0 - fraction_column),
                fraction_row * fraction_column,
            ),
            dim=-1,
        )

        # fft2 assumes sample coordinates start at zero.  This phase translates
        # the transform to the centred physical scene coordinates.
        centre_cross = 0.5 * (geometry.image_width - 1) * geometry.pixel_spacing_cross_range_m
        centre_ground = 0.5 * (geometry.image_height - 1) * geometry.pixel_spacing_ground_range_m
        translation_phase = torch.exp(
            1j * (q_cross * centre_cross + q_ground * centre_ground)
        )

        self.register_buffer("frequencies_hz", frequency, persistent=True)
        self.register_buffer("aperture_positions_cross_range_m", aperture_x, persistent=True)
        self.register_buffer("spatial_frequency_cross_range_rad_m", q_cross, persistent=True)
        self.register_buffer("spatial_frequency_ground_range_rad_m", q_ground, persistent=True)
        self.register_buffer("grid_indices", indices, persistent=True)
        self.register_buffer("grid_weights", weights, persistent=True)
        self.register_buffer("translation_phase", translation_phase, persistent=True)

    @property
    def image_shape(self) -> tuple[int, int]:
        return (self.geometry.image_height, self.geometry.image_width)

    @property
    def aperture_count(self) -> int:
        return self.geometry.aperture_count

    @property
    def frequency_count(self) -> int:
        return self.geometry.frequency_count

    def _validate_image(self, image: torch.Tensor) -> None:
        if not torch.is_complex(image):
            raise TypeError("SAR reflectivity must be complex")
        if tuple(image.shape[-2:]) != self.image_shape:
            raise ValueError(f"expected image shape {self.image_shape}, received {tuple(image.shape[-2:])}")

    def _validate_history(self, history: torch.Tensor) -> None:
        if not torch.is_complex(history):
            raise TypeError("SAR phase history must be complex")
        expected = (self.aperture_count, self.frequency_count)
        if tuple(history.shape[-2:]) != expected:
            raise ValueError(f"expected phase-history shape {expected}, received {tuple(history.shape[-2:])}")

    @staticmethod
    def _mask_batch(mask: torch.Tensor, batch: int, aperture_count: int) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        if tuple(mask.shape) != (batch, aperture_count):
            raise ValueError(
                f"mask must have shape [{aperture_count}] or [{batch}, {aperture_count}], "
                f"received {tuple(mask.shape)}"
            )
        return mask

    def forward_full(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_image(image)
        squeeze = image.ndim == 2
        if squeeze:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError("image must be [H,W] or [B,H,W]")
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(image, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
        ).reshape(image.shape[0], -1)
        output = image.new_empty(
            (image.shape[0], self.aperture_count, self.frequency_count)
        )
        real_dtype = _real_dtype(image.dtype)
        for start in range(0, self.aperture_count, self.geometry.aperture_chunk_size):
            stop = min(start + self.geometry.aperture_chunk_size, self.aperture_count)
            indices = self.grid_indices[start:stop].reshape(-1, 4)
            weights = self.grid_weights[start:stop].reshape(-1, 4).to(real_dtype)
            phase = self.translation_phase[start:stop].reshape(-1).to(image.dtype)
            gathered = spectrum[:, indices]
            samples = (gathered * weights.unsqueeze(0)).sum(dim=-1) * phase.unsqueeze(0)
            output[:, start:stop] = samples.reshape(
                image.shape[0], stop - start, self.frequency_count
            )
        return output[0] if squeeze else output

    def adjoint_full(self, phase_history: torch.Tensor) -> torch.Tensor:
        self._validate_history(phase_history)
        squeeze = phase_history.ndim == 2
        if squeeze:
            phase_history = phase_history.unsqueeze(0)
        if phase_history.ndim != 3:
            raise ValueError("phase history must be [N,F] or [B,N,F]")
        batch = phase_history.shape[0]
        flat = phase_history.new_zeros((batch, self.geometry.image_height * self.geometry.image_width))
        real_dtype = _real_dtype(phase_history.dtype)
        for start in range(0, self.aperture_count, self.geometry.aperture_chunk_size):
            stop = min(start + self.geometry.aperture_chunk_size, self.aperture_count)
            indices = self.grid_indices[start:stop].reshape(-1, 4)
            weights = self.grid_weights[start:stop].reshape(-1, 4).to(real_dtype)
            phase = self.translation_phase[start:stop].reshape(-1).to(phase_history.dtype)
            samples = phase_history[:, start:stop].reshape(batch, -1) * phase.conj().unsqueeze(0)
            contributions = samples.unsqueeze(-1) * weights.unsqueeze(0)
            for neighbour in range(4):
                flat.scatter_add_(
                    1,
                    indices[:, neighbour].unsqueeze(0).expand(batch, -1),
                    contributions[:, :, neighbour],
                )
        grid = flat.reshape(batch, *self.image_shape)
        image = torch.fft.ifft2(
            torch.fft.ifftshift(grid, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
        )
        return image[0] if squeeze else image

    def forward_masked(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        history = self.forward_full(image)
        squeeze = history.ndim == 2
        if squeeze:
            history = history.unsqueeze(0)
        mask_batch = self._mask_batch(mask, history.shape[0], self.aperture_count)
        result = history * mask_batch.to(history.real.dtype).unsqueeze(-1)
        return result[0] if squeeze else result

    def adjoint_masked(self, phase_history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeeze = phase_history.ndim == 2
        history = phase_history.unsqueeze(0) if squeeze else phase_history
        mask_batch = self._mask_batch(mask, history.shape[0], self.aperture_count)
        result = self.adjoint_full(history * mask_batch.to(history.real.dtype).unsqueeze(-1))
        return result[0] if squeeze else result

    def geometry_summary(self) -> dict[str, Any]:
        geometry = self.geometry.to_dict()
        geometry.update(
            {
                "measurement_shape": [self.aperture_count, self.frequency_count],
                "dense_system_matrix": False,
                "matrix_free": True,
                "chunked": True,
                "q_cross_range_rad_m": [
                    float(self.spatial_frequency_cross_range_rad_m.min()),
                    float(self.spatial_frequency_cross_range_rad_m.max()),
                ],
                "q_ground_range_rad_m": [
                    float(self.spatial_frequency_ground_range_rad_m.min()),
                    float(self.spatial_frequency_ground_range_rad_m.max()),
                ],
            }
        )
        return geometry


def realized_masked_snr_db(
    clean: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return per-sample SNR using only actually observed phase-history cells."""

    squeeze = clean.ndim == 2
    if squeeze:
        clean = clean.unsqueeze(0)
        noise = noise.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(clean.shape[0], -1)
    selected = mask.to(clean.real.dtype).unsqueeze(-1)
    denominator = (selected.sum(dim=(-2, -1)) * clean.shape[-1]).clamp_min(1.0)
    signal_power = (clean.abs().square() * selected).sum(dim=(-2, -1)) / denominator
    noise_power = (noise.abs().square() * selected).sum(dim=(-2, -1)) / denominator
    result = 10.0 * torch.log10(signal_power / noise_power.clamp_min(1.0e-20))
    return result[0] if squeeze else result


def simulate_wideband_observation(
    operator: MatrixFreeChunkedWidebandSAROperator,
    image: torch.Tensor,
    mask: torch.Tensor,
    noise_direction: torch.Tensor,
    snr_db: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the physical mask and deterministic complex noise at requested SNR."""

    clean = operator.forward_masked(image, mask)
    squeeze = clean.ndim == 2
    if squeeze:
        clean = clean.unsqueeze(0)
        noise_direction = noise_direction.unsqueeze(0)
    if mask.ndim == 1:
        mask_batch = mask.unsqueeze(0).expand(clean.shape[0], -1)
    else:
        mask_batch = mask
    selected = mask_batch.to(clean.real.dtype).unsqueeze(-1)
    direction = noise_direction.to(clean.dtype) * selected
    denominator = (selected.sum(dim=(-2, -1)) * clean.shape[-1]).clamp_min(1.0)
    clean_power = clean.abs().square().sum(dim=(-2, -1)) / denominator
    direction_power = direction.abs().square().sum(dim=(-2, -1)) / denominator
    target = torch.as_tensor(snr_db, dtype=clean.real.dtype, device=clean.device)
    if target.ndim == 0:
        target = target.expand(clean.shape[0])
    scale = torch.sqrt(
        clean_power / direction_power.clamp_min(1.0e-20) / torch.pow(10.0, target / 10.0)
    )
    noise = scale[:, None, None] * direction
    noisy = clean + noise
    realized = realized_masked_snr_db(clean, noise, mask_batch)
    if squeeze:
        return noisy[0], realized[0]
    return noisy, realized
