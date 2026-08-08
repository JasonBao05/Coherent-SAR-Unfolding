"""Neural components and physics-facing features used by DP-JMRNet."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .torch_complex import channels_to_complex, complex_to_channels


def _local_mean(value: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("local coherence window must be a positive odd integer")
    padding = window_size // 2
    if padding:
        value = F.pad(value, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(value, kernel_size=window_size, stride=1)


def local_coherence_magnitude(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    window_size: int = 7,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Estimate local coherence magnitude without exposing differential phase."""

    if not torch.is_complex(first) or not torch.is_complex(second):
        raise TypeError("local coherence inputs must be complex")
    cross = first * second.conj()
    cross_real = _local_mean(cross.real.unsqueeze(1), window_size)
    cross_imag = _local_mean(cross.imag.unsqueeze(1), window_size)
    first_power = _local_mean(first.abs().square().unsqueeze(1), window_size)
    second_power = _local_mean(second.abs().square().unsqueeze(1), window_size)
    numerator = torch.sqrt(cross_real.square() + cross_imag.square() + eps)
    denominator = torch.sqrt(first_power * second_power + eps)
    return (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)


def normalized_dc_residual(
    dc_gradient: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Map an image-domain data-consistency correction to a stable [0, 1] map."""

    magnitude = dc_gradient.abs().unsqueeze(1)
    rms = torch.sqrt(magnitude.square().mean((-2, -1), keepdim=True) + eps)
    relative = magnitude / rms.clamp_min(eps)
    return relative / (1.0 + relative)


class FeatureBranch(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ComplexStageDenoiser(nn.Module):
    """Real/imaginary residual CNN with optional FiLM conditioning."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(2, channels, 3, padding=1)
        self.hidden = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.output = nn.Conv2d(channels, 2, 3, padding=1)

    def forward(
        self,
        values: torch.Tensor,
        regularization: torch.Tensor,
        gamma: torch.Tensor | None,
        beta: torch.Tensor | None,
    ) -> torch.Tensor:
        features = F.silu(self.input(complex_to_channels(values, channel_dim=1)))
        if gamma is not None and beta is not None:
            features = features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        residual = self.output(F.silu(self.hidden(features)))
        complex_residual = channels_to_complex(residual, channel_dim=1)
        return values + regularization[:, None, None] * complex_residual

