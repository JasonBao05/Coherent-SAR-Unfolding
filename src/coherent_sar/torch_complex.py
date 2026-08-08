"""PyTorch/NumPy complex tensor adapters for DP-JMRNet.

The physical operators use native complex tensors. Neural priors may expose
real/imaginary channels with shape ``[..., 2, H, W]``; conversions are kept in
this module so the convention cannot silently drift between components.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def numpy_complex_to_torch(
    values: np.ndarray,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a NumPy complex array to a native PyTorch complex tensor."""

    array = np.asarray(values)
    if not np.iscomplexobj(array):
        raise TypeError("values must be a NumPy complex array")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("dtype must be torch.complex64 or torch.complex128")
    return torch.as_tensor(array, dtype=dtype, device=device)


def torch_complex_to_numpy(values: torch.Tensor) -> np.ndarray:
    """Detach a native PyTorch complex tensor and return a NumPy array."""

    if not torch.is_complex(values):
        raise TypeError("values must be a native PyTorch complex tensor")
    return values.detach().cpu().numpy()


def complex_to_channels(values: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert ``[..., H, W]`` complex data to real/imaginary channels.

    With the default ``channel_dim=-3``, an input ``[B,H,W]`` becomes
    ``[B,2,H,W]``. Channel 0 is real and channel 1 is imaginary.
    """

    if not torch.is_complex(values):
        raise TypeError("values must be complex")
    return torch.stack((values.real, values.imag), dim=channel_dim)


def channels_to_complex(values: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Convert explicit real/imaginary channels back to a complex tensor."""

    if torch.is_complex(values):
        raise TypeError("channel representation must be real-valued")
    normalized_dim = channel_dim if channel_dim >= 0 else values.ndim + channel_dim
    if normalized_dim < 0 or normalized_dim >= values.ndim:
        raise ValueError("channel_dim is outside the tensor rank")
    if values.shape[normalized_dim] != 2:
        raise ValueError("real/imaginary channel dimension must have length 2")
    real = values.select(normalized_dim, 0)
    imaginary = values.select(normalized_dim, 1)
    return torch.complex(real, imaginary)


def complex_record(value: complex | torch.Tensor | np.generic[Any]) -> dict[str, float]:
    """Return a JSON-safe real/imaginary record for one complex scalar."""

    if isinstance(value, torch.Tensor):
        scalar = complex(value.detach().cpu().item())
    else:
        scalar = complex(value)
    return {"real": float(scalar.real), "imag": float(scalar.imag)}
