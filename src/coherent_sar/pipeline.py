"""Compact training and validation pipeline for DP-JMRNet."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .equivariant_selective_model import (
    FixedMaskExchangeEquivariantSelectiveUnfolding256,
)
from .formal_wideband_operator import (
    MatrixFreeChunkedWidebandSAROperator,
    simulate_wideband_observation,
)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_random_state(value: int) -> None:
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class SimulatedSARDataset:
    """Manifest-backed loader for the included complex SAR dataset."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest = read_json(self.root / "manifest.json")
        self.records = {
            split: [row for row in manifest["samples"] if row["split"] == split]
            for split in ("train", "validation")
        }
        if not self.records["train"] or not self.records["validation"]:
            raise RuntimeError("the dataset must contain train and validation samples")

    def __len__(self) -> int:
        return len(self.records["train"]) + len(self.records["validation"])

    def _load_sample(self, split: str, index: int) -> dict[str, np.ndarray]:
        record = self.records[split][int(index)]
        with np.load(self.root / record["relative_path"], allow_pickle=False) as archive:
            return {
                "x1": np.asarray(archive["x1"], dtype=np.complex64),
                "x2": np.asarray(archive["x2"], dtype=np.complex64),
                "weight": np.asarray(archive["phase_weight"], dtype=np.float32),
                "hard": np.asarray(archive["hard_valid_mask"], dtype=np.bool_),
            }

    def batch(
        self,
        split: str,
        indices: list[int],
        device: torch.device,
        snr_db: float | np.ndarray,
        noise_state: int,
    ) -> dict[str, torch.Tensor]:
        samples = [self._load_sample(split, index) for index in indices]
        output = {
            key: torch.as_tensor(np.stack([sample[key] for sample in samples]), device=device)
            for key in ("x1", "x2", "weight", "hard")
        }
        levels = np.asarray(snr_db, dtype=np.float32)
        if levels.ndim == 0:
            levels = np.full(len(indices), float(levels), dtype=np.float32)
        output["snr"] = torch.as_tensor(levels, dtype=torch.float32, device=device)
        generator = torch.Generator(device=device).manual_seed(int(noise_state))
        shape = (len(indices), 256, 256)
        for name in ("noise1", "noise2"):
            output[name] = torch.complex(
                torch.randn(shape, generator=generator, device=device),
                torch.randn(shape, generator=generator, device=device),
            ) / math.sqrt(2.0)
        return output


def fixed_mask(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    sampling = config["sampling"]
    mask = torch.zeros(int(sampling["aperture_candidates"]), device=device)
    indices = torch.as_tensor(sampling["mask_indices"], dtype=torch.long, device=device)
    mask[indices] = 1.0
    if int(mask.sum().item()) != int(sampling["sample_count"]):
        raise ValueError("mask indices do not match the configured sampling count")
    return mask


def build_model(
    config: dict[str, Any], device: torch.device
) -> FixedMaskExchangeEquivariantSelectiveUnfolding256:
    values = config["model"]
    return FixedMaskExchangeEquivariantSelectiveUnfolding256(
        fixed_mask(config, device),
        aperture_count=int(config["sampling"]["aperture_candidates"]),
        stages=int(values["stages"]),
        hidden_channels=int(values["hidden_channels"]),
        coherence_window_size=int(values["coherence_window_size"]),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
    ).to(device)


def observations(
    model: FixedMaskExchangeEquivariantSelectiveUnfolding256,
    operator: MatrixFreeChunkedWidebandSAROperator,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    first, _ = simulate_wideband_observation(
        operator, batch["x1"], model.fixed_mask, batch["noise1"], batch["snr"]
    )
    second, _ = simulate_wideband_observation(
        operator, batch["x2"], model.fixed_mask, batch["noise2"], batch["snr"]
    )
    return first, second


def circular_error(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.angle(
        torch.exp(torch.complex(torch.zeros_like(estimate), estimate - truth))
    )


def reconstruction_loss(
    estimate1: torch.Tensor,
    estimate2: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    truth1, truth2 = batch["x1"], batch["x2"]
    denominator = (
        truth1.abs().square().sum((-2, -1))
        + truth2.abs().square().sum((-2, -1))
    ).clamp_min(1.0e-8)
    complex_loss = (
        (estimate1 - truth1).abs().square().sum((-2, -1))
        + (estimate2 - truth2).abs().square().sum((-2, -1))
    ) / denominator
    amplitude_loss = (
        (estimate1.abs() - truth1.abs()).square().sum((-2, -1))
        + (estimate2.abs() - truth2.abs()).square().sum((-2, -1))
    ) / denominator
    weight = batch["weight"]
    weight_sum = weight.sum((-2, -1)).clamp_min(1.0e-8)
    raw1 = circular_error(torch.angle(estimate1), torch.angle(truth1))
    raw2 = circular_error(torch.angle(estimate2), torch.angle(truth2))
    raw_phase = torch.sqrt(
        0.5
        * (
            (weight * raw1.square()).sum((-2, -1)) / weight_sum
            + (weight * raw2.square()).sum((-2, -1)) / weight_sum
        )
        + 1.0e-12
    )
    differential = circular_error(
        torch.angle(estimate2 * estimate1.conj()),
        torch.angle(truth2 * truth1.conj()),
    )
    differential_phase = torch.sqrt(
        (weight * differential.square()).sum((-2, -1)) / weight_sum + 1.0e-12
    )
    values = {
        "complex": complex_loss.mean(),
        "amplitude": amplitude_loss.mean(),
        "raw_phase": raw_phase.mean(),
        "differential_phase": differential_phase.mean(),
    }
    weights = config["loss"]
    values["total"] = (
        float(weights["complex_weight"]) * values["complex"]
        + float(weights["amplitude_weight"]) * values["amplitude"]
        + float(weights["raw_phase_weight"]) * values["raw_phase"]
        + float(weights["differential_phase_weight"]) * values["differential_phase"]
    )
    return values


def add_gate_regularization(
    loss: torch.Tensor,
    model: FixedMaskExchangeEquivariantSelectiveUnfolding256,
    config: dict[str, Any],
) -> torch.Tensor:
    values = model.gate_regularization()
    weights = config["loss"]
    return (
        loss
        + float(weights["gate_smoothness_weight"]) * values["smoothness"]
        + float(weights["gate_anti_collapse_weight"]) * values["anti_collapse"]
    )


@torch.no_grad()
def validate_model(
    model: FixedMaskExchangeEquivariantSelectiveUnfolding256,
    operator: MatrixFreeChunkedWidebandSAROperator,
    dataset: SimulatedSARDataset,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    batch_size = int(config["validation"]["batch_size"])
    indices = list(range(len(dataset.records["validation"])))
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = dataset.batch(
            "validation",
            selected,
            device,
            float(config["validation"]["snr_db"]),
            int(config["random_seed"]) + 100_000 + start,
        )
        first, second = observations(model, operator, batch)
        estimate1, estimate2 = model(first, second, operator)
        truth1, truth2 = batch["x1"], batch["x2"]
        denominator = (
            truth1.abs().square().sum((-2, -1))
            + truth2.abs().square().sum((-2, -1))
        ).clamp_min(1.0e-12)
        complex_nmse = (
            (estimate1 - truth1).abs().square().sum((-2, -1))
            + (estimate2 - truth2).abs().square().sum((-2, -1))
        ) / denominator
        amplitude_nmse = (
            (estimate1.abs() - truth1.abs()).square().sum((-2, -1))
            + (estimate2.abs() - truth2.abs()).square().sum((-2, -1))
        ) / denominator
        weight = batch["weight"]
        phase_error = circular_error(
            torch.angle(estimate2 * estimate1.conj()),
            torch.angle(truth2 * truth1.conj()),
        )
        phase_rmse = torch.sqrt(
            (weight * phase_error.square()).sum((-2, -1))
            / weight.sum((-2, -1)).clamp_min(1.0e-12)
        )
        for index in range(len(selected)):
            rows.append(
                {
                    "differential_phase_rmse_degree": math.degrees(
                        float(phase_rmse[index].cpu())
                    ),
                    "amplitude_nmse": float(amplitude_nmse[index].cpu()),
                    "complex_nmse": float(complex_nmse[index].cpu()),
                }
            )
    amplitude = float(np.mean([row["amplitude_nmse"] for row in rows]))
    complex_value = float(np.mean([row["complex_nmse"] for row in rows]))
    return {
        "validation_samples": len(rows),
        "differential_phase_rmse_degree": float(
            np.mean([row["differential_phase_rmse_degree"] for row in rows])
        ),
        "amplitude_nmse_db": 10.0 * math.log10(max(amplitude, 1.0e-12)),
        "complex_nmse_db": 10.0 * math.log10(max(complex_value, 1.0e-12)),
    }


def train(
    config_path: str | Path,
    *,
    steps: int | None = None,
    device_name: str = "auto",
) -> dict[str, float]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = read_json(config_path)
    device = resolve_device(device_name)
    state = int(config["random_seed"])
    set_random_state(state)
    dataset = SimulatedSARDataset(root / config["data_directory"])
    model = build_model(config, device)
    operator = MatrixFreeChunkedWidebandSAROperator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = int(steps or config["training"]["steps"])
    if total_steps <= 0:
        raise ValueError("training steps must be positive")
    rng = np.random.default_rng(state)
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    best_score = math.inf
    best_metrics: dict[str, float] = {}
    for step in range(1, total_steps + 1):
        model.train()
        batch_size = int(config["training"]["batch_size"])
        selected = rng.integers(
            0, len(dataset.records["train"]), size=batch_size
        ).tolist()
        low, high = map(float, config["training"]["snr_db_range"])
        levels = rng.uniform(low, high, size=batch_size).astype(np.float32)
        batch = dataset.batch("train", selected, device, levels, state + step)
        measurement1, measurement2 = observations(model, operator, batch)
        estimate1, estimate2 = model(measurement1, measurement2, operator)
        losses = reconstruction_loss(estimate1, estimate2, batch, config)
        total = add_gate_regularization(losses["total"], model, config)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
        if step == 1 or step % int(config["training"]["log_interval"]) == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(total.detach().cpu()),
                        "differential_phase_loss": float(
                            losses["differential_phase"].detach().cpu()
                        ),
                    }
                ),
                flush=True,
            )
        should_validate = (
            step == total_steps
            or step % int(config["training"]["validation_interval"]) == 0
        )
        if should_validate:
            metrics = validate_model(model, operator, dataset, config, device)
            print(json.dumps({"validation": metrics}), flush=True)
            score = metrics["differential_phase_rmse_degree"]
            if score < best_score:
                best_score = score
                best_metrics = metrics
                torch.save(
                    {"model_state": model.state_dict(), "step": step},
                    output / "best.pt",
                )
    torch.save({"model_state": model.state_dict(), "step": total_steps}, output / "last.pt")
    write_json(output / "metrics.json", best_metrics)
    return best_metrics


def validate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, float]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = read_json(config_path)
    device = resolve_device(device_name)
    set_random_state(int(config["random_seed"]))
    dataset = SimulatedSARDataset(root / config["data_directory"])
    model = build_model(config, device)
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    operator = MatrixFreeChunkedWidebandSAROperator().to(device)
    return validate_model(model, operator, dataset, config, device)
