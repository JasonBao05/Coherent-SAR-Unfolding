"""Guarded access helpers for the final 256x256 simulated SAR dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FINAL_TEST_ENVIRONMENT_VARIABLE = "COHERENT_SAR_ENABLE_FINAL_TEST"
FINAL_TEST_CONFIRMATION = "I_UNDERSTAND_THIS_OPENS_THE_SEALED_FINAL_TEST"


class FinalSARDataset256:
    """Manifest-backed truth loader with a sealed-test hard access gate."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        with (self.root / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.records = {
            split: [record for record in self.manifest["samples"] if record["split"] == split]
            for split in ("train", "validation", "test")
        }
        self.files_opened = {"train": 0, "validation": 0, "test": 0}

    def __len__(self) -> int:
        return sum(len(records) for records in self.records.values())

    def load_sample(
        self,
        split: str,
        index: int,
        *,
        allow_final_test: bool = False,
        confirmation: str | None = None,
    ) -> dict[str, np.ndarray]:
        if split not in self.records:
            raise ValueError(f"unknown split: {split}")
        if split == "test":
            enabled = os.environ.get(FINAL_TEST_ENVIRONMENT_VARIABLE) == FINAL_TEST_CONFIRMATION
            if not (allow_final_test and confirmation == FINAL_TEST_CONFIRMATION and enabled):
                raise PermissionError(
                    "sealed test access denied; explicit final-test argument, confirmation, "
                    "and environment switch are all required"
                )
        record = self.records[split][int(index)]
        path = self.root / record["relative_path"]
        with np.load(path, allow_pickle=False) as archive:
            sample = {name: archive[name] for name in archive.files}
        self.files_opened[split] += 1
        return sample


def full_grid_noise_direction(
    seed: int,
    aperture_count: int,
    frequency_count: int,
) -> np.ndarray:
    """Materialize one deterministic unit-power complex full-grid noise direction."""

    rng = np.random.default_rng(int(seed))
    real = rng.standard_normal((int(aperture_count), int(frequency_count)))
    imaginary = rng.standard_normal((int(aperture_count), int(frequency_count)))
    value = (real + 1j * imaginary) / np.sqrt(2.0)
    power = float(np.mean(np.abs(value) ** 2))
    return np.asarray(value / np.sqrt(max(power, 1.0e-12)), dtype=np.complex64)


def read_noise_seed_bank(root: str | Path, split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise PermissionError("sealed test noise bank is unavailable to default code")
    path = Path(root) / "noise_banks" / f"{split}_noise_seeds.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
