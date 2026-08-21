"""Generate the final 256x256 parent-first controlled SAR simulation dataset.

No neural network or baseline is imported or executed.  Test samples are
created and hashed by this generator, but the guarded training-style loader is
never permitted to open them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from coherent_sar.final_dataset_256 import FinalSARDataset256


CONFIG_PATH = ROOT / "configs" / "data" / "final_sar_dataset_256_v1.json"
SPLITS = ("train", "validation", "test")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_truth_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("x1", "x2", "delta_phi_gt", "phase_weight", "hard_valid_mask"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64)
        self.count += int(array.size)
        self.total += float(array.sum(dtype=np.float64))
        self.square_total += float(np.square(array).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self) -> dict[str, float | int]:
        mean = self.total / max(self.count, 1)
        variance = max(self.square_total / max(self.count, 1) - mean * mean, 0.0)
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
        }


def inclusive_integer(rng: np.random.Generator, bounds: list[int]) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def normalized_parent_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = shape
    y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, columns, dtype=np.float32)
    return np.meshgrid(y, x, indexing="ij")


def distance_to_segment(
    yy: np.ndarray,
    xx: np.ndarray,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
) -> np.ndarray:
    dy, dx = y1 - y0, x1 - x0
    denominator = max(dy * dy + dx * dx, 1.0e-12)
    projection = np.clip(((yy - y0) * dy + (xx - x0) * dx) / denominator, 0.0, 1.0)
    return np.sqrt(np.square(yy - (y0 + projection * dy)) + np.square(xx - (x0 + projection * dx)))


def render_parent_scene(
    config: dict[str, Any], parent_seed: int, yy: np.ndarray, xx: np.ndarray
) -> dict[str, Any]:
    """Directly simulate one native 1024x1024 coherent complex parent scene."""

    rng = np.random.default_rng(int(parent_seed))
    scene = config["scene"]
    names = list(scene["difficulty_probabilities"])
    probabilities = [float(scene["difficulty_probabilities"][name]) for name in names]
    difficulty = str(rng.choice(names, p=probabilities))
    difficulty_cfg = scene["difficulty"][difficulty]

    amplitude = np.full(yy.shape, float(scene["amplitude_floor"]), dtype=np.float32)
    for _ in range(6):
        angle = float(rng.uniform(-np.pi, np.pi))
        axis = np.cos(angle) * xx + np.sin(angle) * yy
        amplitude += np.float32(rng.uniform(0.008, 0.025)) * (
            1.0 + np.cos(rng.uniform(2.0, 7.0) * np.pi * axis + rng.uniform(-np.pi, np.pi))
        ).astype(np.float32)

    for _ in range(inclusive_integer(rng, scene["distributed_region_count_range"])):
        cy, cx = rng.uniform(-0.82, 0.82, size=2)
        sy, sx = rng.uniform(0.10, 0.34, size=2)
        envelope = np.exp(-0.5 * (np.square((yy - cy) / sy) + np.square((xx - cx) / sx)))
        texture_axis = np.cos(rng.uniform(-np.pi, np.pi)) * xx + np.sin(rng.uniform(-np.pi, np.pi)) * yy
        texture = 0.72 + 0.28 * np.cos(rng.uniform(8.0, 22.0) * texture_axis + rng.uniform(-np.pi, np.pi))
        amplitude += np.asarray(rng.uniform(0.12, 0.42) * envelope * texture, dtype=np.float32)

    for _ in range(inclusive_integer(rng, scene["line_count_range"])):
        cy, cx = rng.uniform(-0.82, 0.82, size=2)
        angle = rng.uniform(0.0, np.pi)
        half_length = rng.uniform(0.08, 0.30)
        dy, dx = half_length * np.sin(angle), half_length * np.cos(angle)
        distance = distance_to_segment(yy, xx, cy - dy, cx - dx, cy + dy, cx + dx)
        amplitude += np.asarray(
            rng.uniform(0.12, 0.48) * np.exp(-0.5 * np.square(distance / rng.uniform(0.006, 0.018))),
            dtype=np.float32,
        )

    for _ in range(inclusive_integer(rng, scene["point_cluster_count_range"])):
        cy, cx = rng.uniform(-0.88, 0.88, size=2)
        for _ in range(inclusive_integer(rng, scene["points_per_cluster_range"])):
            py, px = cy + rng.normal(0.0, 0.025), cx + rng.normal(0.0, 0.025)
            width = rng.uniform(0.0025, 0.008)
            amplitude += np.asarray(
                rng.uniform(0.25, 0.95)
                * np.exp(-0.5 * (np.square((yy - py) / width) + np.square((xx - px) / width))),
                dtype=np.float32,
            )

    amplitude -= float(amplitude.min())
    amplitude /= max(float(amplitude.max()), 1.0e-8)
    amplitude = np.asarray(float(scene["amplitude_floor"]) + (1.0 - float(scene["amplitude_floor"])) * amplitude, dtype=np.float32)

    base_phase = rng.uniform(-np.pi, np.pi, size=yy.shape).astype(np.float32)
    for _ in range(4):
        angle = rng.uniform(-np.pi, np.pi)
        axis = np.cos(angle) * xx + np.sin(angle) * yy
        base_phase += np.asarray(
            rng.uniform(0.06, 0.18) * np.sin(rng.uniform(0.8, 3.0) * np.pi * axis + rng.uniform(-np.pi, np.pi)),
            dtype=np.float32,
        )

    amplitude2 = amplitude.copy()
    amplitude_change_bounds = difficulty_cfg["local_amplitude_change_fraction"]
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = rng.uniform(-0.85, 0.85, size=2)
        sy, sx = rng.uniform(0.035, 0.18, size=2)
        region = np.exp(-0.5 * (np.square((yy - cy) / sy) + np.square((xx - cx) / sx)))
        fraction = rng.uniform(float(amplitude_change_bounds[0]), float(amplitude_change_bounds[1]))
        sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
        amplitude2 *= np.asarray(1.0 + sign * fraction * region, dtype=np.float32)

    # Local scatter appearance/disappearance is separate from smooth calibration change.
    for _ in range(int(rng.integers(2, 5))):
        cy, cx = rng.uniform(-0.9, 0.9, size=2)
        width = rng.uniform(0.004, 0.018)
        scatter = np.exp(-0.5 * (np.square((yy - cy) / width) + np.square((xx - cx) / width)))
        amplitude2 += np.asarray(rng.uniform(-0.18, 0.28) * scatter, dtype=np.float32)
    amplitude2 = np.clip(amplitude2, float(scene["amplitude_floor"]), 1.2).astype(np.float32)

    deformation = np.zeros_like(amplitude, dtype=np.float32)
    for _ in range(inclusive_integer(rng, scene["phase_deformation_region_count_range"])):
        cy, cx = rng.uniform(-0.82, 0.82, size=2)
        sy, sx = rng.uniform(0.05, 0.22, size=2)
        sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
        deformation += np.asarray(
            sign * rng.uniform(0.35, 1.0)
            * np.exp(-0.5 * (np.square((yy - cy) / sy) + np.square((xx - cx) / sx))),
            dtype=np.float32,
        )
    deformation += np.asarray(rng.uniform(-0.18, 0.18) * xx + rng.uniform(-0.18, 0.18) * yy, dtype=np.float32)
    target_deformation = rng.uniform(*map(float, difficulty_cfg["maximum_deformation_rad"]))
    deformation *= np.float32(target_deformation / max(float(np.max(np.abs(deformation))), 1.0e-8))

    coherence = np.full_like(amplitude, 0.995, dtype=np.float32)
    minimum_coherence = rng.uniform(*map(float, difficulty_cfg["minimum_local_coherence"]))
    for _ in range(inclusive_integer(rng, scene["coherence_change_region_count_range"])):
        cy, cx = rng.uniform(-0.82, 0.82, size=2)
        sy, sx = rng.uniform(0.04, 0.20, size=2)
        region = np.exp(-0.5 * (np.square((yy - cy) / sy) + np.square((xx - cx) / sx)))
        local_minimum = rng.uniform(minimum_coherence, min(minimum_coherence + 0.08, 0.985))
        coherence = np.minimum(coherence, np.asarray(1.0 - (1.0 - local_minimum) * region, dtype=np.float32))

    independent_phase = rng.uniform(-np.pi, np.pi, size=yy.shape).astype(np.float32)
    shared_phasor = np.exp(np.complex64(1j) * (base_phase + deformation)).astype(np.complex64)
    independent_phasor = np.exp(np.complex64(1j) * independent_phase).astype(np.complex64)
    x1 = np.asarray(amplitude * np.exp(np.complex64(1j) * base_phase), dtype=np.complex64)
    x2 = np.asarray(
        amplitude2
        * (
            coherence * shared_phasor
            + np.sqrt(np.maximum(1.0 - np.square(coherence), 0.0)).astype(np.float32)
            * independent_phasor
        ),
        dtype=np.complex64,
    )
    return {
        "x1": x1,
        "x2": x2,
        "coherence": coherence,
        "difficulty": difficulty,
        "target_deformation_rad": float(target_deformation),
        "minimum_coherence": float(coherence.min()),
    }


def parent_plan(config: dict[str, Any], split_settings: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    maximum = int(config["maximum_patches_per_parent"])
    parent_counts = {
        split: int(math.ceil(int(split_settings[split]["sample_count"]) / maximum))
        for split in SPLITS
    }
    total = sum(parent_counts.values())
    rng = np.random.default_rng(int(config["base_seed"]) + 17)
    numeric_ids = rng.permutation(np.arange(total, dtype=np.int64)).tolist()
    cursor = 0
    output: dict[str, list[dict[str, Any]]] = {}
    for split_index, split in enumerate(SPLITS):
        rows = []
        remaining = int(split_settings[split]["sample_count"])
        for local_index in range(parent_counts[split]):
            numeric_id = int(numeric_ids[cursor])
            cursor += 1
            patch_count = min(maximum, remaining)
            remaining -= patch_count
            rows.append({
                "parent_scene_id": f"sim_parent_{numeric_id:05d}",
                "parent_numeric_id": numeric_id,
                "parent_seed": int(config["base_seed"]) + 100_003 * numeric_id + 10_000_019,
                "patch_count": patch_count,
                "split": split,
                "split_index": split_index,
            })
        output[split] = rows
    return output


def patch_coordinates(config: dict[str, Any], parent_seed: int) -> list[tuple[int, int, int, int]]:
    parent_rows, parent_columns = map(int, config["parent_scene_shape"])
    patch_rows, patch_columns = map(int, config["image_shape"])
    stride_rows, stride_columns = map(int, config["patch_stride"])
    coordinates = [
        (row, row + patch_rows, column, column + patch_columns)
        for row in range(0, parent_rows - patch_rows + 1, stride_rows)
        for column in range(0, parent_columns - patch_columns + 1, stride_columns)
    ]
    rng = np.random.default_rng(int(parent_seed) + 31)
    order = rng.permutation(len(coordinates))
    return [coordinates[int(index)] for index in order]


def transform_patch(value: np.ndarray, rotation: int, flip: bool) -> np.ndarray:
    output = np.rot90(value, k=int(rotation))
    if flip:
        output = np.fliplr(output)
    return np.ascontiguousarray(output)


def make_patch(
    config: dict[str, Any],
    parent: dict[str, Any],
    rendered: dict[str, Any],
    coordinates: tuple[int, int, int, int],
    patch_index: int,
    sample_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row0, row1, column0, column1 = map(int, coordinates)
    generation_seed = int(parent["parent_seed"]) + 1009 * int(patch_index) + 53
    rng = np.random.default_rng(generation_seed)
    rotation = int(rng.integers(0, 4)) if parent["split"] == "train" else 0
    flip = bool(rng.integers(0, 2)) if parent["split"] == "train" else False
    x1 = transform_patch(rendered["x1"][row0:row1, column0:column1], rotation, flip)
    x2 = transform_patch(rendered["x2"][row0:row1, column0:column1], rotation, flip)
    coherence = transform_patch(
        rendered["coherence"][row0:row1, column0:column1], rotation, flip
    ).astype(np.float32)
    amplitude1, amplitude2 = np.abs(x1), np.abs(x2)
    geometric_amplitude = np.sqrt(amplitude1 * amplitude2)
    scene_cfg = config["scene"]
    reliability = np.clip(
        (geometric_amplitude - float(scene_cfg["phase_weight_low_amplitude"]))
        / max(
            float(scene_cfg["phase_weight_high_amplitude"])
            - float(scene_cfg["phase_weight_low_amplitude"]),
            1.0e-8,
        ),
        0.0,
        1.0,
    )
    phase_weight = np.asarray(reliability * coherence, dtype=np.float32)
    hard_valid = np.asarray(
        phase_weight >= float(scene_cfg["hard_valid_threshold"]), dtype=np.uint8
    )
    delta_phi = np.asarray(np.angle(x2 * np.conj(x1)), dtype=np.float32)
    split = str(parent["split"])
    sample_id = f"{split}_{sample_index:06d}"
    parent_numeric_id = int(parent["parent_numeric_id"])
    global_tile_x_m = float((parent_numeric_id % 32) * 2000.0)
    global_tile_y_m = float((parent_numeric_id // 32) * 2000.0)
    pixel_spacing_m = 0.5
    metadata = {
        "sample_id": sample_id,
        "parent_scene_id": parent["parent_scene_id"],
        "split": split,
        "crop_coordinates": [row0, row1, column0, column1],
        "rotation_quarter_turns": rotation,
        "horizontal_flip": flip,
        "difficulty": rendered["difficulty"],
        "generation_seed": generation_seed,
        "parent_seed": int(parent["parent_seed"]),
        "parent_scene_shape": list(map(int, config["parent_scene_shape"])),
        "native_patch_shape": list(map(int, config["image_shape"])),
        "native_direct_simulation": True,
        "interpolation_from_32x32": False,
        "temporal_baseline_days": int(rng.integers(12, 73)),
        "target_deformation_rad": rendered["target_deformation_rad"],
        "minimum_parent_coherence": rendered["minimum_coherence"],
        "mean_patch_coherence": float(coherence.mean()),
        "global_bounds_m": [
            global_tile_y_m + row0 * pixel_spacing_m,
            global_tile_y_m + row1 * pixel_spacing_m,
            global_tile_x_m + column0 * pixel_spacing_m,
            global_tile_x_m + column1 * pixel_spacing_m,
        ],
    }
    arrays = {
        "x1": np.asarray(x1, dtype=np.complex64),
        "x2": np.asarray(x2, dtype=np.complex64),
        "delta_phi_gt": delta_phi,
        "phase_weight": phase_weight,
        "hard_valid_mask": hard_valid,
        "sample_id": np.asarray(sample_id),
        "parent_scene_id": np.asarray(parent["parent_scene_id"]),
        "crop_coordinates": np.asarray([row0, row1, column0, column1], dtype=np.int32),
        "difficulty": np.asarray(rendered["difficulty"]),
        "generation_seed": np.asarray(generation_seed, dtype=np.int64),
        "split": np.asarray(split),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    return arrays, metadata


def validate_generated_arrays(arrays: dict[str, np.ndarray], expected_shape: tuple[int, int]) -> float:
    if arrays["x1"].shape != expected_shape or arrays["x2"].shape != expected_shape:
        raise RuntimeError("generated complex truth has the wrong shape")
    for name in ("delta_phi_gt", "phase_weight", "hard_valid_mask"):
        if arrays[name].shape != expected_shape:
            raise RuntimeError(f"{name} has the wrong shape")
    if not np.iscomplexobj(arrays["x1"]) or not np.iscomplexobj(arrays["x2"]):
        raise RuntimeError("x1/x2 are not complex")
    for name in ("x1", "x2", "delta_phi_gt", "phase_weight"):
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"{name} contains NaN/Inf")
    recalculated = np.angle(arrays["x2"] * np.conj(arrays["x1"]))
    error = np.angle(np.exp(1j * (recalculated - arrays["delta_phi_gt"])))
    maximum_error = float(np.max(np.abs(error)))
    if maximum_error > 1.0e-6:
        raise RuntimeError(f"differential phase label error {maximum_error}")
    if float(arrays["phase_weight"].min()) < 0.0 or float(arrays["phase_weight"].max()) > 1.0:
        raise RuntimeError("phase_weight lies outside [0,1]")
    return maximum_error


def noise_seed_record(config: dict[str, Any], split: str, sample_id: str, sample_index: int) -> dict[str, Any]:
    split_offset = {"train": 100_000_000, "validation": 200_000_000, "test": 300_000_000}[split]
    banks = []
    for snr_index, snr_db in enumerate(config["noise_bank"]["snr_db"]):
        base = int(config["base_seed"]) + split_offset + 100 * int(sample_index) + 2 * snr_index
        banks.append({"snr_db": int(snr_db), "epoch1_seed": base, "epoch2_seed": base + 1})
    return {"sample_id": sample_id, "split": split, "banks": banks}


def generate_dataset(
    mode: str,
    *,
    config_path: Path = CONFIG_PATH,
    output_override: Path | None = None,
) -> dict[str, Any]:
    config = load_json(config_path)
    split_settings = config["dry_run_splits"] if mode == "dry" else config["splits"]
    configured_output = ROOT / config["outputs"][
        "dry_run_directory" if mode == "dry" else "full_directory"
    ]
    output = (output_override or configured_output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        directory = output / split_settings[split]["directory"] / "samples"
        directory.mkdir(parents=True, exist_ok=True)
    noise_directory = output / "noise_banks"
    noise_directory.mkdir(parents=True, exist_ok=True)

    plan = parent_plan(config, split_settings)
    parent_sets = {split: {row["parent_scene_id"] for row in plan[split]} for split in SPLITS}
    intersections = {
        "train_validation": sorted(parent_sets["train"] & parent_sets["validation"]),
        "train_test": sorted(parent_sets["train"] & parent_sets["test"]),
        "validation_test": sorted(parent_sets["validation"] & parent_sets["test"]),
    }
    if any(intersections.values()):
        raise RuntimeError(f"parent split leakage before generation: {intersections}")

    yy, xx = normalized_parent_grid(tuple(map(int, config["parent_scene_shape"])))
    records: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, Moments]] = {
        split: {"amplitude": Moments(), "delta_phi": Moments(), "phase_weight": Moments()}
        for split in SPLITS
    }
    noise_handles = {
        split: (noise_directory / f"{split}_noise_seeds.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    file_hashes: set[str] = set()
    truth_hashes: set[str] = set()
    duplicate_file_hashes = 0
    duplicate_truth_hashes = 0
    maximum_phase_error = 0.0
    generated_counts = {split: 0 for split in SPLITS}
    difficulty_counts = {split: {"easy": 0, "medium": 0, "hard": 0} for split in SPLITS}
    start_time = time.time()
    try:
        for split in SPLITS:
            sample_index = 0
            for parent_number, parent in enumerate(plan[split]):
                rendered = render_parent_scene(config, int(parent["parent_seed"]), yy, xx)
                coordinates = patch_coordinates(config, int(parent["parent_seed"]))
                for patch_index in range(int(parent["patch_count"])):
                    arrays, metadata = make_patch(
                        config, parent, rendered, coordinates[patch_index], patch_index, sample_index
                    )
                    maximum_phase_error = max(
                        maximum_phase_error,
                        validate_generated_arrays(arrays, tuple(map(int, config["image_shape"])))
                    )
                    relative = Path(split_settings[split]["directory"]) / "samples" / f"{metadata['sample_id']}.npz"
                    path = output / relative
                    np.savez(path, **arrays)
                    file_hash = sha256_file(path)
                    truth_hash = canonical_truth_hash(arrays)
                    duplicate_file_hashes += int(file_hash in file_hashes)
                    duplicate_truth_hashes += int(truth_hash in truth_hashes)
                    file_hashes.add(file_hash)
                    truth_hashes.add(truth_hash)
                    amplitude = np.concatenate((np.abs(arrays["x1"]).ravel(), np.abs(arrays["x2"]).ravel()))
                    distributions[split]["amplitude"].update(amplitude)
                    distributions[split]["delta_phi"].update(arrays["delta_phi_gt"])
                    distributions[split]["phase_weight"].update(arrays["phase_weight"])
                    record = {
                        **metadata,
                        "relative_path": relative.as_posix(),
                        "file_sha256": file_hash,
                        "truth_content_sha256": truth_hash,
                        "hard_valid_ratio": float(arrays["hard_valid_mask"].mean()),
                        "phase_weight_mean": float(arrays["phase_weight"].mean()),
                    }
                    records.append(record)
                    noise = noise_seed_record(config, split, metadata["sample_id"], sample_index)
                    noise_handles[split].write(json.dumps(noise, ensure_ascii=False) + "\n")
                    generated_counts[split] += 1
                    difficulty_counts[split][str(metadata["difficulty"])] += 1
                    sample_index += 1
                if (parent_number + 1) % 8 == 0 or parent_number + 1 == len(plan[split]):
                    elapsed = time.time() - start_time
                    print(
                        json.dumps({
                            "mode": mode,
                            "split": split,
                            "parents_done": parent_number + 1,
                            "parents_total": len(plan[split]),
                            "samples_done": generated_counts[split],
                            "elapsed_seconds": round(elapsed, 1),
                        }),
                        flush=True,
                    )
    finally:
        for handle in noise_handles.values():
            handle.close()

    expected_counts = {split: int(split_settings[split]["sample_count"]) for split in SPLITS}
    if generated_counts != expected_counts:
        raise RuntimeError(f"split count mismatch: {generated_counts} versus {expected_counts}")

    manifest = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"] + ("_dry_run" if mode == "dry" else ""),
        "mode": mode,
        "source": "native 1024x1024 controlled complex SAR parent simulation; direct non-overlapping 256x256 crops",
        "native_high_resolution_simulation": True,
        "interpolation_from_32x32": False,
        "truth_only": True,
        "precomputed_masked_observations": False,
        "aperture_candidate_count": int(config["aperture_candidate_count"]),
        "future_sample_counts": list(map(int, config["future_sample_counts"])),
        "split_directories": {split: split_settings[split]["directory"] for split in SPLITS},
        "samples": records,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "split_manifest.json", {
        "split_seed": int(config["base_seed"]) + 17,
        "parent_first": True,
        "maximum_patches_per_parent": int(config["maximum_patches_per_parent"]),
        "parent_plan": plan,
        "parent_intersections": intersections,
    })
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        write_json(output / split_settings[split]["directory"] / "manifest.json", {
            "split": split,
            "sample_count": len(split_records),
            "parent_scene_count": len(parent_sets[split]),
            "samples": split_records,
        })

    noise_hashes = {
        split: sha256_file(noise_directory / f"{split}_noise_seeds.jsonl") for split in SPLITS
    }
    statistics = {
        "split_sample_counts": generated_counts,
        "split_parent_scene_counts": {split: len(parent_sets[split]) for split in SPLITS},
        "difficulty_counts": difficulty_counts,
        "distributions": {
            split: {name: moment.result() for name, moment in values.items()}
            for split, values in distributions.items()
        },
        "maximum_wrapped_phase_label_error_rad": maximum_phase_error,
        "duplicate_file_hash_count": duplicate_file_hashes,
        "duplicate_truth_content_hash_count": duplicate_truth_hashes,
        "noise_bank_seed_file_sha256": noise_hashes,
        "runtime_seconds": time.time() - start_time,
    }
    write_json(output / "statistics.json", statistics)
    description = f"""# Final SAR Dataset 256 ({mode})

- Native truth size: 256x256 complex64 per epoch.
- Source: directly simulated 1024x1024 coherent complex parent scenes; no 32x32 interpolation.
- Parent scenes are assigned to train/validation/test before any patch is cropped.
- Each parent contributes at most 16 non-overlapping 256x256 crops.
- Epoch 2 contains spatial deformation, local amplitude/scatter change, and local coherence change.
- Saved data are clean truth and metadata only; no mask-specific observation is stored.
- Aperture candidate count is N=256; registered future budgets are K=80/104/128.
- Noise banks are deterministic split-independent seed tables for 10/15/20/25/30 dB and are materialized online.
- The test directory is sealed by the default loader.
"""
    (output / "DATASET_DESCRIPTION.md").write_text(description, encoding="utf-8")

    loader = FinalSARDataset256(output)
    for split in ("train", "validation"):
        for index in range(min(3, len(loader.records[split]))):
            sample = loader.load_sample(split, index)
            validate_generated_arrays(sample, tuple(map(int, config["image_shape"])))
    sealed_guard_passed = False
    try:
        loader.load_sample("test", 0)
    except PermissionError:
        sealed_guard_passed = True

    reproduced_plan = parent_plan(config, split_settings)
    split_reproducible = reproduced_plan == plan
    first_record = next(record for record in records if record["split"] == "train")
    first_parent = next(
        parent for parent in plan["train"] if parent["parent_scene_id"] == first_record["parent_scene_id"]
    )
    rerendered = render_parent_scene(config, int(first_parent["parent_seed"]), yy, xx)
    first_coordinates = tuple(map(int, first_record["crop_coordinates"]))
    patch_index = patch_coordinates(config, int(first_parent["parent_seed"])).index(first_coordinates)
    reproduced_arrays, _ = make_patch(
        config, first_parent, rerendered, first_coordinates, patch_index, 0
    )
    sample_reproducible = canonical_truth_hash(reproduced_arrays) == first_record["truth_content_sha256"]

    disk_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    audit = {
        "mode": mode,
        "status": "passed",
        "all_samples_256x256_checked_during_generation": True,
        "all_x1_x2_complex_checked_during_generation": True,
        "all_samples_finite_checked_during_generation": True,
        "maximum_wrapped_phase_label_error_rad": maximum_phase_error,
        "parent_scene_intersections": intersections,
        "parent_scene_leakage_count": sum(len(value) for value in intersections.values()),
        "cross_split_patch_overlap_count": 0,
        "duplicate_file_hash_count": duplicate_file_hashes,
        "duplicate_truth_content_hash_count": duplicate_truth_hashes,
        "split_reproducible": split_reproducible,
        "sample_reproducible": sample_reproducible,
        "sealed_test_guard_passed": sealed_guard_passed,
        "training_style_loader_files_opened": loader.files_opened,
        "test_files_opened_by_training_style_loader": loader.files_opened["test"],
        "disk_bytes": disk_bytes,
        "disk_gib": disk_bytes / (1024 ** 3),
        "split_sample_counts": generated_counts,
        "split_parent_scene_counts": {split: len(parent_sets[split]) for split in SPLITS},
    }
    mandatory = [
        maximum_phase_error <= 1.0e-6,
        audit["parent_scene_leakage_count"] == 0,
        duplicate_file_hashes == 0,
        duplicate_truth_hashes == 0,
        split_reproducible,
        sample_reproducible,
        sealed_guard_passed,
        loader.files_opened["test"] == 0,
    ]
    if not all(mandatory):
        audit["status"] = "failed"
    write_json(output / "generation_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if audit["status"] != "passed":
        raise RuntimeError(f"{mode} audit failed")
    return {"output": output, "audit": audit, "statistics": statistics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic DP-JMRNet simulated SAR dataset."
    )
    parser.add_argument("--mode", choices=("dry", "full"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory; it must be empty or not yet exist.",
    )
    arguments = parser.parse_args()
    generate_dataset(
        arguments.mode,
        config_path=arguments.config.resolve(),
        output_override=arguments.output,
    )


if __name__ == "__main__":
    main()
