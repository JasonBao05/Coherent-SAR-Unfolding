"""Validate a DP-JMRNet checkpoint on the included validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from coherent_sar.pipeline import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "best.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args()
    metrics = validate_checkpoint(
        arguments.config, arguments.checkpoint, device_name=arguments.device
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

