<div align="center">
  <h1>DP-JMRNet</h1>
  <p><strong>Joint complex reconstruction for bitemporal SAR with discontinuous aperture sampling</strong></p>
</div>

## Overview

Discontinuous aperture sampling leaves gaps in the SAR phase history and makes high-quality reconstruction difficult. In a bitemporal setting, reconstructing the two acquisitions independently can further disturb their shared structure and amplify phase errors, reducing the reliability of the differential phase used for change analysis.

DP-JMRNet addresses this problem by reconstructing both complex SAR images jointly. It combines the SAR observation model with a five-stage deep-unfolding network and selectively exchanges information between the two epochs. The interaction is reduced when the local observations do not provide reliable shared information, which helps preserve epoch-specific content while improving differential-phase reconstruction.

This repository provides the reconstruction model, a matrix-free wideband SAR operator, training and validation scripts, numerical tests, a small example dataset, and the code required to reproduce the complete simulated dataset.

## Method

The model takes two incomplete complex phase histories acquired at different epochs and reconstructs the corresponding complex SAR images together. The known SAR acquisition geometry and aperture sampling pattern are used directly in the reconstruction rather than being approximated by a generic image-degradation model.

Each of the five unfolding stages contains three main components:

1. matrix-free data consistency based on the SAR forward and adjoint operators;
2. a shared complex-valued image prior for the two epochs;
3. coherence- and residual-aware selective interaction between the reconstructions.

Data consistency is applied both before the shared prior and after the selective interaction. The shared prior captures structures common to both acquisitions, while the interaction module determines where cross-epoch information is reliable enough to use. In regions with weak coherence or conflicting measurement evidence, the two reconstruction branches operate more independently to avoid transferring misleading information.

The network outputs two complex SAR images. Training considers complex reconstruction quality, amplitude fidelity, individual phase accuracy, and the differential phase between the two epochs. Exchanging the input epochs exchanges the corresponding outputs, so the result does not depend on an arbitrary ordering of the image pair.

## Repository structure

```text
.
├── configs/data/                   simulated-dataset specification
├── data/simulated_dataset/         bundled 32/8/8 example dataset
├── scripts/generate_dataset.py     complete dataset generator
├── scripts/train.py                training entry point
├── scripts/validate.py             checkpoint evaluation entry point
├── src/coherent_sar/               model, SAR operator, and data pipeline
├── tests/                          numerical and architectural tests
├── config.json                     quick example configuration
└── config.full.json                complete-dataset training configuration
```

## Installation

Python 3.10 or newer is required. Install a PyTorch build suitable for your CPU or CUDA environment, then install the project:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run the tests to verify the installation:

```bash
python -m pytest
```

## Dataset

The bundled example dataset contains 32 training samples, 8 validation samples, and 8 held-out samples. It uses the same format as the complete dataset and is intended for testing the installation and training pipeline.

The complete simulated dataset contains 4,000 bitemporal samples:

| Split | Samples | Parent scenes |
|:--|--:|--:|
| Train | 3,000 | 188 |
| Validation | 500 | 32 |
| Test | 500 | 32 |

Each sample contains two native `256 × 256` complex SAR images, wrapped differential-phase ground truth, phase-reliability weights, a validity mask, and generation metadata. Parent scenes are assigned to train, validation, and test before patches are cropped, preventing parent-scene overlap between splits.

Generate the complete dataset with:

```bash
python scripts/generate_dataset.py --mode full
```

The output is written to `data/final_sar_dataset_256/`. The generator also creates manifests, split statistics, integrity checks, per-sample hashes, and deterministic noise-seed banks.

For a smaller generator check, run:

```bash
python scripts/generate_dataset.py --mode dry
```

A custom output directory can be specified with `--output`:

```bash
python scripts/generate_dataset.py --mode full --output /path/to/dataset
```

## Training

Run a one-step end-to-end check with the bundled example dataset:

```bash
python scripts/train.py --steps 1
```

Standard example training:

```bash
python scripts/train.py --config config.json
```

After generating the complete dataset, run the complete-data configuration with:

```bash
python scripts/train.py --config config.full.json
```

Training writes `best.pt`, `last.pt`, and `metrics.json` to the configured output directory. A CUDA-capable GPU is recommended for training the full five-stage `256 × 256` model.

## Validation

Validate a checkpoint trained with the example configuration:

```bash
python scripts/validate.py --config config.json --checkpoint outputs/best.pt
```

For the complete-data configuration:

```bash
python scripts/validate.py --config config.full.json --checkpoint outputs/full_k104/best.pt
```

The validation script reports weighted differential-phase RMSE, amplitude NMSE, and complex NMSE.

## Reproducibility

- The dataset is generated from a versioned configuration and a fixed base seed.
- Train, validation, and test splits are separated at the parent-scene level.
- Manifests record the generation metadata and numerical truth hashes of every sample.
- Noise seeds are deterministic and stored independently for each split and epoch.
- The tests cover the complex adjoint relation, operator gradients, exchange equivariance, selective fallback, and model backpropagation.

## Scope

The provided dataset is a controlled coherent complex SAR simulation intended for reproducible method development. Evaluation on independently acquired SAR data is still necessary when assessing performance across sensors, terrain types, and acquisition conditions.

## License

Released under the [MIT License](LICENSE).
