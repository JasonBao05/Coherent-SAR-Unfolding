# DP-JMRNet

DP-JMRNet is a physics-driven bitemporal SAR reconstruction model for
discontinuous aperture sampling. It reconstructs two complex SAR images
jointly and explicitly protects their differential phase.

The repository contains only the final model, the matrix-free wideband SAR
operator, a compact training/validation pipeline, two numerical validation
tests, and an included simulated dataset.

## Method

For epochs \(t\in\{1,2\}\), the observation model is

\[
y_t=M_tA_tx_t+n_t,
\]

where \(A_t\) is the SAR forward operator and \(M_t\) is a sampling mask in
the acquisition domain. The differential-phase convention is

\[
\Delta\phi=\arg(x_2x_1^*).
\]

The five-stage network applies data consistency before and after a shared
complex prior. Its selective interaction is exchange equivariant: swapping
the two inputs swaps the two outputs. Interaction automatically falls back to
independent reconstruction when local coherence is absent or the two data
consistency residuals conflict strongly.

## Files

```text
config.json                         final model and training configuration
data/simulated_dataset/             included 256 x 256 complex SAR dataset
scripts/train.py                    training entry point
scripts/validate.py                 validation entry point
src/coherent_sar/                   model, operator, and pipeline code
tests/                              operator and model validation tests
LICENSE                             MIT license
```

The included dataset contains 32 training samples, 8 validation samples, and
8 held-out samples. Each sample stores two complex SAR images, differential
phase, phase-reliability weights, and a validity mask. The dataset was created
directly from controlled 1024 x 1024 coherent complex parent scenes and then
cropped into non-overlapping 256 x 256 patches.

## Installation

Python 3.10 or newer is required. Install a PyTorch build suitable for your
CPU or CUDA environment, then install this project:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Validate the implementation

```bash
python -m pytest
```

The tests verify the complex adjoint relation, masked-operator gradients,
shape and SNR stability, selective-gate boundary behavior, exact exchange
equivariance, independent fallback, and model backpropagation.

## Train

```bash
python scripts/train.py
```

For a quick end-to-end check:

```bash
python scripts/train.py --steps 1
```

Training writes `best.pt`, `last.pt`, and `metrics.json` to `outputs/`.
A CUDA-capable GPU is recommended for the full 256 x 256 model.

## Validate a checkpoint

```bash
python scripts/validate.py --checkpoint outputs/best.pt
```

The command reports differential-phase RMSE, amplitude NMSE, and complex NMSE
on the included validation split.

## License

Released under the [MIT License](LICENSE).

