# QuanKAN

Paper-aligned implementation of **QuanKAN: Spectrum-Conditioned DE and Kolmogorov-Arnold Quantum Residual Learning for Cross-Subject EEG Emotion Recognition**.

The model consumes precomputed differential-entropy features smoothed by a linear dynamical system (DE-LDS). Raw EEG preprocessing is outside this repository. The implementation follows the architecture and strict leave-one-subject-out (LOSO) protocol in the author's updated LaTeX (2026-10-06); see [`METHOD_CONTRACT.md`](METHOD_CONTRACT.md) for the equation-to-code map and details the paper leaves unspecified.

## Repository layout

```text
configs/base/default.json  Paper settings and explicit implementation choices
src/quankan/
  datasets/                Dataset readers, source-only scaling, electrode graph
  models/                  Feature enhancer, graph encoder, KAN, quantum branch
  evaluation/              Validation and test metrics
  training/                Fold engine, LOSO orchestration, CLI, RNG setup
  augmentation.py           Training-only DE-LDS augmentation
  losses.py                 Contrastive, prototype, and paper objective terms
tests/                      Method-contract and model-shape checks
channel_62_pos.locs         Electrode coordinates used by the fixed graph
```

The model follows the modular separation used in the [FireQuan repository](https://github.com/nhanhqq/firequan): model components, data loading, training, configuration, tests, and command-line entry points are kept independent. The V4 spectrum-conditioned Feature Enhancer is the source for the paper's input-refinement stage; the remaining model components implement the paper's explicit architecture and equations.

## Method contract

- Input tensors have shape `[batch, time, 62 electrodes, 5 bands]` and retain zero-valued padding.
- Feature normalization is fit on the training portion of the source subjects only. The held-out subject is not used for normalization, training, validation, checkpoint selection, or hyperparameter selection.
- Validation uses a stratified 20% trial split from source subjects. The best checkpoint is selected by source-validation accuracy.
- Each checkpoint is evaluated on its held-out subject once, after training ends. The runner requires exactly five seeds.
- Reported means and standard deviations average seeds within each subject first, then summarize across held-out subjects.
- The shared embedding follows Eq. (19), with 192 dimensions. The temporally pooled encoder representation has 256 dimensions before this projection.
- Six auxiliary-loss coefficients are learned jointly with the model, as clarified by the author. Their effective values are `max(raw_lambda, 0)`. Config values are initialization only; learned coefficients and their state are saved with the selected checkpoint. The main emotion loss retains coefficient one.

See [`METHOD_CONTRACT.md`](METHOD_CONTRACT.md) for remaining paper/implementation ambiguities. Results produced by the older target-selected V4 runner are not strict-LOSO results and must not be reported as results from this runner.

## Data layout

Pass a dataset root with `--data-root`:

| Dataset | Expected directory below the root |
| --- | --- |
| SEED | `ExtractedFeatures_1s/` |
| SEED IV | Session directories `1/`, `2/`, and `3/` |
| SEED V | `EEG_DE_features/` |
| SEED VII | `EEG_features/` |

Loaders expect the published precomputed DE-LDS feature files. They pad trials with zeros and preserve those padded windows through preprocessing and the model.

## Install

The verified environment uses Python 3.10.20, Torch 2.11.0 (CPU build) and PennyLane 0.42.3. The bundled PDF predates the updated LaTeX supplied by the author; see the method contract for the source authority and unresolved manuscript settings.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The default quantum simulator is PennyLane `default.qubit`. Install a hardware-specific PyTorch build separately if required by the machine.

For the CPU environment used in the release checks, installation can also use `uv`:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --torch-backend cpu -e '.[test]'
```

The complete tested CPU environment is pinned in
[`requirements/cpu-verified.txt`](requirements/cpu-verified.txt).

## Run strict LOSO

Run from the repository root. Supply five integer seeds and the feature directory for one dataset:

```bash
python -m quankan.train_loso \
  --dataset seediv \
  --data-root PATH_TO_SEED_IV_ROOT \
  --seeds 1 2 3 4 5
```

The public defaults are in [`configs/base/default.json`](configs/base/default.json), with an identical copy and electrode coordinates bundled in the installed package; command-line values override that file. Use `--help` to see the available overrides. Results are written under `outputs/<dataset>/` as `fold_results.json`, `summary.json`, and one checkpoint per held-out subject and seed. Output directories are never shared with the source V4 project.

## Execution and method checks

```bash
pytest -q
quankan-smoke
quankan-size
```

The smoke command runs two epochs on synthetic trials using the real fold engine, exercises quantum warm-up, selects on source validation and tests the held-out synthetic subject once. It writes to a new temporary directory and labels its artifacts as development checks. `quankan-size` reports actual parameters and FP32 weights; it does not reproduce the older manuscript MAC/FLOP table.

Geometry overrides to latent256 or a different token count are rejected. Supply five distinct seeds and all SEED/SEED-IV sessions. A fresh output directory is required for each experiment. Epoch/batch/LR overrides are saved and labeled when they differ from manuscript settings.

The published accuracy/F1 tables have not been reproduced by these checks. Loss-weight initialization/parameterization, source-validation details and other choices absent from the updated LaTeX are documented in the method contract. Current model-size counts also differ from the LaTeX's legacy V4 table.

The completed checks and real-data execution evidence are recorded in
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) and
[`docs/verification.json`](docs/verification.json).

## License

MIT. See [`LICENSE`](LICENSE).
