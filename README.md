# QuanKAN

Reference implementation for **“Spectrum-Conditioned DE and Kolmogorov-Arnold Quantum Residual Learning for Cross-Subject EEG Emotion Recognition.”** It trains and evaluates QuanKAN from precomputed DE-LDS features; raw EEG preprocessing is outside this repository.

## Method

The model follows the Methodology dimensions: 16 spatial tokens with 32 features each (512 concatenated features), a 256-dimensional temporal path, and a 192-dimensional shared embedding. The quantum branch uses a four-qubit, four-layer circuit. See `quankan/model.py`, `quankan/losses.py`, and `quankan/train_loso.py` for the model, objectives, and LOSO runner.

## Repository layout

```text
quankan/                 Model, data loaders, objectives, augmentation, LOSO runner
tests/                   Method-contract tests
channel_62_pos.locs      Electrode coordinates used to build the graph
outputs/                 Generated checkpoints and metrics (git-ignored)
```

## Environment

Use Python 3.10. Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
```

The default simulator is PennyLane `default.qubit` and runs without a quantum device. CUDA is optional; install a PyTorch build compatible with your CUDA setup when using a GPU.

## Data

The loaders expect licensed DE-LDS feature files in the dataset's original layout. Dataset files are not included or redistributed here. Each input trial is converted to `[time, 62 electrodes, 5 bands]` and padded within the loaded dataset.

| Dataset | `--data-root` contents |
| --- | --- |
| SEED | `ExtractedFeatures_1s/` |
| SEED IV | `1/`, `2/`, and `3/` session folders |
| SEED V | `EEG_DE_features/` |
| SEED VII | `EEG_features/` |

## Train and evaluate

Run from the repository root. The runner executes all held-out-subject folds and requires exactly five integer seed IDs. Replace the placeholders with the verified experiment seeds; the manuscript reports five seeds but does not list their values.

```bash
python -m quankan.train_loso \
  --dataset seediv \
  --data-root "<path-to-SEED-IV-root>" \
  --seeds <seed-1> <seed-2> <seed-3> <seed-4> <seed-5>
```

Supported dataset names are `seed`, `seediv`, `seedv`, and `seedvii`. Use `python -m quankan.train_loso --help` to see all options. Defaults include batch size 32, 300 epochs, AdamW, and the paper's listed learning rates and augmentation magnitudes. Validation is a 20% split from source subjects only; this fraction is an implementation choice because the manuscript does not specify it. The held-out subject is evaluated after checkpoint selection.

Checkpoints and metrics are written to `outputs/<dataset>/`: `fold_results.json`, `summary.json`, and per-subject checkpoints under `checkpoints/`. These generated artifacts are excluded from Git.

## Tests

```bash
python -m pytest -q
```

The tests check KAN basis dimensions, the quantum MSE objective, schedules, dataset graph/labels, and paper-specified model dimensions. They do not validate the reported experimental scores; reproducing those requires the original licensed data and verified seed/configuration values.

## Reproducibility notes

- The Methodology specifies nonnegative loss coefficients but does not give numeric values or their parameterization. As instructed for this implementation, the six coefficients are learned, initialized to the values used by the V4 training source, and constrained to nonnegative effective values. The quantum auxiliary loss follows the paper's raw-output MSE equation.
- With Methodology dimensions, model parameter counts are 468,789, 472,727, 476,858, and 485,506 for `(classes, source subjects)` configurations `(3,14)`, `(4,14)`, `(5,15)`, and `(7,19)`. The manuscript instead reports 589,366–691,652. This internal discrepancy remains unresolved; undocumented parameters are not added to force agreement.
- A complete paper citation is not included because the manuscript source currently has an unfilled author block and no final venue/DOI metadata. Add finalized citation metadata before the public release.

## License

This code is released under the MIT License; see `LICENSE`. Dataset terms remain with their respective providers.
