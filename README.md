# QuanKAN

**Spectrum-Conditioned DE and Kolmogorov-Arnold Quantum Residual Learning for Cross-Subject EEG Emotion Recognition**

QuanKAN is a hybrid quantum-classical framework for cross-subject EEG emotion recognition from precomputed differential-entropy features smoothed by a linear dynamical system (DE-LDS).

## Abstract

QuanKAN coordinates spectrum-conditioned input refinement, graph-based spatial encoding, temporal aggregation, and selective quantum residual refinement. Subject-adversarial, supervised contrastive, and prototype objectives regularize the shared representation. A Kolmogorov-Arnold network (KAN) interface maps a detached embedding through a four-qubit, four-layer parameterized quantum circuit. An entropy gate and bounded fusion coefficient control the quantum feature residual added to the classical pathway.

## Core Components

### 1. Spectrum-Conditioned Feature Enhancer

A MobileNetV3-based encoder extracts spectral context from DE-LDS streams. The context generates bounded electrode- and band-specific gains, an additive shift, and temporal and graph residual corrections.

### 2. Spatio-Temporal Graph Encoder

Two graph-attention layers produce electrode features. Learned spatial tokens aggregate them before temporal convolutions, attention pooling, and projection to a 192-dimensional shared embedding.

### 3. KAN-Based Quantum Residual Classifier

Separate KAN mappings implement classical classification, quantum-angle encoding, measurement readout, and hybrid classification. A four-qubit, four-layer data-reuploading circuit produces quantum features, which are fused through a predictive-entropy gate.

## Repository Structure & Components

```text
quankan/
  model.py          Feature enhancer, spatio-temporal encoder, KAN, quantum circuit
  data.py           DE-LDS dataset loaders and electrode-affinity graph
  losses.py         Emotion, subject, contrastive, prototype, and quantum losses
  augmentation.py   EEG feature augmentation
  train_loso.py     LOSO training and evaluation
tests/              Method-contract tests
channel_62_pos.locs Electrode coordinates
outputs/            Checkpoints and evaluation results
```

## Datasets

The model input consists of precomputed DE-LDS features with shape `[trials, time, 62, 5]`. Raw EEG preprocessing is not included. Place each dataset in its original layout and pass its root directory through `--data-root`.

| Dataset | Expected directory under `--data-root` |
| --- | --- |
| SEED | `ExtractedFeatures_1s/` |
| SEED IV | `1/`, `2/`, and `3/` |
| SEED V | `EEG_DE_features/` |
| SEED VII | `EEG_features/` |

## Install

Python 3.10 is used with the pinned dependencies in `requirements.txt`.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
```

The default quantum simulator is PennyLane `default.qubit`.

## Usage

Run from the repository root. Replace the path and seed placeholders with the dataset root and five integer seed IDs.

```bash
python -m quankan.train_loso \
  --dataset seediv \
  --data-root PATH_TO_SEED_IV_ROOT \
  --seeds SEED_1 SEED_2 SEED_3 SEED_4 SEED_5
```

Supported dataset names are `seed`, `seediv`, `seedv`, and `seedvii`. Use `python -m quankan.train_loso --help` for the full list of training options.

## LOSO Evaluation

The evaluation follows leave-one-subject-out (LOSO): each fold holds out one subject for testing, and training and checkpoint selection use source subjects. The runner evaluates every held-out subject for each of the five seeds. It writes `fold_results.json`, `summary.json`, and subject checkpoints under `outputs/<dataset>/`.

## Tests

```bash
python -m pytest -q
```

## License

This project is licensed under the MIT License. See `LICENSE`.
