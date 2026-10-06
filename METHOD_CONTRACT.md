# Contract for the updated manuscript

## Authority and scope

The authority for this release is the complete updated LaTeX supplied by the author on 2026-10-06. The bundled `QuanKAN (27).pdf` is an older manuscript and is preserved as an archival document. Its Table I and quantum-cost entries must not override the newer text.

This package implements the full model and its training/evaluation pipeline. It does not implement raw-EEG acquisition, DE-LDS extraction, or claim that the published accuracy, ablation, diagnostic, or significance tables have been reproduced. The original V4 files and completed experiment outputs in the sibling repository remain unchanged.

## Equation-to-code map

| Updated LaTeX location | Implemented operation | Source |
| --- | --- | --- |
| Architecture Overview | `[B,T,62,5]`, shared latent 192 | `models/network.py` |
| Feature Enhancer, Eqs. 2–4 | All-zero mask, electrode mean, valid-window population statistics, MA3, periodic Hann16, centered zero-padded STFT, hop4, log power, sample-band normalization | `models/feature_enhancer.py::SpectralContextEncoder` |
| Feature Enhancer, Eq. 5 | Specified five MobileNetV3 blocks, 1x1 projection followed directly by GAP, context64 | `models/feature_enhancer.py::SpectralContextEncoder` |
| Feature Enhancer, Eqs. 6–10 | Bounded gains/shift; depthwise kernel5 and pointwise band mixing; sigmoid temporal scale initialized0.05; signed graph residual; final validity mask | `models/feature_enhancer.py::FeatureEnhancer` |
| Graph Encoder, Eqs. 11–15 | Two attention layers; raw learned queries against LN electrode features; original values; 16x32 tokens; LN512 + Linear512→256 | `models/graph_encoder.py` |
| Graph Encoder, Eqs. 16–19 | Sinusoidal positions; 256→96→96→256 temporal blocks, kernels3/5, zero final BN scale; attention pooling; LN256 + Linear256→192 + GELU | `models/graph_encoder.py`, `models/network.py` |
| Cross-Subject Regularization, Eqs. 20–23 | Source-only subject classes; GRL slope10/max0.10; outer-log SupCon temperature0.07; eligible minibatch prototypes and ordered-pair hinge margin0.20 | `models/adversarial.py`, `models/schedules.py`, `losses.py` |
| KAN-Based Adaptive Readout, Eq. 24 | SiLU base plus eight cubic spline bases; independent base, spline and edge-scale parameters; uniform extended grid | `models/kan.py` |
| Quantum Residual, Eqs. 25–31 | Detached latent; compression192→64→32; KAN angles; RY/RZ/Rot per qubit; alternating ring CNOTs; ordered Z/X/ZZ readout; KAN12→32; residual32→192 and auxiliary32→C | `models/network.py`, `models/quantum.py` |
| Quantum Fusion, Eqs. 32–37 | Detached entropy, exponent1.5, bounded alpha0.25, warm-up10–30%, independent final KAN | `models/network.py`, `models/schedules.py` |
| Optimization Objective, Eqs. 38–40 / `eq:objective-total` | Unsmoothened batch-mean CE, raw auxiliary MSE against onehot minus detached probabilities, six learnable nonnegative auxiliary coefficients (author clarification) | `losses.py` |
| Experimental Setup | Five distinct seeds; S−1 source subjects; source validation; training-only normalization/augmentation; one target evaluation; seed averages within subject then population SD across subjects | `training/contract.py`, `training/engine.py`, `training/loso.py` |

Geometry is fixed in the model and CLI: 62 electrodes, 5 bands, 16 tokens, temporal256 and shared192. An override to the legacy shared256 architecture raises an error. Missing requested SEED-IV sessions raise an error rather than silently dropping sessions. Full LOSO checks the manuscript subject counts (15/15/16/20).

## Explicit implementation choices absent from the manuscript

The updated LaTeX still leaves the following choices unspecified. The defaults are recorded in every checkpoint and fold result; they are not established as the settings that generated the manuscript tables.

- Author clarification after supplying the LaTeX: the six auxiliary-loss coefficients are **learnable**. The original release's `nn.ParameterDict` and `lambda=max(raw_lambda,0)` parameterization are restored. Initial values are classical0.20, frontend0.25, subject0.20, contrastive0.08, prototype0.05, quantum0.15. They are optimized with the head AdamW group (initial LR5e-4, weight decay1e-4), receive gradients only from source-training batches, and are restored from the same selected epoch as the model. The emotion-loss coefficient remains one. The supplied LaTeX does not explicitly describe the parameterization, initial values or optimizer assignment; its comment about numerical coefficients should be revised to distinguish initialization from learned values.
- Source validation: stratified 80/20 trial split across the source pool, seeded by the experiment seed. Every source subject must remain represented in training. Selection uses validation accuracy with earliest-epoch tie breaking.
- Dataset scaling: per-band population mean/std fitted only on source training trials; padded windows remain zero.
- GAT internals: V4 dense attention with 8 heads, dropout0.30, LayerNorm and same-width residual. The manuscript names the two GAT layers but does not fully specify their internals.
- Fixed graph construction: bundled electrode coordinates and the inherited distance/symmetric-pair affinity rule. The manuscript specifies the nonnegative fixed matrix and its row normalization, but not this construction rule.
- Spectral numerical stabilization: standard deviations are clamped below by1e-4. Moving-average zero padding and interleaved sinusoidal positions are explicit in the code.
- Augmentation application probability0.85 and detailed sampling rules come from V4; the manuscript table specifies perturbation magnitudes only.
- Initialization of parameters other than those explicitly fixed in the manuscript follows PyTorch/V4 conventions.
- Across-subject SD uses the population convention (`ddof=0`). The manuscript describes the aggregation order but does not specify `ddof`.
- Validation uses the current epoch's quantum warm-up, and final evaluation restores the selected epoch's warm-up along with its weights. This avoids changing the evaluated fusion pathway after checkpoint selection; the manuscript does not separately specify inference scheduling.
- The PennyLane `default.qubit` simulation runs on CPU, including when the classical model is on CUDA. Device copies preserve autograd; no GPU quantum-simulation claim is made.

## Corrections relative to legacy V4

The updated equations require the spatial512→temporal256 projection and shared192 latent. V4 runner defaults instead used temporal512 and shared256. The current model has no unused inherited `fc_emotion` KAN and no unreported SEED-VII intensity head. The spectral context head now follows the stated direct 1x1→GAP path, without V4's additional BN/activation after that projection. The MA3 output is passed directly to the STFT as Eq.4 specifies; it is not masked again before the STFT. The final enhanced features are still masked as Eq.10 requires.

Quantum loss is raw MSE, replacing V4's SmoothL1(tanh(auxiliary)). Classification uses plain CE, replacing V4's label smoothing. Target-based checkpoint selection is replaced with source-only selection.

## Stored parameters and manuscript tables

Direct counts for this implementation with S−1 subject classes are:

| Dataset | Source subject classes | Parameters | Pure FP32 weights, MiB |
| --- | ---: | ---: | ---: |
| SEED | 14 | 468661 | 1.79 |
| SEED IV | 14 | 472599 | 1.80 |
| SEED V | 15 | 476730 | 1.82 |
| SEED VII | 19 | 485378 | 1.85 |

`quankan-size` recomputes these network counts and separately reports six trainable objective parameters. The loss-weight module is training-only and stored separately in the checkpoint. The supplied LaTeX's `tab:complexity-full` still lists 589366/614424/639739/691652, which reproduce the legacy V4 stored models. Those entries and the abstract/model-size prose describe a different implementation. They must not be presented as measurements of this code. MACs/FLOPs and matched-ablation counts require fresh profiling; this release does not invent replacement values.

The reported accuracy/F1 and statistical tables require their original checkpoint/config provenance or complete new experiments. Aligning a model to the written equations does not establish reproduction of those numbers.

## Artifacts and completion

Each fold saves its checkpoint (including `loss_weights` state), learned effective coefficients, training settings, runtime versions, epoch history with weight trajectories, selected warm-up, source-subject mapping, normalization statistics, and a separate NPZ with train/validation/test indices and target predictions. `test_evaluations=1` describes the actual final evaluation performed by the engine. Existing checkpoints are not overwritten. Full LOSO writes a summary only after all folds and five seeds finish.

Overrides to the specified training settings or a reduced subject set are labeled `development` in saved settings. The two-epoch synthetic smoke is an execution check. It is not a full LOSO run or evidence for manuscript performance. Training with the manuscript settings is also not certified as exact reproduction while the unspecified choices above remain unverified.
