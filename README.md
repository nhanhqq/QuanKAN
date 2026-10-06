# QuanKAN

Camera-ready implementation package for the QuanKAN EEG emotion-recognition experiments. The model takes precomputed DE-LDS features with shape [batch, time, 62, 5]; raw EEG preprocessing is outside the model.

## Install and run

Install the Python dependencies with pip install -r requirements.txt. The four loaders expect the public dataset files in their original layouts:

- SEED: <root>/ExtractedFeatures_1s
- SEED IV: <root>/1, <root>/2, and <root>/3
- SEED V: <root>/EEG_DE_features
- SEED VII: <root>/EEG_features

Run all LOSO folds for one dataset by supplying the five seed IDs used for the experiment:

    python -m quankan.train_loso --dataset seediv --data-root D:\data\SEED_IV --seeds <seed1> <seed2> <seed3> <seed4> <seed5>

Run the method-contract tests with:

    python -m pytest -q

The runner splits source-subject samples into training and validation sets, selects a checkpoint using source validation only, and evaluates each held-out subject after selection. It reports the average across five seeds per held-out subject, followed by the mean and population SD across subjects. Checkpoints and JSON results go under outputs/<dataset>/.

The defaults follow the paper's implementation table for batch size, epoch limit, optimizer, learning rates, weight decay, and augmentation magnitudes. The validation fraction is 0.20, taken from the strict-LOSO source runner. The five seed IDs are required because the paper reports their count but does not list their values.

## Method mapping

- quankan/model.py implements the V4 feature enhancer, KAN mappings, graph/spatial/temporal path, entropy gate, and four-qubit circuit.
- quankan/losses.py implements cross-entropy, supervised contrastive, prototype-alignment, and quantum residual objectives.
- quankan/data.py loads the four DE-LDS datasets, builds the electrode-affinity graph, and applies source-only feature standardization.
- quankan/train_loso.py runs source-validated LOSO and evaluates held-out subjects after checkpoint selection.

The six nonnegative loss coefficients are trainable parameters, initialized to the values used by QuanKAN_main/train_v4.py: classical 0.20, frontend 0.25, subject 0.20, supervised contrastive 0.08, prototype 0.05, and quantum 0.15. The quantum auxiliary loss follows the Methodology equation and uses mean squared error on the raw residual prediction.

## Paper/source alignment notes

The implementation follows the Methodology dimensions: 16 spatial tokens of 32 features (512 concatenated), a 256-dimensional temporal path, and a 192-dimensional shared embedding. These dimensions conflict with the V4 source implementation, so V4 dimensions are not used here.

The V4 source uses smooth_l1_loss(tanh(aux_logits), residual_target) for the quantum auxiliary objective; the paper equation specifies mean squared error without tanh. This package follows the paper equation. The source V4 training runner also selects checkpoints on the held-out subject, whereas this package follows the paper's source-only model-selection protocol.

The Methodology specifies nonnegative loss coefficients but does not provide their numerical values or learning parameterization. Following the user's clarification, this package learns the coefficients, initializes them from the V4 source values, and constrains their effective values to be nonnegative. The validation fraction and exact seed IDs are also not stated in the paper; the fraction follows the source strict-LOSO runner, and seed IDs must be provided at run time.

With Methodology dimensions, the model has 468,789, 472,727, 476,858, and 485,506 parameters for (classes, source subjects) configurations (3,14), (4,14), (5,15), and (7,19). The manuscript reports 589,366--691,652 parameters, a substantial internal inconsistency that cannot be reconciled without clarifying the paper's architecture/counting configuration. The Methodology dimensions are prioritized; no undocumented parameters are added to force the published counts.
