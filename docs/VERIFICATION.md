# Updated-manuscript implementation checks

See [verification.json](verification.json) for machine-readable evidence and hashes.

- Environment: Python 3.10.20, Torch 2.11.0+cpu, PennyLane 0.42.3.
- 23 tests passed. Ruff check/format, compilation, CLI help and wheel build passed.
- An installed wheel imported from outside the repository, resolved its bundled config/coordinates and completed a model forward pass.
- All four real-data loaders completed: SEED675 trials/15 subjects; SEED-IV1080/15; SEED-V720/16; SEED-VII1600/20. Features were finite and labels covered the expected classes.
- Historical check before the author clarified learnable loss weights: a two-epoch real-data SEED-IV fold completed on all three sessions: 806 source-training trials, 202 source-validation trials, 72 target-test trials. The source-validation checkpoint was selected at epoch2; the target was evaluated once. The checkpoint reloaded with `weights_only=True`.
- Current revision: a two-epoch synthetic fold completed with all six learnable coefficients receiving updates. Their state was restored from the source-validation-selected epoch and reloaded from the checkpoint. Regression checks cover detached quantum gradients, gate/CNOT/observable ordering, padding, the STFT formula, the direct context projection/GAP, prototype empty sets, geometry, parameter counts, split isolation and persisted checkpoint reload.
- Original V4 files match their recorded SHA256 hashes; the archived paper PDF matches its Git HEAD blob.

Execution artifacts are in the temporary directories recorded in the JSON. They are explicitly development checks. No published accuracy/F1 claim follows from these short runs, and no full five-seed LOSO campaign was executed. Verified execution is on CPU; a CUDA performance/execution claim has not been made.

The scientific authority and unspecified implementation choices are documented in [METHOD_CONTRACT.md](../METHOD_CONTRACT.md). The author has clarified that loss weights are learned. The supplied LaTeX still does not describe their parameterization or initialization; legacy model-size entries also remain unresolved as exact-reproduction evidence. The real-data check above predates this correction; no new real-data campaign was run for the learnable-weight revision.
