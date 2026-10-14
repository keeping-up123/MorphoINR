# Code

The executable model implementation is split into three layers:

- `datasets.py`: AnnData readers and the `ST2D` dataset used by both stages.
- `networks/`: aligned autoencoder, histology projector, and SIREN modules.
- `systems/`: Lightning training systems for the aligned AE and FiLM-SIREN stages.
- `train.py`: command-line dispatcher for `embedder` and `inr_film` modes.

From the project root, the equivalent convenience entry point is `python train.py`.
