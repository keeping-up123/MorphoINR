# Experiment outputs

This directory is the default output root configured by the experiment YAML files. A run creates a case directory containing:

- `02_aealign/`: aligned AE checkpoints, TensorBoard logs, and optional stage-1 metadata;
- `05_inr_aealign_film_zlocal_siren/` (or the `1142243F` `08_` stage): FiLM-SIREN checkpoints, decoder weights, and reconstructed AnnData output.

Large generated files are excluded from the ordinary source release. Use Git LFS or a GitHub Release if archived checkpoints and predictions are distributed.
