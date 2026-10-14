# Configurations

`experiments/<case>/` contains the reproducibility pair for each case:

1. `02_aealign.yaml` trains the aligned autoencoder and writes `embedded-all.h5ad`.
2. `05_inr_aealign_film_zlocal_siren.yaml` trains the FiLM-conditioned SIREN and writes `reconstructed-val.h5ad`.

The stage-1 file also declares `pipeline.embedded_output`, a stable path under `data_final/random/<case>/`. This removes the need to manually copy a file out of a Lightning `version_0` directory before starting stage 2. All relative paths assume commands are run from the `MorphoINR/` root.

The `1142243F` experiment uses the equivalent stage-2 file prefixed with `08_` because it follows the original run numbering. Paths are relative to the `MorphoINR/` project root.
