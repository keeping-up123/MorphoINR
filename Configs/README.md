# Configurations

The release contains the two fixed configurations used for the `Human_Brain_prefrontal_cortex_151507` example:

1. `experiments/Human_Brain_prefrontal_cortex_151507/aealign.yaml` trains the histology-aligned autoencoder;
2. `experiments/Human_Brain_prefrontal_cortex_151507/inr_aealign_film_zlocal_siren.yaml` trains the FiLM-conditioned SIREN and predicts the held-out evaluation set.

All paths are explicit and relative to the `MorphoINR/` project root:

```text
data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507.h5ad
data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507_with_phikon.h5ad
data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507_aealign_with_embeddings_phikon_zlocal.h5ad
log/random/Human_Brain_prefrontal_cortex_151507/aealign/
log/random/Human_Brain_prefrontal_cortex_151507/inr_aealign_film_zlocal_siren/
```

The example base, Phikon-enriched, and aligned-output `.h5ad` files are included. The end-to-end script reuses the Phikon-enriched input, then runs both configurations in order.

The included example follows the SUICA spatial-imputation protocol: a fixed 80/20 split is used, with 80% of spots for model fitting and 20% held out for evaluation. The held-out subset is reserved for final evaluation rather than hyperparameter tuning or model selection. The aligned autoencoder uses a Dice-loss weight of `0.05`, while decoder fine-tuning uses `0.01`, matching the YAML configurations. The configuration and implementation retain the internal `val` name for compatibility with the released AnnData split and Lightning metric keys.
