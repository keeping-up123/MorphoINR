# Human_Brain_prefrontal_cortex_151507

This directory contains the example inputs and intermediate output used by the MorphoINR pipeline:

```text
Human_Brain_prefrontal_cortex_151507.h5ad
Human_Brain_prefrontal_cortex_151507_with_phikon.h5ad
Human_Brain_prefrontal_cortex_151507_aealign_with_embeddings_phikon_zlocal.h5ad
tissue_hires_image.png
```

The base file contains the spatial-transcriptomics matrix and coordinates. The `_with_phikon` file contains the 1,024-dimensional histology features. The `_aealign_with_embeddings_phikon_zlocal` file contains the aligned 64-dimensional latent representation and local conditioning code. The image is the matching histology image for `spatial_pixel` patch extraction.

The end-to-end command is:

```bash
bash Scripts/train_aligned_film.sh
```

Training logs, checkpoints, and reconstructed outputs are written outside this directory under `log/random/` and are ignored by default.
