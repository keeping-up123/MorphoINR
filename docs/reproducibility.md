# Reproducibility checklist

The source tree includes the `Human_Brain_prefrontal_cortex_151507` base AnnData file, its Phikon-enriched input, its first-stage aligned output, and the matching histology image.

1. Create the environment with `conda env create -f environment.yml` and activate `morphoinr`.
2. From the `MorphoINR/` root, verify that these files are present under `data/random/Human_Brain_prefrontal_cortex_151507/`:
   - `Human_Brain_prefrontal_cortex_151507.h5ad`
   - `Human_Brain_prefrontal_cortex_151507_with_phikon.h5ad`
   - `Human_Brain_prefrontal_cortex_151507_aealign_with_embeddings_phikon_zlocal.h5ad`
   - `tissue_hires_image.png`
3. Run `bash Scripts/train_aligned_film.sh` to train the aligned AE and the FiLM-SIREN in order.
4. The script reuses the included Phikon-enriched file, writes the aligned-AE output, loads the aligned-AE checkpoint, trains the FiLM-SIREN, and predicts the fixed 20% held-out evaluation split. Following the SUICA spatial-imputation protocol, this split is not used for hyperparameter tuning or model selection; it is reserved for final performance evaluation. The source code retains the internal `val` identifier for compatibility with the released AnnData split and Lightning metric keys.
5. Inspect checkpoints, TensorBoard logs, and reconstructed AnnData under `log/random/Human_Brain_prefrontal_cortex_151507/`.

To regenerate the Phikon-enriched input manually:

```bash
python Scripts/preprocess/extract_phikon_features.py \
  --h5ad data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507.h5ad \
  --image data/random/Human_Brain_prefrontal_cortex_151507/tissue_hires_image.png \
  --pixel_coord_key spatial_pixel \
  --output data/random/Human_Brain_prefrontal_cortex_151507/Human_Brain_prefrontal_cortex_151507_with_phikon.h5ad
```

To run the stages manually:

```bash
python Code/train.py \
  --mode embedder \
  --conf Configs/experiments/Human_Brain_prefrontal_cortex_151507/aealign.yaml

python Code/train.py \
  --mode inr_film \
  --conf Configs/experiments/Human_Brain_prefrontal_cortex_151507/inr_aealign_film_zlocal_siren.yaml
```

Record the dataset accession/license, Phikon model revision, GPU/CUDA version, random seed, and generated checkpoint paths for a paper-level reproducibility record.
