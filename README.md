# MorphoINR

Histology-guided continuous spatial transcriptomics imputation via cross-modal aligned latent implicit neural representations.

MorphoINR is the reproducibility package for the aligned autoencoder (AE) + FiLM-conditioned SIREN pipeline. The repository contains the model code, experiment configurations, analysis scripts, selected generated artifacts, checkpoints, paper figures, and reproducibility notes. Licensed raw datasets are downloaded separately from their official providers.

![MorphoINR framework](./Pic/framework.png)

## Quick result validation

The following command lists the available aligned experiments and project-local artifacts:

```bash
find Configs/experiments -mindepth 2 -maxdepth 2 -name '*.yaml' | sort
find logs_final/random -mindepth 1 -maxdepth 1 -type d | sort
```

The repository defines the generated-output locations under `data_final/` and `logs_final/`; large generated files are optional release artifacts and are not required to run the training code from raw data.

## Environment setup

```bash
conda env create -f environment.yml
conda activate morphoinr
```

The Conda file installs the pinned Python packages automatically. If Conda is not available, create a Python 3.9 environment first and install the same pinned packages with `pip install -r requirements.txt`.

The tested reference environment uses Python 3.9.25, PyTorch 2.8.0, Lightning 2.6.0, Scanpy 1.10.3, AnnData 0.10.9, and CUDA 12.x. GPU training requires a CUDA-compatible PyTorch installation; for another operating system or CUDA version, replace only the PyTorch wheel as described in `environment.yml`.

Main packages include PyTorch, Lightning, Scanpy/AnnData, SciPy, NumPy, pandas, scikit-learn, OmegaConf, Transformers, TensorBoard, and Matplotlib. The complete unconstrained package list is in [requirements.txt](requirements.txt).

## Data

Raw data are not committed to GitHub. Download the data from the official sources and read [data/README.md](data/README.md) before running an experiment.

| Dataset group | Source |
|---|---|
| Human prefrontal cortex | [Globus file manager](https://app.globus.org/file-manager?origin_id=abdefb8c-731b-4907-a89d-9b5649d9967a&origin_path=%2F&two_pane=false) |
| Human breast cancer | [Zenodo record 4739739](https://zenodo.org/records/4739739) |

The training code expects AnnData `.h5ad` files with spatial coordinates in `adata.obsm["spatial"]`. Histology features are read from the configured image key, and the aligned stage-1 output contains `adata.obsm["embeddings"]` and `adata.obsm["z_local"]`.

The links above document the currently confirmed Human prefrontal cortex and Human breast cancer sources. Add the official source and accession for every additional tissue included in a final release.

## Reproducing the aligned AE + FiLM-SIREN pipeline

Run commands from the `MorphoINR/` root. Each dataset has a pair of YAML files under `Configs/experiments/<case>/`:

1. Train the aligned AE and export the latent representation.
2. Train the FiLM-conditioned SIREN using the aligned latent representation and the AE checkpoint.
3. Evaluate the generated AnnData files with the analysis scripts.

Example for mouse breast cancer sample `1142243F`:

```bash
python Code/train.py \
  --mode embedder \
  --conf Configs/experiments/1142243F/02_aealign.yaml

python Code/train.py \
  --mode inr_film \
  --conf Configs/experiments/1142243F/08_inr_aealign_film_zlocal_siren.yaml
```

For a single-case end-to-end run, use the path-resolving helper:

```bash
bash Scripts/train_aligned_film.sh 1142243F
```

For the other cases, use the corresponding `02_aealign.yaml` and `05_inr_aealign_film_zlocal_siren.yaml` files. The stage-1 configuration reads external data from `data/raw/`; the stage-2 configuration consumes the generated file under `data_final/` and writes to `logs_final/`.

## Outputs

- `data_final/`: generated aligned AE intermediate AnnData files (created during reproduction).
- `logs_final/`: optional checkpoints, latent embeddings, reconstructed AnnData files, and training metadata.
- `results/final_figures/` and `results/final_tables/`: selected publication outputs.

The checkpoint and AnnData artifacts are large. Use Git LFS or a GitHub Release instead of ordinary Git blobs for files exceeding GitHub's regular file-size limit. See [checkpoints/README.md](checkpoints/README.md).

## Applying MorphoINR to a custom dataset

Prepare an AnnData object containing spatial coordinates and the configured histology feature matrix, then create a dataset-specific YAML by copying one of the files under `Configs/experiments/`. Set `dataset.data_file`, the feature key, the output directory, and the optimization parameters. Train the two stages in order with `Code/train.py` as shown above.

## Repository layout

```text
MorphoINR/
├── train.py       # Root convenience entry point
├── Code/          # Dataset loader, AE, SIREN, training systems
├── Configs/       # Shared and per-dataset YAML configurations
├── Scripts/       # Training helper, preprocessing, analysis, validation
├── data/          # Official download links and data conventions
├── data_final/    # Generated aligned intermediate data location
├── logs_final/    # Curated checkpoints and generated outputs
├── results/       # Figures, tables, and result notes
├── docs/          # Paper and reproducibility checklist
└── Pic/           # Method figure
```

## Citation and license

Please cite the MorphoINR paper and the original dataset publications. Dataset terms are defined by their providers and are not replaced by this repository's license. Complete author metadata in [CITATION.cff](CITATION.cff) before the public release.

## Acknowledgment

This project is based on and extends the SUICA framework for continuous spatial transcriptomics modeling. In particular, the present repository organizes the aligned autoencoder and histology-conditioned FiLM-SIREN implementation used for the MorphoINR experiments. Please also cite the original SUICA paper:

> Qingtian Zhu, Yumin Zheng, Yuling Sang, Yifan Zhan, Ziyan Zhu, Jun Ding, and Yinqiang Zheng. “SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics.” *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267, 2025. arXiv:2412.01124.
