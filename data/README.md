# Data

Raw datasets are intentionally not included in this repository because they are large and may have separate data-use licenses. Download them from the official sources below, then place the prepared files at the paths referenced by the YAML configurations.

Place downloaded or locally prepared files under `data/raw/<case>/`, or point the YAML configuration files to an external data directory. The training code expects AnnData `.h5ad` files with spatial coordinates in `adata.obsm["spatial"]`; optional image/embedding features use the keys documented in `Code/datasets.py`.

## Official download sources

| Dataset | Source | Notes |
|---|---|---|
| Human prefrontal cortex | [Globus file manager](https://app.globus.org/file-manager?origin_id=abdefb8c-731b-4907-a89d-9b5649d9967a&origin_path=%2F&two_pane=false) | Download with Globus; access may require authentication and acceptance of the provider's data-use terms. |
| Human breast cancer | [Zenodo record 4739739](https://zenodo.org/records/4739739) | Use the files and version information provided by the record. Cite the Zenodo record in addition to the paper. |

Before a final release, add the exact downloaded filenames, accession/version, license, preprocessing command, and SHA-256 checksums for every dataset used in the paper.

## Files used by the aligned AE + FiLM-SIREN pipeline

For each case listed in `Configs/experiments/`, the aligned AE stage generates:

```text
data_final/random/<case>/<case>_aealign_with_embeddings_phikon_zlocal.h5ad
```

The raw `<case>_with_phikon.h5ad` inputs are intentionally external because of dataset size and licensing. The generated stage-1 output used by FiLM-INR contains `adata.obsm["embeddings"]` and `adata.obsm["z_local"]`. Obtain raw inputs from the official dataset sources above and place them under `data/raw/<case>/` or update `dataset.data_file` in the YAML file.
