import scanpy as sc
import numpy as np
import anndata as ad
import sys
import os
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)
from utils import plot_slice

from lightning import pytorch as pl
pl.seed_everything(8848, workers=True)

def main():
    parser = argparse.ArgumentParser(description="Filter, normalize, and PCA-project an AnnData file.")
    parser.add_argument("--input", type=Path, required=True, help="Input H5AD file.")
    parser.add_argument("--output", type=Path, required=True, help="Output H5AD file.")
    parser.add_argument("--debug-figure", type=Path, default=None, help="Optional path for a spatial debug figure.")
    args = parser.parse_args()

    adata = sc.read_h5ad(args.input)
    print("raw shape:", adata.X.shape)
    sc.pp.filter_cells(adata, min_counts=200)
    sc.pp.filter_genes(adata, min_counts=3)
    print("after filtering:", adata.X.shape)

    count = adata.layers["count"].toarray()
    column_mask = np.all(count <= 200, axis=0)
    filtered_count = count[:, column_mask]
    print(filtered_count.shape)
    new_adata = ad.AnnData(X=filtered_count)
    new_adata.obsm["spatial"] = adata.obsm["spatial"]
    sc.pp.normalize_total(new_adata, target_sum=200)
    print(new_adata.X[new_adata.X > 0].min(), new_adata.X.max())

    if args.debug_figure is not None:
        args.debug_figure.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_slice(new_adata.obsm["spatial"], spot_size=1)
        fig.savefig(args.debug_figure)

    n_cells = new_adata.X.shape[0]
    train_idx, eval_idx = train_test_split(
        list(range(n_cells)), test_size=0.2, random_state=8848, shuffle=True
    )
    split = np.full(n_cells, "train", dtype=object)
    split[eval_idx] = "val"  # internal compatibility label for held-out evaluation
    new_adata.obs["split"] = split
    print(f"held-out evaluation spots: {len(eval_idx)}")

    pca = PCA(n_components=16, random_state=0)
    pca.fit(new_adata.X[train_idx, :])
    new_adata.obsm["embeddings"] = pca.transform(new_adata.X)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_adata.write_h5ad(args.output)


if __name__ == "__main__":
    main()
