#!/usr/bin/env python3
import argparse
import anndata as ad
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_h5ad", required=True)
    parser.add_argument("--embedded_h5ad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["embeddings", "z_local"],
        help="obsm keys to copy from embedded_h5ad to base_h5ad",
    )
    args = parser.parse_args()

    base = ad.read_h5ad(args.base_h5ad)
    emb = ad.read_h5ad(args.embedded_h5ad)

    assert base.n_obs == emb.n_obs, f"n_obs mismatch: {base.n_obs} vs {emb.n_obs}"

    if len(base.obs_names) == len(emb.obs_names):
        # embedded-all.h5ad may not preserve obs_names, so only warn if mismatch
        if not np.all(base.obs_names.astype(str).values == emb.obs_names.astype(str).values):
            print("[WARN] obs_names are not identical. Assuming same order.")

    for key in args.keys:
        if key in emb.obsm:
            base.obsm[key] = emb.obsm[key]
            print(f"Copied obsm['{key}']: {base.obsm[key].shape}")
        else:
            print(f"[WARN] obsm['{key}'] not found in embedded_h5ad. Skipped.")

    base.write_h5ad(args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()