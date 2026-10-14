import argparse
import numpy as np
import torch
import anndata as ad
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def crop_patch(image, x, y, patch_size=224):
    """
    image: PIL RGB image
    x, y: pixel coordinates on tissue_hires_image.png
    """
    w, h = image.size
    r = patch_size // 2

    left = int(round(x)) - r
    upper = int(round(y)) - r
    right = left + patch_size
    lower = upper + patch_size

    patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))

    crop_left = max(left, 0)
    crop_upper = max(upper, 0)
    crop_right = min(right, w)
    crop_lower = min(lower, h)

    crop = image.crop((crop_left, crop_upper, crop_right, crop_lower))

    paste_x = max(0, -left)
    paste_y = max(0, -upper)
    patch.paste(crop, (paste_x, paste_y))

    return patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pixel_coord_key", default="spatial_pixel")
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    adata = ad.read_h5ad(args.h5ad)
    image = Image.open(args.image).convert("RGB")

    if args.pixel_coord_key not in adata.obsm:
        raise KeyError(
            f"{args.pixel_coord_key} not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    coords = adata.obsm[args.pixel_coord_key]
    coords = np.asarray(coords).astype(float)

    w, h = image.size
    print(f"Image size: width={w}, height={h}")
    print("coord x range:", coords[:, 0].min(), coords[:, 0].max())
    print("coord y range:", coords[:, 1].min(), coords[:, 1].max())

    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2").to(device)
    model.eval()

    all_features = []

    for start in tqdm(range(0, len(coords), args.batch_size)):
        end = min(start + args.batch_size, len(coords))
        batch_coords = coords[start:end]

        patches = [
            crop_patch(image, x, y, patch_size=args.patch_size)
            for x, y in batch_coords
        ]

        inputs = processor(patches, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # [B, 1024]

        all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    adata.obsm["phikon_cls"] = all_features

    adata.write_h5ad(args.output)
    print(f"Saved Phikon features to {args.output}")
    print(f"adata.obsm['phikon_cls'].shape = {all_features.shape}")


if __name__ == "__main__":
    main()