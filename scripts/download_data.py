"""
Download the allietran/CAMEO dataset from HuggingFace.

Saves:
  data/raw/           — parquet files for each split
  data/raw/images/    — product images (named <product_id>_<idx>.jpg)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
IMAGE_DIR = DATA_RAW / "images"

DATASET_NAME = "allietran/CAMEO"
CONFIGS = ["products", "reviews"]


def download_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    all_splits = {}
    for cfg in CONFIGS:
        print(f"Loading {DATASET_NAME} config='{cfg}' ...")
        ds = load_dataset(DATASET_NAME, cfg)
        for split_name, split_data in ds.items():
            key = f"{cfg}__{split_name}"
            all_splits[key] = split_data
            print(f"  {key}: {len(split_data):,} rows | columns: {list(split_data.features)}")
    return all_splits


def save_splits(all_splits):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for key, split_data in all_splits.items():
        out_path = DATA_RAW / f"{key}.parquet"
        split_data.to_parquet(str(out_path))
        print(f"  Saved {key}: {len(split_data):,} rows → {out_path.relative_to(ROOT)}")


def download_images():
    """Download images.zip from the HF repo and extract to data/raw/images/."""
    import zipfile
    from huggingface_hub import hf_hub_download

    zip_path = DATA_RAW / "images.zip"

    if not zip_path.exists():
        print("Downloading images.zip from HuggingFace repo ...")
        hf_hub_download(
            repo_id=DATASET_NAME,
            filename="images.zip",
            repo_type="dataset",
            local_dir=str(DATA_RAW),
        )
        print(f"  Saved to {zip_path.relative_to(ROOT)}")
    else:
        print(f"  images.zip already exists, skipping download.")

    if not IMAGE_DIR.exists() or not any(IMAGE_DIR.rglob("*.jpg")):
        print("Extracting images.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(DATA_RAW))
        print("  Extraction complete.")
    else:
        print("  images/ already extracted, skipping.")

    n_jpg = len(list(IMAGE_DIR.rglob("*.jpg")))
    n_mp4 = len(list(IMAGE_DIR.rglob("*.mp4")))
    return n_jpg, n_mp4


def count_queries(all_splits):
    """Return number of CIR query rows across query-like splits."""
    total = 0
    for name, split in all_splits.items():
        feats = set(split.features.keys())
        if feats & {"query", "query_text", "modification", "caption", "text", "review", "processed_comment"}:
            total += len(split)
            print(f"  Query/review split '{name}': {len(split):,} rows  columns={list(split.features)}")
    return total


def print_summary(all_splits, n_jpg, n_mp4):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    for name, split in all_splits.items():
        print(f"  {name:30s}: {len(split):>6,} rows | columns: {list(split.features)}")

    n_queries = count_queries(all_splits)

    print(f"\n  JPEG images on disk    : {n_jpg:,}")
    print(f"  MP4 videos on disk     : {n_mp4:,}")
    print(f"  Review/query rows      : {n_queries:,}")
    print("=" * 60)


def main():
    all_splits = download_dataset()
    print("\nSaving splits to parquet ...")
    save_splits(all_splits)
    print("\nDownloading images ...")
    n_jpg, n_mp4 = download_images()
    print_summary(all_splits, n_jpg, n_mp4)


if __name__ == "__main__":
    main()
