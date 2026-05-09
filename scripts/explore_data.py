"""
Exploratory data analysis for the CAMEO dataset.

Requires data/raw/ to be populated by download_data.py first.
Expected parquets:
  data/raw/products__test.parquet   — product corpus (one row per image)
  data/raw/reviews__test.parquet    — user reviews (used as query-like text)
"""

import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
IMAGE_DIR = DATA_RAW / "images"


def load(name: str):
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not installed.", file=sys.stderr)
        sys.exit(1)
    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_data.py first.", file=sys.stderr)
        sys.exit(1)
    return pd.read_parquet(path)


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Query length distribution
# ---------------------------------------------------------------------------

def query_length_distribution(reviews):
    section("1. Query Length Distribution (word count of processed_comment)")
    import pandas as pd

    text = reviews["processed_comment"].dropna()
    lengths = text.apply(lambda t: len(str(t).split()))

    bins  = [0,  2,  4,  6,  8, 10, 15, 20, 9999]
    labels = ["1-2","3-4","5-6","7-8","9-10","11-15","16-20","21+"]
    cut = pd.cut(lengths, bins=bins, labels=labels, right=True)
    dist = cut.value_counts().sort_index()

    print(f"  Total reviews : {len(lengths):,}\n")
    print(f"  {'Words':<10}  {'Count':>8}  Bar")
    print(f"  {'-'*10}  {'-'*8}  {'-'*30}")
    max_count = dist.max()
    for label, count in dist.items():
        bar = "#" * (count * 30 // max_count) if max_count else ""
        print(f"  {label:<10}  {count:>8,}  {bar}")

    print(f"\n  min={lengths.min()}  median={lengths.median():.0f}  "
          f"mean={lengths.mean():.1f}  max={lengths.max()}")


# ---------------------------------------------------------------------------
# 2. Sample queries
# ---------------------------------------------------------------------------

def sample_queries(reviews, n=10):
    section(f"2. {n} Sample Reviews (product_id + processed_comment)")
    sample = reviews[["product_id", "processed_comment", "rating"]].dropna(
        subset=["processed_comment"]
    ).sample(min(n, len(reviews)), random_state=42)

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"  [{i:2d}] product_id={row['product_id']}  rating={row['rating']}")
        text = str(row["processed_comment"])
        # show up to 120 chars
        print(f"        {text[:120]}{'...' if len(text) > 120 else ''}")
        print()


# ---------------------------------------------------------------------------
# 3. Coverage: products referenced in reviews vs corpus
# ---------------------------------------------------------------------------

def coverage_analysis(reviews, products):
    section("3. Review Coverage vs Product Corpus")

    corpus_ids   = set(products["product_id"].dropna().unique())
    reviewed_ids = set(reviews["product_id"].dropna().unique())

    n_corpus   = len(corpus_ids)
    n_reviewed = len(reviewed_ids)
    n_overlap  = len(corpus_ids & reviewed_ids)
    n_reviews_no_product = len(reviewed_ids - corpus_ids)

    print(f"  Unique products in corpus            : {n_corpus:,}")
    print(f"  Unique products with ≥1 review       : {n_reviewed:,}")
    print(f"  Corpus products that have reviews    : {n_overlap:,}  "
          f"({100*n_overlap/n_corpus:.1f}%)")
    print(f"  Reviewed products NOT in corpus      : {n_reviews_no_product:,}")

    reviews_per_product = reviews["product_id"].value_counts()
    print(f"\n  Reviews per product — min={reviews_per_product.min()}  "
          f"median={reviews_per_product.median():.0f}  "
          f"mean={reviews_per_product.mean():.1f}  "
          f"max={reviews_per_product.max()}")
    print("  Top-5 most-reviewed products:")
    for pid, cnt in reviews_per_product.head(5).items():
        print(f"    {pid}: {cnt:,} reviews")


# ---------------------------------------------------------------------------
# 4. Multiple images per product
# ---------------------------------------------------------------------------

def multi_image_analysis(products):
    section("4. Multiple Images per Product")

    imgs_per_product = products.groupby("product_id")["image_path"].count()
    total = len(imgs_per_product)
    multi = (imgs_per_product > 1).sum()
    dist  = imgs_per_product.value_counts().sort_index()

    print(f"  Unique products            : {total:,}")
    print(f"  Total image-path rows      : {len(products):,}")
    print(f"  Products with >1 image     : {multi:,}  ({100*multi/total:.1f}%)")
    print(f"\n  Images/product  Count")
    print(f"  {'-'*14}  -----")
    for n_img, cnt in dist.items():
        print(f"  {n_img:<14}  {cnt:,}")

    # What fraction are videos (.mp4)?
    total_paths = products["image_path"].dropna()
    n_mp4 = total_paths.str.endswith(".mp4").sum()
    n_jpg = total_paths.str.endswith(".jpg").sum()
    print(f"\n  File types in image_path:")
    print(f"    .jpg : {n_jpg:,}")
    print(f"    .mp4 : {n_mp4:,}")
    other = len(total_paths) - n_jpg - n_mp4
    if other:
        print(f"    other: {other:,}")

    # Note about downloading
    print(f"\n  NOTE: image_path contains relative paths (no base URL in dataset).")
    print(f"        Images were not downloaded — need CDN base URL.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import pandas as pd  # noqa — early import to check dependency

    print("Loading parquets ...")
    products = load("products__test")
    reviews  = load("reviews__test")

    print(f"  products : {len(products):,} rows | {products['product_id'].nunique():,} unique products")
    print(f"  reviews  : {len(reviews):,} rows | {reviews['product_id'].nunique():,} unique products")

    query_length_distribution(reviews)
    sample_queries(reviews)
    coverage_analysis(reviews, products)
    multi_image_analysis(products)

    print("\nDone.")


if __name__ == "__main__":
    main()
