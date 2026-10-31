
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Load datasets
items = pd.read_csv(
    DATA_DIR / "olist_order_items_dataset.csv"
)

products = pd.read_csv(
    PROCESSED_DIR / "products_cleaned.csv"
)

sellers = pd.read_csv(
    DATA_DIR / "olist_sellers_dataset.csv"
)

orders = pd.read_csv(
    PROCESSED_DIR / "orders_merged.csv",
    usecols=["order_id", "customer_state"]
)

print("=" * 60)
print("PRODUCT AND SELLER FEATURE ENGINEERING")
print("=" * 60)

# Merge product information into order items
items_enriched = items.merge(
    products[
        ["product_id", "product_weight_g"]
    ],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Merge seller information
items_enriched = items_enriched.merge(
    sellers[["seller_id", "seller_state"]],
    on="seller_id",
    how="left",
    validate="many_to_one"
)

# Add customer state
items_enriched = items_enriched.merge(
    orders,
    on="order_id",
    how="left",
    validate="many_to_one"
)

assert len(items_enriched) == len(items)

# Compare seller and customer locations
items_enriched["same_state"] = np.where(
    items_enriched["seller_state"].notna()
    & items_enriched["customer_state"].notna(),
    (
        items_enriched["seller_state"]
        == items_enriched["customer_state"]
    ).astype(int),
    np.nan
)

# Aggregate features to one row per order
features = (
    items_enriched.groupby("order_id")
    .agg(
        total_product_weight_g=(
            "product_weight_g", "sum"
        ),
        average_product_weight_g=(
            "product_weight_g", "mean"
        ),
        same_state_item_ratio=(
            "same_state", "mean"
        ),
        distinct_seller_states=(
            "seller_state", "nunique"
        ),
    )
    .reset_index()
)

# Preserve missing totals when no weight is available
weight_totals = (
    items_enriched.groupby("order_id")["product_weight_g"]
    .sum(min_count=1)
)

features["total_product_weight_g"] = (
    features["order_id"].map(weight_totals)
)

assert features["order_id"].is_unique

print("Original item rows:", len(items))
print("Enriched item rows:", len(items_enriched))
print("Orders with generated features:", len(features))

print("\nMissing feature values:")
print(features.isna().sum().to_string())

output_path = (
    PROCESSED_DIR / "product_seller_features.csv"
)

features.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nPRODUCT SELLER FEATURES COMPLETED")
print("Saved file:", output_path)
