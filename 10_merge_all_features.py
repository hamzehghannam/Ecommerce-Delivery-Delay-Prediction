
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

orders = pd.read_csv(
    PROCESSED_DIR / "orders_with_features.csv"
)

product_features = pd.read_csv(
    PROCESSED_DIR / "product_seller_features.csv"
)

print("=" * 60)
print("FINAL FEATURE MERGING")
print("=" * 60)

print("Orders before merging:", len(orders))
print("Product feature orders:", len(product_features))

# Validate unique order IDs
assert orders["order_id"].is_unique
assert product_features["order_id"].is_unique

# Merge all features
final_df = orders.merge(
    product_features,
    on="order_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

# Check unmatched orders
print("\nMerge status:")
print(final_df["_merge"].value_counts().to_string())

# Remove merge indicator
final_df = final_df.drop(columns=["_merge"])

# Validate no duplicated orders
assert len(final_df) == len(orders)
assert final_df["order_id"].is_unique

# Validate existing financial totals
for column in [
    "total_price",
    "total_freight",
    "total_payment"
]:
    assert np.isclose(
        final_df[column].sum(),
        orders[column].sum()
    )

new_features = [
    "total_product_weight_g",
    "average_product_weight_g",
    "same_state_item_ratio",
    "distinct_seller_states",
]

print("\nMissing values in new features:")
print(final_df[new_features].isna().sum().to_string())

print("\nFINAL VALIDATION:")
print("Final rows:", len(final_df))
print("Unique orders:", final_df["order_id"].nunique())
print("Duplicate orders:", final_df["order_id"].duplicated().sum())

output_path = (
    PROCESSED_DIR / "orders_all_features.csv"
)

final_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nFINAL FEATURE MERGING COMPLETED")
print("Saved file:", output_path)
