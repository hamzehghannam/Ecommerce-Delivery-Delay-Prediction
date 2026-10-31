
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

items = pd.read_csv(
    DATA_DIR / "olist_order_items_dataset.csv"
)

print("=" * 60)
print("ORDER ITEMS AGGREGATION")
print("=" * 60)

print("Original item rows:", len(items))
print("Unique orders:", items["order_id"].nunique())

# Aggregate items to one row per order
items_aggregated = (
    items.groupby("order_id")
    .agg(
        item_count=("order_item_id", "count"),
        seller_count=("seller_id", "nunique"),
        product_count=("product_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    .reset_index()
)

items_aggregated["total_items_value"] = (
    items_aggregated["total_price"]
    + items_aggregated["total_freight"]
)

# Validate aggregation
assert items_aggregated["order_id"].is_unique
assert items_aggregated["item_count"].sum() == len(items)

assert abs(
    items_aggregated["total_price"].sum()
    - items["price"].sum()
) < 0.01

assert abs(
    items_aggregated["total_freight"].sum()
    - items["freight_value"].sum()
) < 0.01

print("\nAggregated orders:", len(items_aggregated))
print("Duplicate order IDs:",
      items_aggregated["order_id"].duplicated().sum())

print("\nTotal items:", items_aggregated["item_count"].sum())
print("Total price:", round(items_aggregated["total_price"].sum(), 2))
print("Total freight:", round(items_aggregated["total_freight"].sum(), 2))

print("\nOrders with multiple sellers:",
      (items_aggregated["seller_count"] > 1).sum())

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "order_items_aggregated.csv"

items_aggregated.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nAGGREGATION COMPLETED")
print("Saved file:", output_path)
