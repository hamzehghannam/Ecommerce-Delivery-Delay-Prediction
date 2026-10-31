
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Load datasets
orders = pd.read_csv(
    PROCESSED_DIR / "orders_dates_cleaned.csv"
)

customers = pd.read_csv(
    DATA_DIR / "olist_customers_dataset.csv"
)

items = pd.read_csv(
    PROCESSED_DIR / "order_items_aggregated.csv"
)

payments = pd.read_csv(
    PROCESSED_DIR / "payments_aggregated.csv"
)

print("=" * 60)
print("DATASET MERGING")
print("=" * 60)

print("Original orders:", len(orders))

# Check that join keys are unique
assert orders["order_id"].is_unique
assert customers["customer_id"].is_unique
assert items["order_id"].is_unique
assert payments["order_id"].is_unique

# Merge orders with customers
merged = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

print("\nAfter customers merge:", len(merged))

# Merge aggregated items
merged = merged.merge(
    items,
    on="order_id",
    how="left",
    validate="one_to_one"
)

print("After items merge:", len(merged))

# Merge aggregated payments
merged = merged.merge(
    payments,
    on="order_id",
    how="left",
    validate="one_to_one"
)

print("After payments merge:", len(merged))

# Verify row count and uniqueness
assert len(merged) == len(orders)
assert merged["order_id"].is_unique

# Check that no customer was lost
assert merged["customer_unique_id"].notna().all()

# Check for unmatched orders
print("\nOrders without items:", merged["item_count"].isna().sum())
print("Orders without payments:", merged["payment_count"].isna().sum())

# Validate totals
assert abs(
    merged["total_price"].sum()
    - items["total_price"].sum()
) < 0.01

assert abs(
    merged["total_freight"].sum()
    - items["total_freight"].sum()
) < 0.01

assert abs(
    merged["total_payment"].sum()
    - payments["total_payment"].sum()
) < 0.01

print("\nVALIDATION RESULTS:")
print("Final rows:", len(merged))
print("Unique orders:", merged["order_id"].nunique())
print("Duplicate orders:", merged["order_id"].duplicated().sum())
print("Total price:", round(merged["total_price"].sum(), 2))
print("Total freight:", round(merged["total_freight"].sum(), 2))
print("Total payments:", round(merged["total_payment"].sum(), 2))

# Save merged dataset
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "orders_merged.csv"

merged.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nMERGING COMPLETED")
print("Saved file:", output_path)

print("\n" + "=" * 60)
print("ORDERS WITHOUT ITEMS ANALYSIS")
print("=" * 60)

# Orders without item records
orders_without_items = merged[
    merged["item_count"].isna()
].copy()

print("\nTotal orders without items:", len(orders_without_items))

print("\nOrder status distribution:")
print(
    orders_without_items["order_status"]
    .value_counts()
    .to_string()
)

print("\nDelivered orders without items:")
print(
    (
        orders_without_items["order_status"]
        == "delivered"
    ).sum()
)

# Orders without payment records
orders_without_payments = merged[
    merged["payment_count"].isna()
]

print("\nOrders without payments:")
print(len(orders_without_payments))

print("\nStatus of orders without payments:")
print(
    orders_without_payments["order_status"]
    .value_counts()
    .to_string()
)

print("\nMISSING RELATIONSHIP ANALYSIS COMPLETED")
