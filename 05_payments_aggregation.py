
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

payments = pd.read_csv(
    DATA_DIR / "olist_order_payments_dataset.csv"
)

print("=" * 60)
print("PAYMENTS AGGREGATION")
print("=" * 60)

print("Original payment rows:", len(payments))
print("Unique orders:", payments["order_id"].nunique())

# Aggregate payment records to order level
payments_aggregated = (
    payments.groupby("order_id")
    .agg(
        payment_count=("payment_sequential", "count"),
        total_payment=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_method_count=("payment_type", "nunique"),
    )
    .reset_index()
)

# Validate aggregation
assert payments_aggregated["order_id"].is_unique

assert (
    payments_aggregated["payment_count"].sum()
    == len(payments)
)

assert abs(
    payments_aggregated["total_payment"].sum()
    - payments["payment_value"].sum()
) < 0.01

print("\nAggregated orders:", len(payments_aggregated))

print(
    "Duplicate order IDs:",
    payments_aggregated["order_id"].duplicated().sum()
)

print(
    "Total payment records:",
    payments_aggregated["payment_count"].sum()
)

print(
    "Total payment value:",
    round(payments_aggregated["total_payment"].sum(), 2)
)

print(
    "Orders with multiple payment records:",
    (payments_aggregated["payment_count"] > 1).sum()
)

print(
    "Orders with multiple payment methods:",
    (payments_aggregated["payment_method_count"] > 1).sum()
)

# Save processed file
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "payments_aggregated.csv"

payments_aggregated.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nPAYMENTS AGGREGATION COMPLETED")
print("Saved file:", output_path)
