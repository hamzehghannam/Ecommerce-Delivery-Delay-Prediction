
from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load original orders without changing source files
orders = pd.read_csv(
    DATA_DIR / "olist_orders_dataset.csv"
)

# Work on a separate copy
clean_orders = orders.copy()

# Convert date columns
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for column in date_columns:
    clean_orders[column] = pd.to_datetime(
        clean_orders[column],
        errors="coerce"
    )

print("=" * 60)
print("DATA CLEANING - ORDERS")
print("=" * 60)

print("Original rows:", len(orders))
print("Processed rows:", len(clean_orders))

print("\nMissing values after date conversion:")
print(clean_orders[date_columns].isna().sum())

print("\nDATA CLEANING COMPLETED")

# Create a separate folder for processed data
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Save processed orders without modifying raw data
output_path = PROCESSED_DIR / "orders_dates_cleaned.csv"

clean_orders.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nPROCESSED FILE SAVED")
print("File:", output_path)
print("Saved rows:", len(clean_orders))

# Document the cleaning decisions
cleaning_report = pd.DataFrame({
    "column": date_columns,
    "missing_before": [
        orders[col].isna().sum()
        for col in date_columns
    ],
    "missing_after": [
        clean_orders[col].isna().sum()
        for col in date_columns
    ],
    "action": "Converted to datetime; missing dates preserved",
    "reason": "Avoid inventing dates or deleting valid orders",
})

report_path = PROCESSED_DIR / "orders_cleaning_report.csv"

cleaning_report.to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nCLEANING REPORT SAVED")
print(cleaning_report.to_string(index=False))
print("Report:", report_path)
