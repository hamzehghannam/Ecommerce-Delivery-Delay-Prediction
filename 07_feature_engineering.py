
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load the merged dataset
df = pd.read_csv(
    PROCESSED_DIR / "orders_merged.csv"
)

# Convert date columns
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print("Original rows:", len(df))

# 1. Day of approval
df["approval_day_of_week"] = (
    df["order_approved_at"].dt.day_of_week
)

# 2. Approval month
df["approval_month"] = (
    df["order_approved_at"].dt.month
)

# 3. Approval hour
df["approval_hour"] = (
    df["order_approved_at"].dt.hour
)


# 4. Planned delivery duration
df["planned_delivery_days"] = (
    df["order_estimated_delivery_date"]
    - df["order_approved_at"]
).dt.total_seconds() / 86400

# Flag negative durations before cleaning
df["invalid_planned_duration"] = (
    df["planned_delivery_days"] < 0
).astype(int)

# Preserve invalid values as missing for modeling
df.loc[
    df["planned_delivery_days"] < 0,
    "planned_delivery_days"
] = np.nan


# 5. Total order value
df["order_total_value"] = (
    df["total_price"] + df["total_freight"]
)

# 6. Average item price
df["average_item_price"] = (
    df["total_price"] / df["item_count"].replace(0, np.nan)
)

# 7. Freight ratio
df["freight_ratio"] = np.where(
    df["order_total_value"] > 0,
    df["total_freight"] / df["order_total_value"],
    np.nan
)

# 8. Multiple sellers indicator
df["multiple_sellers"] = np.where(
    df["seller_count"].isna(),
    np.nan,
    (df["seller_count"] > 1).astype(int)
)

# 9. Multiple payments indicator
df["multiple_payments"] = np.where(
    df["payment_count"].isna(),
    np.nan,
    (df["payment_count"] > 1).astype(int)
)

# 10. Weekend approval indicator
df["weekend_approval"] = np.where(
    df["order_approved_at"].isna(),
    np.nan,
    df["approval_day_of_week"].isin([5, 6]).astype(int)
)

feature_columns = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "planned_delivery_days",
    "order_total_value",
    "average_item_price",
    "freight_ratio",
    "multiple_sellers",
    "multiple_payments",
    "weekend_approval",
]

# Validate row count and unique orders
assert len(df) == 99441
assert df["order_id"].is_unique

print("\nCreated features:", len(feature_columns))

print("\nMissing values in new features:")
print(df[feature_columns].isna().sum().to_string())

print("\nFeature sample:")
print(df[feature_columns].head().to_string(index=False))

# Save the dataset
output_path = PROCESSED_DIR / "orders_with_features.csv"

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nFEATURE ENGINEERING COMPLETED")
print("Saved file:", output_path)
print("Final rows:", len(df))
