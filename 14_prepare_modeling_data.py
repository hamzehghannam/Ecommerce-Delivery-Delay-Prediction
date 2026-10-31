
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    PROCESSED_DIR / "orders_all_features.csv",
    parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)

print("=" * 60)
print("MODELING DATA PREPARATION")
print("=" * 60)

# Initial eligibility conditions
eligible = (
    df["order_status"].eq("delivered")
    & df["order_approved_at"].notna()
    & df["order_delivered_customer_date"].notna()
    & df["order_estimated_delivery_date"].notna()
    & (
        df["order_delivered_customer_date"]
        >= df["order_purchase_timestamp"]
    )
)

# Chronological inconsistencies
approval_after_delivery = (
    df["order_approved_at"]
    > df["order_delivered_customer_date"]
)

invalid_planned = (
    df["invalid_planned_duration"].eq(1)
)

temporal_anomaly = (
    approval_after_delivery | invalid_planned
)

# Keep only eligible, chronologically valid orders
model_df = df.loc[
    eligible & ~temporal_anomaly
].copy()

# Calculate the target from calendar dates
actual_date = (
    model_df["order_delivered_customer_date"]
    .dt.normalize()
)

estimated_date = (
    model_df["order_estimated_delivery_date"]
    .dt.normalize()
)

model_df["is_late"] = (
    actual_date > estimated_date
).astype(int)

# Validate modeling dataset
assert model_df["order_id"].is_unique
assert model_df["is_late"].isin([0, 1]).all()
assert model_df["order_approved_at"].notna().all()
assert model_df["planned_delivery_days"].ge(0).all()

print("Original orders:", len(df))
print("Initially eligible:", eligible.sum())
print(
    "Excluded temporal anomalies:",
    (eligible & temporal_anomaly).sum()
)
print("Final modeling orders:", len(model_df))

print("\nTarget distribution:")
print(model_df["is_late"].value_counts().sort_index())

# Save full modeling audit dataset
output_path = (
    PROCESSED_DIR / "modeling_data.csv"
)

model_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nMODELING DATA PREPARATION COMPLETED")
print("Saved file:", output_path)
