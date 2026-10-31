
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("MACHINE LEARNING FEATURE PREPARATION")
print("=" * 60)

# Candidate features available or assumed available
# at the time an order is approved.
NUMERIC_FEATURES = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "planned_delivery_days",
    "item_count",
    "seller_count",
    "product_count",
    "order_total_value",
    "average_item_price",
    "freight_ratio",
    "multiple_sellers",
    "multiple_payments",
    "weekend_approval",
    "total_product_weight_g",
    "average_product_weight_g",
    "same_state_item_ratio",
    "distinct_seller_states",
    "payment_method_count",
    "max_installments"
]

CATEGORICAL_FEATURES = [
    "customer_state"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_COLUMNS = {
    "order_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "is_late",
    "order_id",
    "customer_id",
    "customer_unique_id"
}

assert not (set(FEATURE_COLUMNS) & FORBIDDEN_COLUMNS)

for split_name in ["train", "validation", "test"]:
    file_path = DATA_DIR / f"{split_name}_data.csv"

    df = pd.read_csv(file_path)

    missing_columns = [
        col for col in FEATURE_COLUMNS + ["is_late"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {split_name}: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["is_late"].astype(int)

    X.to_csv(
        DATA_DIR / f"X_{split_name}.csv",
        index=False
    )

    y.to_frame().to_csv(
        DATA_DIR / f"y_{split_name}.csv",
        index=False
    )

    print(f"\n{split_name.upper()}")
    print("Rows:", len(X))
    print("Features:", X.shape[1])
    print("Missing feature values:", int(X.isna().sum().sum()))
    print("Late orders:", int(y.sum()))

print("\nFEATURE PREPARATION COMPLETED")
