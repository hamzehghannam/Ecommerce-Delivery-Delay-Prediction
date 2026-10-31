
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    PROCESSED_DIR / "orders_all_features.csv"
)

# Features available at order approval
# The assignment assumes item, payment and expected
# delivery information is available at approval.

model_features = [
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
    "customer_state",
    "payment_method_count",
    "max_installments",
]

# Fields that must never be used as model inputs
forbidden_features = {
    "order_status",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "is_late",
    "order_id",
    "customer_id",
    "customer_unique_id",
}

assert not forbidden_features.intersection(model_features)

assert all(
    feature in df.columns
    for feature in model_features
)

X = df[model_features].copy()

print("=" * 60)
print("MODEL FEATURE SAFETY CHECK")
print("=" * 60)

print("Input features:", len(X.columns))
print("Rows:", len(X))
print("Forbidden features found:",
      forbidden_features.intersection(X.columns))

print("\nMissing values:")
print(X.isna().sum().to_string())

print("\nMODEL FEATURE SAFETY CHECK COMPLETED")
