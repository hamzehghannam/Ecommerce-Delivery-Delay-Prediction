
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

features = [
    {
        "feature": "approval_day_of_week",
        "calculation": "Day of week from approval date",
        "source": "orders.order_approved_at",
        "purpose": "Detect weekly delivery patterns",
        "availability": "At approval",
    },
    {
        "feature": "approval_month",
        "calculation": "Month from approval date",
        "source": "orders.order_approved_at",
        "purpose": "Capture seasonal patterns",
        "availability": "At approval",
    },
    {
        "feature": "approval_hour",
        "calculation": "Hour from approval timestamp",
        "source": "orders.order_approved_at",
        "purpose": "Analyze approval timing",
        "availability": "At approval",
    },
    {
        "feature": "planned_delivery_days",
        "calculation": "Estimated delivery - approval time",
        "source": "orders",
        "purpose": "Measure planned delivery duration",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "order_total_value",
        "calculation": "Total price + total freight",
        "source": "order_items aggregated",
        "purpose": "Measure order monetary size",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "average_item_price",
        "calculation": "Total price / item count",
        "source": "order_items aggregated",
        "purpose": "Measure average item price",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "freight_ratio",
        "calculation": "Total freight / order total value",
        "source": "order_items aggregated",
        "purpose": "Measure shipping cost proportion",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "multiple_sellers",
        "calculation": "1 if seller count > 1 else 0",
        "source": "order_items aggregated",
        "purpose": "Identify multi-seller orders",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "multiple_payments",
        "calculation": "1 if payment count > 1 else 0",
        "source": "payments aggregated",
        "purpose": "Identify multiple payment records",
        "availability": "At approval (assumed)",
    },
    
    {
        "feature": "weekend_approval",
        "calculation": "1 if approval is Saturday or Sunday",
        "source": "orders.order_approved_at",
        "purpose": "Analyze weekend effects",
        "availability": "At approval",
    },
    {
        "feature": "total_product_weight_g",
        "calculation": "Sum of item weights in grams",
        "source": "order_items + products",
        "purpose": "Measure total order weight",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "average_product_weight_g",
        "calculation": "Average item weight in grams",
        "source": "order_items + products",
        "purpose": "Study relationship between weight and delay",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "same_state_item_ratio",
        "calculation": "Share of items whose seller is in the customer state",
        "source": "order_items + sellers + customers",
        "purpose": "Measure geographic proximity",
        "availability": "At approval (assumed)",
    },
    {
        "feature": "distinct_seller_states",
        "calculation": "Number of distinct seller states per order",
        "source": "order_items + sellers",
        "purpose": "Measure geographic distribution of sellers",
        "availability": "At approval (assumed)",
    },
]


dictionary = pd.DataFrame(features)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "feature_dictionary.csv"

dictionary.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 60)
print("FEATURE DICTIONARY")
print("=" * 60)

print("Documented features:", len(dictionary))
print("Columns:", list(dictionary.columns))
print("Saved file:", output_path)

print("\nFEATURE DICTIONARY COMPLETED")
