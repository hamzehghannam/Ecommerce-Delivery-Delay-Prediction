
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def test_final_modeling_dataset():
    df = pd.read_csv(
        PROCESSED_DIR / "modeling_data.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    # One row per order
    assert df["order_id"].is_unique
    assert df["order_id"].notna().all()

    # Validate the agreed eligibility rules
    assert df["order_status"].eq("delivered").all()
    assert df["order_approved_at"].notna().all()
    assert df["order_delivered_customer_date"].notna().all()
    assert df["order_estimated_delivery_date"].notna().all()

    assert (
        df["order_approved_at"]
        <= df["order_delivered_customer_date"]
    ).all()

    assert df["invalid_planned_duration"].eq(0).all()
    assert df["planned_delivery_days"].ge(0).all()

    # Recalculate the actual target
    expected_target = (
        df["order_delivered_customer_date"].dt.normalize()
        > df["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)

    assert df["is_late"].equals(expected_target)

    # Confirm recorded dataset size
    assert len(df) == 96394
    assert df["is_late"].sum() == 6532
