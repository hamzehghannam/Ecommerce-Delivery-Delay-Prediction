
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@pytest.fixture(scope="module")
def datasets():
    return {
        "orders": pd.read_csv(
            PROCESSED_DIR / "orders_all_features.csv"
        ),
        "items": pd.read_csv(
            PROCESSED_DIR / "order_items_aggregated.csv"
        ),
        "payments": pd.read_csv(
            PROCESSED_DIR / "payments_aggregated.csv"
        ),
        "features": pd.read_csv(
            PROCESSED_DIR / "feature_dictionary.csv"
        ),
    }


def test_unique_order_ids(datasets):
    orders = datasets["orders"]

    assert orders["order_id"].is_unique
    assert orders["order_id"].notna().all()


def test_order_count_preserved(datasets):
    raw_orders = pd.read_csv(
        BASE_DIR / "data" / "olist_orders_dataset.csv"
    )

    assert len(datasets["orders"]) == len(raw_orders)

    assert set(datasets["orders"]["order_id"]) == set(
        raw_orders["order_id"]
    )


def test_item_totals_preserved(datasets):
    orders = datasets["orders"]
    items = datasets["items"]

    assert np.isclose(
        orders["total_price"].sum(),
        items["total_price"].sum()
    )

    assert np.isclose(
        orders["total_freight"].sum(),
        items["total_freight"].sum()
    )


def test_payment_totals_preserved(datasets):
    orders = datasets["orders"]
    payments = datasets["payments"]

    assert np.isclose(
        orders["total_payment"].sum(),
        payments["total_payment"].sum()
    )


def test_feature_dictionary_complete(datasets):
    dictionary = datasets["features"]
    orders = datasets["orders"]

    assert len(dictionary) >= 8
    assert dictionary["feature"].is_unique

    for feature in dictionary["feature"]:
        assert feature in orders.columns


def test_no_duplicate_aggregated_orders(datasets):
    assert datasets["items"]["order_id"].is_unique
    assert datasets["payments"]["order_id"].is_unique


def test_geographic_ratio_valid(datasets):
    ratio = datasets["orders"]["same_state_item_ratio"]

    assert ratio.dropna().between(0, 1).all()

def test_negative_planned_durations_handled(datasets):
    orders = datasets["orders"]

    # All negative durations must be flagged
    assert "invalid_planned_duration" in orders.columns

    invalid = orders["invalid_planned_duration"] == 1

    # Count based on our earlier data audit
    assert invalid.sum() == 12

    # Invalid durations must be missing
    assert orders.loc[
        invalid, "planned_delivery_days"
    ].isna().all()

    # Valid durations must not be negative
    assert (
        orders["planned_delivery_days"]
        .dropna()
        .ge(0)
        .all()
    )
