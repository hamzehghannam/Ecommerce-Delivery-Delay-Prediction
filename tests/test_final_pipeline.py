
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

FEATURES = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "weekend_approval",
    "planned_delivery_days",
    "customer_state"
]


def load_split(name):
    return pd.read_csv(
        DATA_DIR / f"{name}_data.csv",
        parse_dates=["order_approved_at"]
    )


def test_split_sizes_and_unique_orders():
    train = load_split("train")
    val = load_split("validation")
    test = load_split("test")

    assert len(train) + len(val) + len(test) == 96394

    all_ids = pd.concat([
        train["order_id"],
        val["order_id"],
        test["order_id"]
    ])

    assert all_ids.is_unique


def test_strict_temporal_order():
    train = load_split("train")
    val = load_split("validation")
    test = load_split("test")

    assert (
        train["order_approved_at"].max()
        < val["order_approved_at"].min()
    )

    assert (
        val["order_approved_at"].max()
        < test["order_approved_at"].min()
    )


def test_target_is_binary_and_complete():
    for name in ["train", "validation", "test"]:
        df = load_split(name)

        assert df["is_late"].notna().all()
        assert df["is_late"].isin([0, 1]).all()


def test_no_future_columns_in_model_features():
    forbidden = {
        "order_status",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "is_late",
        "order_id"
    }

    assert not (set(FEATURES) & forbidden)

    for name in ["train", "validation", "test"]:
        X = pd.read_csv(DATA_DIR / f"X_{name}.csv")

        assert all(
            feature in X.columns
            for feature in FEATURES
        )


def test_final_prediction_integrity():
    test = load_split("test")

    predictions = pd.read_csv(
        TABLES_DIR / "final_test_predictions.csv"
    )

    assert len(predictions) == len(test)
    assert predictions["order_id"].is_unique

    merged = test[
        ["order_id", "is_late"]
    ].merge(
        predictions,
        on="order_id",
        validate="one_to_one"
    )

    assert len(merged) == len(test)

    assert (
        merged["is_late"]
        == merged["actual_is_late"]
    ).all()

    assert merged["predicted_is_late"].isin([0, 1]).all()
    assert merged["late_score"].between(0, 1).all()


def test_final_confusion_matrix_counts():
    predictions = pd.read_csv(
        TABLES_DIR / "final_test_predictions.csv"
    )

    y_true = predictions["actual_is_late"]
    y_pred = predictions["predicted_is_late"]

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    assert (tn, fp, fn, tp) == (11479, 2363, 469, 149)
