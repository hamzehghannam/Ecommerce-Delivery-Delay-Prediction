
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FINAL MODEL ERROR ANALYSIS")
print("=" * 60)

test = pd.read_csv(
    DATA_DIR / "test_data.csv",
    usecols=[
        "order_id",
        "customer_state",
        "planned_delivery_days",
        "is_late"
    ]
)

predictions = pd.read_csv(
    TABLES_DIR / "final_test_predictions.csv"
)

# Merge by order ID, not by row position.
df = test.merge(
    predictions,
    on="order_id",
    how="inner",
    validate="one_to_one"
)

assert len(df) == len(test) == len(predictions)
assert (
    df["is_late"] == df["actual_is_late"]
).all()

conditions = [
    (df["is_late"] == 0) & (df["predicted_is_late"] == 0),
    (df["is_late"] == 0) & (df["predicted_is_late"] == 1),
    (df["is_late"] == 1) & (df["predicted_is_late"] == 0),
    (df["is_late"] == 1) & (df["predicted_is_late"] == 1)
]

df["error_type"] = np.select(
    conditions,
    ["TN", "FP", "FN", "TP"],
    default="UNKNOWN"
)

counts = df["error_type"].value_counts().reindex(
    ["TN", "FP", "FN", "TP"],
    fill_value=0
)

print("\nERROR COUNTS:")
print(counts.to_string())

assert counts.sum() == len(df)
assert "UNKNOWN" not in df["error_type"].values

# Error analysis by customer state
state_summary = (
    df.groupby("customer_state")
    .agg(
        total_orders=("order_id", "count"),
        actual_late=("is_late", "sum"),
        false_positives=(
            "error_type", lambda s: (s == "FP").sum()
        ),
        false_negatives=(
            "error_type", lambda s: (s == "FN").sum()
        )
    )
    .reset_index()
)

state_summary["false_positive_rate"] = (
    state_summary["false_positives"]
    / (state_summary["total_orders"] - state_summary["actual_late"])
    .replace(0, np.nan)
)

state_summary["missed_late_rate"] = (
    state_summary["false_negatives"]
    / state_summary["actual_late"].replace(0, np.nan)
)

print("\nTOP STATES BY MISSED LATE ORDERS:")
print(
    state_summary.sort_values(
        "false_negatives",
        ascending=False
    ).head(10).to_string(index=False)
)

# Planned delivery duration by error type
duration_summary = (
    df.groupby("error_type")["planned_delivery_days"]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nPLANNED DURATION BY ERROR TYPE:")
print(duration_summary.to_string())

# Save outputs
state_summary.to_csv(
    TABLES_DIR / "error_analysis_by_state.csv",
    index=False
)

duration_summary.to_csv(
    TABLES_DIR / "error_analysis_by_duration.csv"
)

df[
    [
        "order_id",
        "customer_state",
        "planned_delivery_days",
        "is_late",
        "predicted_is_late",
        "late_score",
        "error_type"
    ]
].to_csv(
    TABLES_DIR / "test_error_analysis.csv",
    index=False
)

print("\nERROR ANALYSIS COMPLETED")
