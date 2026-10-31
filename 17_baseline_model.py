
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("BASELINE MODEL EVALUATION")
print("=" * 60)

validation_df = pd.read_csv(
    DATA_DIR / "validation_data.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test_data.csv"
)

def evaluate_baseline(name, df):
    y_true = df["is_late"].astype(int).to_numpy()

    # Baseline: predict every order will be on time
    y_pred = np.zeros(len(y_true), dtype=int)

    print(f"\n{name} BASELINE RESULTS")
    print("Orders:", len(y_true))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(
        "Precision:",
        round(precision_score(y_true, y_pred, zero_division=0), 4)
    )
    print(
        "Recall:",
        round(recall_score(y_true, y_pred, zero_division=0), 4)
    )
    print(
        "F1 Score:",
        round(f1_score(y_true, y_pred, zero_division=0), 4)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

evaluate_baseline("VALIDATION", validation_df)
evaluate_baseline("TEST", test_df)

print("\nBASELINE EVALUATION COMPLETED")
