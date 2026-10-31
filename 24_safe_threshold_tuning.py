
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("APPROVAL-TIME LOGISTIC THRESHOLD TUNING")
print("=" * 60)

NUMERIC = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "weekend_approval",
    "planned_delivery_days"
]

CATEGORICAL = ["customer_state"]

FEATURES = NUMERIC + CATEGORICAL

X_train = pd.read_csv(
    DATA_DIR / "X_train.csv"
)[FEATURES]

X_val = pd.read_csv(
    DATA_DIR / "X_validation.csv"
)[FEATURES]

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
)["is_late"]

y_val = pd.read_csv(
    DATA_DIR / "y_validation.csv"
)["is_late"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        NUMERIC
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        CATEGORICAL
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_val)[:, 1]

results = []

for threshold in np.arange(0.05, 1.00, 0.05):
    threshold = round(float(threshold), 2)

    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON:")
print(results_df.round(4).to_string(index=False))

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nBEST SAFE MODEL THRESHOLD:")
print("Threshold:", best_threshold)
print("Precision:", round(best_row["precision"], 4))
print("Recall:", round(best_row["recall"], 4))
print("F1 Score:", round(best_row["f1"], 4))

print(
    "Average Precision:",
    round(average_precision_score(y_val, y_prob), 4)
)

best_predictions = (
    y_prob >= best_threshold
).astype(int)

print("\nCONFUSION MATRIX AT BEST THRESHOLD:")
print(
    confusion_matrix(
        y_val,
        best_predictions,
        labels=[0, 1]
    )
)

results_df.to_csv(
    OUTPUT_DIR / "safe_threshold_comparison.csv",
    index=False
)

pd.DataFrame({
    "selected_model": ["Logistic Regression"],
    "feature_set": ["approval_time_6_features"],
    "selection_metric": ["average_precision"],
    "threshold_metric": ["f1"],
    "selected_threshold": [best_threshold]
}).to_csv(
    OUTPUT_DIR / "selected_model_configuration.csv",
    index=False
)

print("\nSAFE THRESHOLD TUNING COMPLETED")
