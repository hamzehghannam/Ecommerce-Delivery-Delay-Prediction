
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LOGISTIC REGRESSION THRESHOLD TUNING")
print("=" * 60)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val = pd.read_csv(DATA_DIR / "X_validation.csv")

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
)["is_late"]

y_val = pd.read_csv(
    DATA_DIR / "y_validation.csv"
)["is_late"]

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    col for col in X_train.columns
    if col not in numeric_features
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
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
    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 2),
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

print("\nBEST VALIDATION THRESHOLD BY F1:")
print("Threshold:", best_row["threshold"])
print("Precision:", round(best_row["precision"], 4))
print("Recall:", round(best_row["recall"], 4))
print("F1 Score:", round(best_row["f1"], 4))

print(
    "\nAverage Precision:",
    round(average_precision_score(y_val, y_prob), 4)
)

results_df.to_csv(
    OUTPUT_DIR / "logistic_threshold_comparison.csv",
    index=False
)

print("\nTHRESHOLD TUNING COMPLETED")
