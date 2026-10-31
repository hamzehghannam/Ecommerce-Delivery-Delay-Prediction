
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("APPROVAL-TIME FEATURE COMPARISON")
print("=" * 60)

FEATURES = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "weekend_approval",
    "planned_delivery_days",
    "customer_state"
]

NUMERIC = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "weekend_approval",
    "planned_delivery_days"
]

CATEGORICAL = ["customer_state"]

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

assert X_train.columns.tolist() == FEATURES
assert X_val.columns.tolist() == FEATURES

def make_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer([
        (
            "numeric",
            Pipeline(numeric_steps),
            NUMERIC
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            CATEGORICAL
        )
    ])

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor(True)),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor(False)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])
}

results = []

for name, model in models.items():
    print(f"\nTRAINING: {name}")

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_val, y_prob
        )
    }

    results.append(metrics)

    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print(
        "Average Precision:",
        round(metrics["average_precision"], 4)
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_val, y_pred, labels=[0, 1]
        )
    )

results_df = pd.DataFrame(results)

print("\nFINAL VALIDATION COMPARISON:")
print(results_df.round(4).to_string(index=False))

winner = results_df.loc[
    results_df["average_precision"].idxmax()
]

print("\nBEST MODEL BY AVERAGE PRECISION:")
print(winner["model"])
print(
    "Average Precision:",
    round(winner["average_precision"], 4)
)

OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    OUTPUT_DIR / "approval_time_model_comparison.csv",
    index=False
)

print("\nAPPROVAL-TIME COMPARISON COMPLETED")
