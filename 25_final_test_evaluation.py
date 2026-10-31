
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FINAL HELD-OUT TEST EVALUATION")
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

# Read the choice saved before evaluating Test.
configuration = pd.read_csv(
    TABLES_DIR / "selected_model_configuration.csv"
)

assert len(configuration) == 1
assert configuration.loc[0, "selected_model"] == "Logistic Regression"
assert configuration.loc[0, "feature_set"] == "approval_time_6_features"

threshold = float(
    configuration.loc[0, "selected_threshold"]
)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")[FEATURES]
X_test = pd.read_csv(DATA_DIR / "X_test.csv")[FEATURES]

y_train = pd.read_csv(
    DATA_DIR / "y_train.csv"
)["is_late"]

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
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

# Fit preprocessing and model only on training data.
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "f1": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "average_precision": average_precision_score(
        y_test, y_prob
    )
}

print("\nSELECTED MODEL: Logistic Regression")
print("Feature count:", len(FEATURES))
print("Threshold:", threshold)
print("Test orders:", len(y_test))
print("Actual late orders:", int(y_test.sum()))

print("\nFINAL TEST METRICS:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nFINAL TEST CONFUSION MATRIX:")
print(cm)

# Save evaluation metrics
pd.DataFrame([{
    "model": "Logistic Regression",
    "threshold": threshold,
    "test_orders": len(y_test),
    **metrics
}]).to_csv(
    TABLES_DIR / "final_test_metrics.csv",
    index=False
)

# Save predictions for later error analysis
test_metadata = pd.read_csv(
    DATA_DIR / "test_data.csv",
    usecols=["order_id"]
)

assert len(test_metadata) == len(y_test)

predictions = pd.DataFrame({
    "order_id": test_metadata["order_id"],
    "actual_is_late": y_test.to_numpy(),
    "predicted_is_late": y_pred,
    "late_score": y_prob
})

predictions.to_csv(
    TABLES_DIR / "final_test_predictions.csv",
    index=False
)

# Save confusion matrix figure
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["On Time", "Late"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Final Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "09_final_test_confusion_matrix.png",
    dpi=300
)
plt.close()

print("\nFINAL TEST EVALUATION COMPLETED")
