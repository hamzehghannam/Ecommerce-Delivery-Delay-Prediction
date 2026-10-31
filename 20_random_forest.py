
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

print("=" * 60)
print("RANDOM FOREST MODEL")
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

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
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

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")
print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
print(
    "Precision:",
    round(precision_score(y_val, y_pred, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(y_val, y_pred, zero_division=0), 4)
)
print(
    "F1 Score:",
    round(f1_score(y_val, y_pred, zero_division=0), 4)
)
print(
    "Average Precision:",
    round(average_precision_score(y_val, y_prob), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

print("\nRANDOM FOREST COMPLETED")
