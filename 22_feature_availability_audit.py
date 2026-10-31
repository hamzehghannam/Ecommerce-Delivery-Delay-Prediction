
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Features directly derived from approval date or
# available as order-level information at approval.
confirmed_features = [
    "approval_day_of_week",
    "approval_month",
    "approval_hour",
    "weekend_approval",
    "planned_delivery_days",
    "customer_state"
]

# Historical dataset does not prove exact availability
# of all item and payment information at approval time.
assumed_features = [
    col for col in X_train.columns
    if col not in confirmed_features
]

audit = pd.DataFrame([
    {
        "feature": col,
        "availability_status": (
            "Approval-time candidate"
            if col in confirmed_features
            else "Requires availability assumption"
        )
    }
    for col in X_train.columns
])

print("=" * 60)
print("FEATURE AVAILABILITY AUDIT")
print("=" * 60)

print("\nApproval-time candidate features:")
print(confirmed_features)

print("\nFeatures requiring availability assumptions:")
print(assumed_features)

print("\nTotal features:", len(X_train.columns))
print("Approval-time candidates:", len(confirmed_features))
print("Requiring assumptions:", len(assumed_features))

audit.to_csv(
    DATA_DIR / "feature_availability_audit.csv",
    index=False
)

print("\nFEATURE AVAILABILITY AUDIT COMPLETED")
