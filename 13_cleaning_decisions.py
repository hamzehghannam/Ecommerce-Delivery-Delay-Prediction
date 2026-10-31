
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

decisions = [
    {
        "issue": "Missing order dates",
        "finding": "160 approval, 1783 carrier, 2965 actual delivery",
        "action": "Preserved missing values",
        "reason": "Missing delivery dates may represent unfinished orders",
    },
    {
        "issue": "Missing product categories",
        "finding": "610 products",
        "action": "Assigned Unknown for analysis",
        "reason": "Avoid inventing category information",
    },
    {
        "issue": "Untranslated categories",
        "finding": "13 products",
        "action": "Preserved original category names",
        "reason": "Avoid losing valid product categories",
    },
    {
        "issue": "Missing product weights",
        "finding": "16 orders without available weights",
        "action": "Preserved missing values",
        "reason": "Avoid inventing product weights",
    },
    {
        "issue": "Orders without items",
        "finding": "775 orders",
        "action": "Preserved through left joins",
        "reason": "Avoid losing original orders",
    },
    {
        "issue": "Orders without payments",
        "finding": "1 order",
        "action": "Preserved missing payment data",
        "reason": "Missing payment record does not imply zero payment",
    },
    {
        "issue": "Negative planned delivery duration",
        "finding": "12 orders",
        "action": "Flagged and converted derived duration to missing",
        "reason": "Negative planned duration is not valid for modeling",
    },
    {
        "issue": "Temporal inconsistencies",
        "finding": "1351 orders with at least one flagged anomaly",
        "action": "Recorded for review",
        "reason": "Not all timestamp anomalies justify removing orders",
    },
]

decisions.extend([
    {
        "issue": "Initial modeling exclusions",
        "finding": "2985 orders",
        "action": "Excluded from modeling dataset only",
        "reason": "Orders failed delivery status, date availability, or chronological eligibility rules",
    },
    {
        "issue": "Final temporal exclusions",
        "finding": "62 orders: 61 approval after delivery, 6 negative planned durations, 5 overlapping",
        "action": "Excluded from modeling dataset only",
        "reason": "Prediction at approval requires a consistent event timeline",
    },
    {
        "issue": "Final modeling dataset",
        "finding": "96394 orders: 89862 on time and 6532 late",
        "action": "Preserved in modeling_data.csv",
        "reason": "Eligible orders with known delivery outcomes and valid chronology",
    },
    {
        "issue": "Feature availability assumption",
        "finding": "Item, payment, and estimated delivery fields",
        "action": "Documented as available at approval under assignment scenario",
        "reason": "Historical data does not verify the exact availability time of every field",
    },
    {
        "issue": "Missing modeling feature values",
        "finding": "One eligible order lacks payment information",
        "action": "Preserved missing values for later preprocessing",
        "reason": "Imputation must be fitted only on training data to prevent leakage",
    },
])

report = pd.DataFrame(decisions)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "cleaning_decisions.csv"

report.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 60)
print("CLEANING DECISIONS REPORT")
print("=" * 60)
print("Documented decisions:", len(report))
print("Saved file:", output_path)
print("\nCLEANING DECISIONS COMPLETED")
