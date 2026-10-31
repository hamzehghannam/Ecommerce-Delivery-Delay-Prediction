
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    PROCESSED_DIR / "orders_all_features.csv"
)

print("=" * 60)
print("NUMPY NUMERICAL VALIDATION")
print("=" * 60)

# Operation 1: Logical mask
# Identify orders with multiple sellers
multi_seller_mask = np.greater(
    df["seller_count"].to_numpy(dtype=float),
    1
)

multi_seller_count = np.count_nonzero(multi_seller_mask)

print("\n1. NUMPY LOGICAL MASK")
print("Orders with multiple sellers:", multi_seller_count)

# Operation 2: Vectorized calculation
# Calculate freight ratio using NumPy
price = df["total_price"].to_numpy(dtype=float)
freight = df["total_freight"].to_numpy(dtype=float)

total = np.add(price, freight)

ratio = np.divide(
    freight,
    total,
    out=np.full(total.shape, np.nan),
    where=total > 0
)

print("\n2. NUMPY VECTORIZED CALCULATION")
print("Calculated freight ratios:", np.isfinite(ratio).sum())

# Operation 3: Numerical statistics
# Analyze planned delivery duration
planned_days = df["planned_delivery_days"].to_numpy(
    dtype=float
)

valid_planned_days = planned_days[
    np.isfinite(planned_days)
]

print("\n3. NUMPY NUMERICAL STATISTICS")
print("Average planned days:", round(np.mean(valid_planned_days), 2))
print("Median planned days:", round(np.median(valid_planned_days), 2))
print("Minimum planned days:", round(np.min(valid_planned_days), 2))
print("Maximum planned days:", round(np.max(valid_planned_days), 2))

print("\nNUMPY VALIDATION COMPLETED")

print("\n" + "=" * 60)
print("PLANNED DELIVERY DURATION AUDIT")
print("=" * 60)

# Identify invalid or unusual planned durations
negative_duration = np.isfinite(planned_days) & (
    planned_days < 0
)

zero_duration = np.isfinite(planned_days) & (
    planned_days == 0
)

long_duration = np.isfinite(planned_days) & (
    planned_days > 90
)

print("\nNegative planned durations:",
      np.count_nonzero(negative_duration))

print("Zero planned durations:",
      np.count_nonzero(zero_duration))

print("Planned durations over 90 days:",
      np.count_nonzero(long_duration))

print("\nORDERS WITH NEGATIVE DURATIONS:")

columns = [
    "order_id",
    "order_approved_at",
    "order_estimated_delivery_date",
    "planned_delivery_days",
    "order_status",
]

print(
    df.loc[negative_duration, columns]
    .sort_values("planned_delivery_days")
    .head(10)
    .to_string(index=False)
)

print("\nPLANNED DURATION AUDIT COMPLETED")

