
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(
    DATA_DIR / "modeling_data.csv",
    parse_dates=["order_approved_at"]
)

print("=" * 60)
print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

# Verify required information
assert df["order_id"].is_unique
assert df["order_approved_at"].notna().all()
assert df["is_late"].isin([0, 1]).all()

# Sort chronologically, with a stable tie-breaker
df = df.sort_values(
    ["order_approved_at", "order_id"]
).reset_index(drop=True)

# Approximate 70% / 15% / 15% chronological split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

# Prevent the same approval timestamp appearing on
# both sides of a split boundary
assert (
    train_df["order_approved_at"].max()
    <= val_df["order_approved_at"].min()
)

assert (
    val_df["order_approved_at"].max()
    <= test_df["order_approved_at"].min()
)

for name, subset in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(f"\n{name}")
    print("Orders:", len(subset))
    print("Start:", subset["order_approved_at"].min())
    print("End:", subset["order_approved_at"].max())
    print("Late orders:", int(subset["is_late"].sum()))
    print("Late rate (%):", round(subset["is_late"].mean() * 100, 2))

train_df.to_csv(DATA_DIR / "train_data.csv", index=False)
val_df.to_csv(DATA_DIR / "validation_data.csv", index=False)
test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

print("\nTEMPORAL SPLIT COMPLETED")
