
from pathlib import Path
import pandas as pd

# Path to the dataset folder
DATA_DIR = Path(__file__).resolve().parent / "data"

# The seven required datasets
datasets = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}

print("=" * 60)
print("ECOMMERCE DATA INSPECTION")
print("=" * 60)

for name, filename in datasets.items():
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"[MISSING] {filename}")
        continue

    df = pd.read_csv(file_path, low_memory=False)

    print(f"\nDataset: {name}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("-" * 60)

print("\nDATA INSPECTION COMPLETED")

print("\n" + "=" * 60)
print("DETAILED DATA INSPECTION")
print("=" * 60)

for name, filename in datasets.items():
    file_path = DATA_DIR / filename

    if not file_path.exists():
        continue

    df = pd.read_csv(file_path, low_memory=False)

    print(f"\nDATASET: {name}")
    print("-" * 60)

    print("\nColumns and data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate full rows:")
    print(df.duplicated().sum())

    print("=" * 60)



print("\n" + "=" * 60)
print("PRIMARY KEY AND RELATIONSHIP CHECK")
print("=" * 60)

tables = {
    name: pd.read_csv(DATA_DIR / filename, low_memory=False)
    for name, filename in datasets.items()
}

# Check expected unique keys
primary_keys = {
    "orders": "order_id",
    "customers": "customer_id",
    "products": "product_id",
    "sellers": "seller_id",
    "categories": "product_category_name",
}

for table_name, key in primary_keys.items():
    df = tables[table_name]

    print(f"\nTable: {table_name}")
    print(f"Key: {key}")
    print(f"Duplicate keys: {df[key].duplicated().sum()}")
    print(f"Missing keys: {df[key].isna().sum()}")

# Check relationships between tables
relationships = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    (
        "products",
        "product_category_name",
        "categories",
        "product_category_name",
    ),
]

print("\nRELATIONSHIP VALIDATION")

for child, child_key, parent, parent_key in relationships:
    child_values = tables[child][child_key].dropna()
    parent_values = tables[parent][parent_key].dropna()

    unmatched = (~child_values.isin(parent_values)).sum()

    print(
        f"{child}.{child_key} -> {parent}.{parent_key}: "
        f"{unmatched} unmatched records"
    )

print("\nRELATIONSHIP CHECK COMPLETED")

print("\n" + "=" * 60)
print("UNMATCHED PRODUCT CATEGORIES")
print("=" * 60)

products_df = tables["products"]
categories_df = tables["categories"]

unmatched_categories = products_df[
    products_df["product_category_name"].notna()
    & ~products_df["product_category_name"].isin(
        categories_df["product_category_name"]
    )
]

print("\nNumber of unmatched product records:")
print(len(unmatched_categories))

print("\nUnmatched category names and counts:")
print(
    unmatched_categories["product_category_name"]
    .value_counts()
    .to_string()
)

print("\nProducts with missing category names:")
print(products_df["product_category_name"].isna().sum())

print("\nCATEGORY CHECK COMPLETED")

print("\n" + "=" * 60)
print("ORDER STATUS AND DATE VALIDATION")
print("=" * 60)

orders_df = tables["orders"].copy()

# Examine order statuses
print("\nORDER STATUS DISTRIBUTION:")
print(orders_df["order_status"].value_counts().to_string())

# Convert date columns without changing original files
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for column in date_columns:
    orders_df[column] = pd.to_datetime(
        orders_df[column],
        errors="coerce"
    )

# Missing dates grouped by order status
print("\nMISSING DATES BY ORDER STATUS:")

missing_by_status = (
    orders_df.groupby("order_status")[date_columns]
    .agg(lambda col: col.isna().sum())
)

print(missing_by_status.to_string())

# Check chronological consistency
approved_before_purchase = (
    orders_df["order_approved_at"]
    < orders_df["order_purchase_timestamp"]
).sum()

delivered_before_purchase = (
    orders_df["order_delivered_customer_date"]
    < orders_df["order_purchase_timestamp"]
).sum()

delivered_before_carrier = (
    orders_df["order_delivered_customer_date"]
    < orders_df["order_delivered_carrier_date"]
).sum()

print("\nDATE CONSISTENCY CHECKS:")
print("Approved before purchase:", approved_before_purchase)
print("Delivered before purchase:", delivered_before_purchase)
print("Delivered before carrier:", delivered_before_carrier)

print("\nORDER DATE VALIDATION COMPLETED")


print("\n" + "=" * 60)
print("DELIVERY DELAY TARGET ANALYSIS")
print("=" * 60)

# Use a copy without modifying the original dataset
target_df = orders_df.copy()

# Define which orders have valid target information
eligible_mask = (
    target_df["order_status"].eq("delivered")
    & target_df["order_approved_at"].notna()
    & target_df["order_delivered_customer_date"].notna()
    & target_df["order_estimated_delivery_date"].notna()
    & (
        target_df["order_delivered_customer_date"]
        >= target_df["order_purchase_timestamp"]
    )
)

eligible_orders = target_df.loc[eligible_mask].copy()

# Compare calendar dates rather than exact timestamps
actual_date = eligible_orders[
    "order_delivered_customer_date"
].dt.normalize()

estimated_date = eligible_orders[
    "order_estimated_delivery_date"
].dt.normalize()

eligible_orders["is_late"] = (
    actual_date > estimated_date
).astype(int)

print("\nTOTAL ORDERS:", len(target_df))
print("ELIGIBLE ORDERS:", len(eligible_orders))
print("EXCLUDED ORDERS:", (~eligible_mask).sum())

print("\nTARGET DISTRIBUTION:")
print(eligible_orders["is_late"].value_counts().sort_index())

print("\nTARGET PERCENTAGES:")
print(
    eligible_orders["is_late"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nTARGET ANALYSIS COMPLETED")

print("\n" + "=" * 60)
print("TARGET EXCLUSION AUDIT")
print("=" * 60)

audit = target_df.copy()

# Identify exclusion reasons
audit["not_delivered"] = (
    audit["order_status"] != "delivered"
)

audit["missing_approval"] = (
    audit["order_approved_at"].isna()
)

audit["missing_actual_delivery"] = (
    audit["order_delivered_customer_date"].isna()
)

audit["missing_estimated_delivery"] = (
    audit["order_estimated_delivery_date"].isna()
)

audit["invalid_delivery_order"] = (
    audit["order_delivered_customer_date"]
    < audit["order_purchase_timestamp"]
)

reason_columns = [
    "not_delivered",
    "missing_approval",
    "missing_actual_delivery",
    "missing_estimated_delivery",
    "invalid_delivery_order",
]

print("\nEXCLUSION REASONS (MAY OVERLAP):")

for column in reason_columns:
    print(f"{column}: {audit[column].sum()}")

# Count unique excluded orders
excluded_mask = audit[reason_columns].any(axis=1)

print("\nUNIQUE EXCLUDED ORDERS:", excluded_mask.sum())

print("\nELIGIBLE ORDERS:", (~excluded_mask).sum())

print("\nEXCLUSION AUDIT COMPLETED")

print("\n" + "=" * 60)
print("FINAL APPROVAL DATE VALIDATION")
print("=" * 60)

# Check date consistency among eligible orders
approval_after_delivery = (
    eligible_orders["order_approved_at"]
    > eligible_orders["order_delivered_customer_date"]
).sum()

estimated_before_approval = (
    eligible_orders["order_estimated_delivery_date"]
    < eligible_orders["order_approved_at"].dt.normalize()
).sum()

carrier_before_approval = (
    eligible_orders["order_delivered_carrier_date"]
    < eligible_orders["order_approved_at"]
).sum()

print("Approval after actual delivery:", approval_after_delivery)
print("Estimated delivery before approval:", estimated_before_approval)
print("Carrier handoff before approval:", carrier_before_approval)

print("\nFINAL DATE VALIDATION COMPLETED")

print("\n" + "=" * 60)
print("DATE ANOMALY OVERLAP ANALYSIS")
print("=" * 60)

anomaly_df = eligible_orders.copy()

anomaly_df["approval_after_delivery"] = (
    anomaly_df["order_approved_at"]
    > anomaly_df["order_delivered_customer_date"]
)

anomaly_df["estimated_before_approval"] = (
    anomaly_df["order_estimated_delivery_date"]
    < anomaly_df["order_approved_at"].dt.normalize()
)

anomaly_df["carrier_before_approval"] = (
    anomaly_df["order_delivered_carrier_date"]
    < anomaly_df["order_approved_at"]
)

anomaly_columns = [
    "approval_after_delivery",
    "estimated_before_approval",
    "carrier_before_approval",
]

print("\nANOMALY COUNTS:")
print(anomaly_df[anomaly_columns].sum().to_string())

print("\nNUMBER OF ANOMALIES PER ORDER:")
print(
    anomaly_df[anomaly_columns]
    .sum(axis=1)
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nORDERS WITH AT LEAST ONE ANOMALY:")
print(anomaly_df[anomaly_columns].any(axis=1).sum())

print("\nDATE ANOMALY ANALYSIS COMPLETED")
