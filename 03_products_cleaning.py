
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

products = pd.read_csv(
    DATA_DIR / "olist_products_dataset.csv"
)

categories = pd.read_csv(
    DATA_DIR / "product_category_name_translation.csv"
)

# Preserve original data
clean_products = products.copy()

# Add English category names without losing products
clean_products = clean_products.merge(
    categories,
    on="product_category_name",
    how="left",
    validate="many_to_one"
)

# Use the original category when translation is unavailable
clean_products["category_for_analysis"] = (
    clean_products["product_category_name_english"]
    .fillna(clean_products["product_category_name"])
    .fillna("Unknown")
)

print("=" * 60)
print("PRODUCT DATA CLEANING")
print("=" * 60)

print("Original products:", len(products))
print("Processed products:", len(clean_products))

print(
    "Unknown categories:",
    (clean_products["category_for_analysis"] == "Unknown").sum()
)

print(
    "Categories without translation:",
    (
        clean_products["product_category_name"].notna()
        & clean_products["product_category_name_english"].isna()
    ).sum()
)

print("\nMissing numeric values:")
numeric_columns = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

print(clean_products[numeric_columns].isna().sum())

# Save separately from raw data
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

output_path = PROCESSED_DIR / "products_cleaned.csv"

clean_products.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nPRODUCT FILE SAVED")
print("File:", output_path)
print("Saved rows:", len(clean_products))
