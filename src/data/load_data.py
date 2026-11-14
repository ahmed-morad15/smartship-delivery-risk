
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

TABLES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_table(table_name: str) -> pd.DataFrame:
    if table_name not in TABLES:
        raise ValueError(f"Unknown table: {table_name}")

    file_path = RAW_DATA_DIR / TABLES[table_name]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def load_all_tables() -> dict:
    return {
        name: load_table(name)
        for name in TABLES
    }
