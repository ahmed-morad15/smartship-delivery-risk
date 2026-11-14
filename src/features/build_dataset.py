"""Build an order-level, leakage-aware dataset for SmartShip."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def read(name):
    return pd.read_csv(RAW_DIR / name)


def build_dataset() -> pd.DataFrame:
    orders = read("olist_orders_dataset.csv")
    items = read("olist_order_items_dataset.csv")
    products = read("olist_products_dataset.csv")
    customers = read("olist_customers_dataset.csv")
    sellers = read("olist_sellers_dataset.csv")
    translation = read("product_category_name_translation.csv")
    geolocation = read("olist_geolocation_dataset.csv")

    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")

    # Label is derived from delivery outcome; outcome timestamps are never model features.
    labeled = orders.loc[
        orders.order_status.eq("delivered")
        & orders.order_purchase_timestamp.notna()
        & orders.order_delivered_customer_date.notna()
        & orders.order_estimated_delivery_date.notna(),
        ["order_id", "customer_id", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    ].copy()
    labeled["is_late"] = (
        labeled.order_delivered_customer_date > labeled.order_estimated_delivery_date
    ).astype("int8")
    labeled["estimated_lead_days"] = (
        labeled.order_estimated_delivery_date - labeled.order_purchase_timestamp
    ).dt.total_seconds() / 86400

    # Geolocation contains repeated observations per ZIP prefix; use robust median coordinates.
    geo = geolocation.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        geo_lat=("geolocation_lat", "median"), geo_lng=("geolocation_lng", "median")
    )
    customers = customers.merge(
        geo, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left", validate="many_to_one"
    ).drop(columns=["geolocation_zip_code_prefix"])
    sellers = sellers.merge(
        geo, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left", validate="many_to_one"
    ).drop(columns=["geolocation_zip_code_prefix"])

    # Enrich items with product/seller attributes before aggregating to one row/order.
    products = products.merge(translation, on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")
    item_detail = items.merge(
        products[["product_id", "product_category_name_english", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
        on="product_id", how="left", validate="many_to_one"
    ).merge(
        sellers[["seller_id", "seller_state", "geo_lat", "geo_lng"]].rename(
            columns={"geo_lat": "seller_lat", "geo_lng": "seller_lng"}
        ), on="seller_id", how="left", validate="many_to_one"
    )
    item_detail = item_detail.merge(
        labeled[["order_id", "order_purchase_timestamp"]], on="order_id", how="inner", validate="many_to_one"
    )
    item_detail["shipping_limit_days"] = (
        item_detail.shipping_limit_date - item_detail.order_purchase_timestamp
    ).dt.total_seconds() / 86400

    def mode_or_unknown(s):
        s = s.dropna().astype(str)
        return s.mode().iloc[0] if not s.empty else "unknown"

    item_agg = item_detail.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        seller_count=("seller_id", "nunique"),
        product_count=("product_id", "nunique"),
        avg_product_weight_g=("product_weight_g", "mean"),
        total_product_weight_g=("product_weight_g", "sum"),
        avg_shipping_limit_days=("shipping_limit_days", "mean"),
        min_shipping_limit_days=("shipping_limit_days", "min"),
        dominant_category=("product_category_name_english", mode_or_unknown),
        seller_state=("seller_state", mode_or_unknown),
        seller_lat=("seller_lat", "mean"),
        seller_lng=("seller_lng", "mean"),
    ).reset_index()
    # Derive volume without introducing post-purchase information.
    item_detail["product_volume_cm3"] = (
        item_detail.product_length_cm * item_detail.product_height_cm * item_detail.product_width_cm
    )
    volume = item_detail.groupby("order_id").product_volume_cm3.mean().rename("avg_product_volume_cm3")
    item_agg = item_agg.merge(volume, on="order_id", how="left")

    customer = customers[["customer_id", "customer_state", "geo_lat", "geo_lng"]].rename(
        columns={"geo_lat": "customer_lat", "geo_lng": "customer_lng"}
    )
    dataset = labeled.merge(customer, on="customer_id", how="left", validate="many_to_one")
    dataset = dataset.merge(item_agg, on="order_id", how="left", validate="one_to_one")

    # Haversine distance is a coarse route-length proxy, not an actual road distance.
    lat1 = np.radians(dataset["customer_lat"])
    lon1 = np.radians(dataset["customer_lng"])
    lat2 = np.radians(dataset["seller_lat"])
    lon2 = np.radians(dataset["seller_lng"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dataset["customer_seller_distance_km"] = 6371.0 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
    dataset["same_state"] = (dataset["customer_state"] == dataset["seller_state"]).astype("int8")
    dataset = dataset.drop(columns=["customer_lat", "customer_lng", "seller_lat", "seller_lng"])

    ts = dataset.order_purchase_timestamp
    dataset["purchase_hour"] = ts.dt.hour
    dataset["purchase_weekday"] = ts.dt.day_name()
    dataset["purchase_month"] = ts.dt.month.astype(str)
    dataset["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    # Drop all identifiers and post-outcome timestamps from the model table.
    dataset = dataset.drop(columns=[
        "customer_id", "order_delivered_customer_date", "order_estimated_delivery_date", "order_status"
    ], errors="ignore")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DIR / "smartship_modeling_dataset.csv", index=False)
    return dataset


if __name__ == "__main__":
    df = build_dataset()
    print(f"Saved dataset: {df.shape}")
    print("Target distribution:")
    print(df.is_late.value_counts().to_string())
    print("Missing values (top):")
    print(df.isna().sum().sort_values(ascending=False).head(12).to_string())
