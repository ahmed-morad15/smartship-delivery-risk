
import pandas as pd

from src.data.load_data import load_all_tables


def main():
    tables = load_all_tables()

    orders = tables["orders"]
    items = tables["order_items"]
    payments = tables["payments"]
    geo = tables["geolocation"]
    reviews = tables["reviews"]

    print("\n" + "=" * 65)
    print("SMARTSHIP - ADVANCED DATA PROFILING")
    print("=" * 65)

    # 1. Order status distribution
    print("\n[1] ORDER STATUS DISTRIBUTION")

    status_counts = orders["order_status"].value_counts(dropna=False)

    print(status_counts)
    print("\nPercentages:")
    print((orders["order_status"].value_counts(
        normalize=True
    ) * 100).round(2))

    # 2. Delivery target analysis
    print("\n[2] DELIVERY TARGET ANALYSIS")

    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col], errors="coerce"
        )

    delivered = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    delivered["is_late"] = (
        delivered["order_delivered_customer_date"]
        > delivered["order_estimated_delivery_date"]
    ).astype(int)

    print(f"Total orders: {len(orders):,}")
    print(f"Delivered orders: {len(delivered):,}")
    print(f"Late orders: {delivered['is_late'].sum():,}")
    print(f"On-time orders: {(delivered['is_late'] == 0).sum():,}")

    print("\nTarget distribution (%):")
    print(
        (delivered["is_late"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # 3. Date range
    print("\n[3] DATE RANGE")

    for col in date_columns:
        print(
            f"{col}: "
            f"{orders[col].min()} -> {orders[col].max()}"
        )

    # 4. Order items
    print("\n[4] ORDER ITEMS")

    items_per_order = items.groupby("order_id").size()

    print(f"Total items: {len(items):,}")
    print(f"Unique orders: {items['order_id'].nunique():,}")
    print(f"Average items per order: {items_per_order.mean():.2f}")
    print(f"Maximum items per order: {items_per_order.max()}")

    # 5. Payments
    print("\n[5] PAYMENTS")

    payments_per_order = payments.groupby("order_id").size()

    print(f"Total payment records: {len(payments):,}")
    print(f"Unique orders: {payments['order_id'].nunique():,}")
    print(
        f"Average payment records per order: "
        f"{payments_per_order.mean():.2f}"
    )

    # 6. Reviews
    print("\n[6] REVIEWS")

    print(f"Review records: {len(reviews):,}")
    print(f"Unique reviewed orders: {reviews['order_id'].nunique():,}")
    print("\nReview score distribution:")
    print(reviews["review_score"].value_counts().sort_index())

    # 7. Geolocation
    print("\n[7] GEOLOCATION")

    print(f"Total records: {len(geo):,}")
    print(
        f"Unique ZIP prefixes: "
        f"{geo['geolocation_zip_code_prefix'].nunique():,}"
    )
    print(
        f"Exact duplicate rows: "
        f"{geo.duplicated().sum():,}"
    )

    # 8. Relationship checks
    print("\n[8] RELATIONSHIP CHECKS")

    order_ids = set(orders["order_id"])

    for name, df in {
        "items": items,
        "payments": payments,
        "reviews": reviews,
    }.items():
        unmatched = (~df["order_id"].isin(order_ids)).sum()
        print(f"{name} unmatched order IDs: {unmatched:,}")

    print("\n" + "=" * 65)
    print("PROFILING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
