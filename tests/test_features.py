from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_modeling_dataset_is_one_row_per_order_and_has_binary_target():
    path = ROOT / "data" / "processed" / "smartship_modeling_dataset.csv"
    if not path.exists():
        import pytest
        pytest.skip("Run python -m src.features.build_dataset first")
    df = pd.read_csv(path)
    assert df.order_id.is_unique
    assert set(df.is_late.unique()).issubset({0, 1})
    assert "order_delivered_customer_date" not in df.columns
    assert "order_estimated_delivery_date" not in df.columns
    assert len(df) > 90_000
