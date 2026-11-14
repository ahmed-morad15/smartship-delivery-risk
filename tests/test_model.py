from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_saved_pipeline_predicts_probability():
    model_path = ROOT / "models" / "smartship_pipeline.joblib"
    data_path = ROOT / "data" / "processed" / "smartship_modeling_dataset.csv"
    if not model_path.exists() or not data_path.exists():
        import pytest
        pytest.skip("Build the dataset and train the model first")
    artifact = joblib.load(model_path)
    data = pd.read_csv(data_path, nrows=2)
    X = data.drop(columns=["is_late", "order_id", "order_purchase_timestamp"])
    X = X[artifact["feature_columns"]]
    probabilities = artifact["pipeline"].predict_proba(X)[:, 1]
    assert len(probabilities) == 2
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
