"""Train, compare and evaluate leakage-aware delivery risk models."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "results"
FIG_DIR = ROOT / "reports" / "figures"


def make_preprocessor(numeric, categorical):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], remainder="drop")


def metrics(y_true, pred, proba):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_late": float(precision_score(y_true, pred, zero_division=0)),
        "recall_late": float(recall_score(y_true, pred, zero_division=0)),
        "f1_late": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df = pd.read_csv(ROOT / "data" / "processed" / "smartship_modeling_dataset.csv", parse_dates=["order_purchase_timestamp"])
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    y = df.pop("is_late").astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.20, stratify=y, random_state=42
    )
    test_period_start = str(X_test["order_purchase_timestamp"].min())
    test_period_end = str(X_test["order_purchase_timestamp"].max())
    X_train = X_train.drop(columns=["order_id", "order_purchase_timestamp"])
    X_test = X_test.drop(columns=["order_id", "order_purchase_timestamp"])
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    pre = make_preprocessor(numeric, categorical)

    # Stratified CV is performed on a reproducible sample drawn ONLY from the training period
    # to keep runtime practical; final fitting still uses the complete training period.
    from sklearn.model_selection import train_test_split
    X_cv, _, y_cv, _ = train_test_split(
        X_train, y_train, train_size=min(12000, len(X_train)),
        stratify=y_train, random_state=42
    )
    # The CV sample is drawn exclusively from the training partition.

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=35, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=2, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=40, min_samples_leaf=3, class_weight="balanced", n_jobs=2, random_state=42),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_rows = []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
        scores = cross_validate(pipe, X_cv, y_cv, cv=cv, scoring="f1", n_jobs=1, return_train_score=False)
        cv_rows.append({"model": name, "cv_f1_mean": float(scores["test_score"].mean()), "cv_f1_std": float(scores["test_score"].std())})

    best_name = max(cv_rows, key=lambda x: x["cv_f1_mean"])["model"]
    if best_name == "Random Forest":
        estimator = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=2, random_state=42)
        param_dist = {"model__n_estimators": [40, 60], "model__max_depth": [None, 24], "model__min_samples_leaf": [2, 4]}
    elif best_name == "Extra Trees":
        estimator = ExtraTreesClassifier(class_weight="balanced", n_jobs=2, random_state=42)
        param_dist = {"model__n_estimators": [40, 60], "model__max_depth": [None, 24], "model__min_samples_leaf": [1, 3]}
    else:
        estimator = LogisticRegression(max_iter=1200, class_weight="balanced")
        param_dist = {"model__C": [0.1, 0.5, 1.0, 2.0]}

    tune_pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
    search = RandomizedSearchCV(tune_pipe, param_dist, n_iter=2, scoring="f1", cv=2, n_jobs=1, random_state=42, refit=True, verbose=1)
    search.fit(X_cv, y_cv)
    # Choose the classification threshold using out-of-fold probabilities from training data only.
    oof_proba = cross_val_predict(search.best_estimator_, X_cv, y_cv, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold_candidates = np.unique(np.quantile(oof_proba, np.linspace(0.05, 0.95, 91)))
    threshold_scores = [(float(t), float(f1_score(y_cv, (oof_proba >= t).astype(int), zero_division=0))) for t in threshold_candidates]
    threshold, train_oof_f1 = max(threshold_scores, key=lambda z: z[1])
    final_model = search.best_estimator_
    # Refit the selected configuration on the complete training period.
    final_model.fit(X_train, y_train)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = metrics(y_test, test_pred, test_proba)
    cm = confusion_matrix(y_test, test_pred).tolist()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_model, "feature_columns": X_train.columns.tolist(), "threshold": threshold,
                 "model_name": best_name, "best_params": search.best_params_, "numeric_features": numeric,
                 "categorical_features": categorical}, MODEL_DIR / "smartship_pipeline.joblib", compress=3)
    pd.DataFrame(cv_rows).to_csv(REPORT_DIR / "cross_validation_results.csv", index=False)
    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline = {"accuracy": float(accuracy_score(y_test, baseline_pred)),
                "precision_late": 0.0, "recall_late": 0.0, "f1_late": 0.0}
    summary = {"selected_model_family": best_name, "cv_sample_size": int(len(X_cv)),
               "best_cv_f1": float(search.best_score_), "oof_selected_threshold": float(threshold), "oof_f1_at_threshold": float(train_oof_f1), "best_params": search.best_params_,
               "majority_baseline_test_metrics": baseline,
               "test_metrics": test_metrics, "test_confusion_matrix": cm,
               "train_size": int(len(X_train)), "test_size": int(len(X_test)),
               "train_late_rate": float(y_train.mean()), "test_late_rate": float(y_test.mean()),
               "test_period_start": test_period_start,
               "test_period_end": test_period_end,
               "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0)}
    (REPORT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{"actual": int(a), "predicted": int(p), "late_probability": float(s)} for a,p,s in zip(y_test, test_pred, test_proba)]).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    # Tree-based feature importance supports a first-pass, non-causal business interpretation.
    fitted_estimator = final_model.named_steps["model"]
    if hasattr(fitted_estimator, "feature_importances_"):
        names = final_model.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({"feature": names, "importance": fitted_estimator.feature_importances_}).sort_values("importance", ascending=False)
        importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
        top = importance.head(20).sort_values("importance")
    else:
        importance = pd.DataFrame()
        top = importance

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
    fig, ax = plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test, test_pred, ax=ax, cmap="Blues", colorbar=False); ax.set_title("SmartShip - Test Confusion Matrix"); fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); RocCurveDisplay.from_predictions(y_test, test_proba, ax=ax); ax.set_title("ROC Curve - Stratified Test Set"); fig.tight_layout(); fig.savefig(FIG_DIR / "roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4)); PrecisionRecallDisplay.from_predictions(y_test, test_proba, ax=ax); ax.set_title("Precision-Recall Curve - Stratified Test Set"); fig.tight_layout(); fig.savefig(FIG_DIR / "precision_recall_curve.png", dpi=160); plt.close(fig)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(9,7)); ax.barh(top["feature"], top["importance"]); ax.set_title("Top 20 Model Feature Importances"); ax.set_xlabel("Importance"); fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=160); plt.close(fig)

    print("\nCross-validation (training-only stratified sample):")
    print(pd.DataFrame(cv_rows).to_string(index=False))
    print("\nSelected model family:", best_name)
    print("Best params:", search.best_params_)
    print(f"Training OOF-selected threshold: {threshold:.4f} (OOF F1={train_oof_f1:.4f})")
    print("\nUntouched stratified test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Majority baseline:", baseline)
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)
    print("Saved compressed model and evaluation reports.")

if __name__ == "__main__":
    main()
