# SmartShip – Business Insights & Evaluation

## Evaluation design

- Target: `1` when actual customer delivery is after the estimated delivery timestamp; otherwise `0`.
- Modeling population: 96,470 delivered orders with complete purchase, actual delivery, and estimated delivery timestamps.
- Class prevalence: 8.11% late and 91.89% on-time.
- Split: stratified random 80/20 holdout (`random_state=42`).
- Training: three candidate algorithms compared with 3-fold stratified CV on a reproducible 12,000-row sample from the training partition. Hyperparameter search and threshold selection use training data only. The final model is refit on the full training partition.
- Model selected: Extra Trees.

## Final test-set results

Read exact, reproducible values from `evaluation_summary.json`. The current run produced:

| Metric | Result |
|---|---:|
| Accuracy | 63.80% |
| Precision (Late) | 14.02% |
| Recall (Late) | 67.48% |
| F1 (Late) | 23.22% |
| ROC-AUC | 0.7081 |
| PR-AUC | 0.2150 |

The majority-class baseline has 91.89% accuracy but zero recall and zero F1 for late deliveries. This illustrates why accuracy alone is misleading for this imbalanced target. SmartShip trades overall accuracy for substantially better detection of late orders.

At the selected training-only threshold (~0.399), the test confusion matrix is:

| | Predicted on-time | Predicted late |
|---|---:|---:|
| Actual on-time | 11,253 | 6,476 |
| Actual late | 509 | 1,056 |

The model identifies about two-thirds of late deliveries, but flags many on-time orders as at risk. It is therefore more suitable as a prioritization/triage aid than an automatic intervention or customer-facing promise.

## Model feature importance

See `feature_importance.csv` and `reports/figures/feature_importance.png`. In the fitted Extra Trees model, purchase month and estimated delivery lead time rank highly, followed by the approximate customer–seller geographic distance and selected customer-state indicators. These are model associations, not causal explanations. Importance may reflect dataset-specific patterns and should be revalidated on newer data.

## Operational recommendations

1. Use risk scores to prioritize manual review or proactive shipment checks, not to automatically penalize sellers.
2. Track precision and recall together; a lower threshold captures more late orders but creates more false alarms.
3. Monitor performance by month, state, and seller cohort to detect changing logistics patterns.
4. Retrain and validate on recent orders before real-world use; historical Olist data is from 2016–2018.

## Limitations

- Only delivered orders with observed outcomes can be labeled; cancelled, unavailable, and still-in-transit orders are excluded.
- The dataset is historical and does not contain live carrier capacity, route-level road conditions, or real-time seller operations.
- The primary stratified test split measures within-dataset discrimination, not future-period generalization. A separate chronological experiment during development had ROC-AUC close to 0.50 and PR-AUC close to 0.055, indicating substantial temporal distribution shift.
- Haversine distance from ZIP-prefix median coordinates is a coarse straight-line proxy, not actual road distance.
- Tree feature importance is not causal evidence, and probabilities are not guaranteed to be calibrated.
