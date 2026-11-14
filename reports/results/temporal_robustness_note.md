# Temporal robustness experiment

During development, a strict chronological split was tested by sorting orders by purchase timestamp and using the earliest 80% for training and latest 20% for testing. With the same order-time feature policy, the later-period holdout had a late-order prevalence of approximately 5.3%, compared with approximately 8.8% in the training period. The resulting ROC-AUC was approximately 0.50 and PR-AUC approximately 0.055, showing that the fitted model did not generalize reliably across this temporal shift.

The primary reported evaluation therefore uses a stratified random 80/20 holdout to measure within-dataset discrimination. This does **not** establish reliable performance on future orders. The temporal result is retained as an explicit limitation and motivates retraining with more recent logistics data, time-aware validation, and monitoring before any operational deployment.

These temporal metrics came from the development experiment before the final stratified evaluation; the final primary metrics are generated in `evaluation_summary.json`.
