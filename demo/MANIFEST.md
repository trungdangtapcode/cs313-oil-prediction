# Demo Asset Manifest

## Authoritative Current Artifacts

- `data/processed/dataset_final_noleak.csv`
- `data/processed/dataset_final_noleak_processed.csv`
- `data/processed/dataset_final_noleak_step5c_scaler.csv`
- `data/processed/dataset_final_noleak_step5c_preprocessor.joblib`
- `artifacts/improve_classification/REPORT.md`
- `artifacts/improve_classification/INTERPRETATION.md`
- `artifacts/improve_classification/results/primary_test_leaderboard.csv`
- `artifacts/improve_classification/results/ml_test_predictions.csv`
- `artifacts/improve_classification/results/dl_test_predictions.csv`
- `artifacts/improve_classification/results/ensemble_test_predictions.csv`
- `artifacts/improve_classification/results/feature_ranking.csv`
- `artifacts/improve_classification/results/threshold_diagnostics.csv`
- `artifacts/improve_classification/results/selective_coverage_diagnostics.csv`
- `artifacts/improve_classification/models/best_ml_full_coverage.joblib`
- `artifacts/improve_classification/models/best_ml_auc.joblib`

## Current EDA Assets

- `assets/eda_current/step5_upgraded_00_data_quality_overview.png`
- `assets/eda_current/step5_upgraded_03_target_over_time.png`
- `assets/eda_current/step5_upgraded_06_signal_scores.png`
- `assets/eda_current/step5_upgraded_07_ranking_and_shift.png`
- `assets/eda_current/step5_upgraded_13_class_shift.png`
- `assets/eda_current/tables/step5_upgraded_feature_ranking_clf_full.csv`
- `assets/eda_current/tables/step5_upgraded_leakage_risk_assessment.csv`
- `assets/eda_current/tables/step5_upgraded_distribution_shift_ks_test.csv`

## Legacy EDA / Presentation Assets

These are useful for storyboarding and comparison, but should not define the
forecasting contract.

- `assets/eda_legacy_presentation/01_oil_returns_over_time.png`
- `assets/eda_legacy_presentation/02_target_class_balance.png`
- `assets/eda_legacy_presentation/03_target_over_time.png`
- `assets/eda_legacy_presentation/04_data_quality.png`
- `assets/eda_legacy_presentation/05_feature_signals.png`
- `assets/eda_legacy_presentation/06_timeseries_properties.png`
- `assets/eda_legacy_presentation/07_split_leakage_check.png`

## Source Snapshots

- `code/pipeline_scripts/`: raw-to-processed pipeline and preprocessing helpers
- `code/eda/`: current and legacy EDA scripts
- `code/improve_classification/`: unified current ML experiment/report scripts

