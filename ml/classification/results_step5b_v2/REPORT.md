# Classification Report

## Scope

- Dataset: `/home/vund/.svn/data/processed/dataset_final_noleak_processed.csv`
- Price source for technical features: `/home/vund/.svn/data/processed/dataset_step4_noleak.csv`
- Workflow: `train + inner TimeSeriesSplit CV + test`
- Train target window: `2015-01-08 -> 2022-12-30`
- Test target window: `2023-01-02 -> 2026-03-20`
- Train rows: `2082`
- Test rows: `840`

Important note:
- There is no untouched final holdout in this run.
- The `test` set is being used to compare/select models, per the current requested workflow.
- Because of that, the reported `test` metrics are useful for model selection, but they are not a fully unbiased final estimate.

## EDA Summary

EDA was rerun with the upgraded script at:

- `/home/vund/.svn/eda_classification/eda_clf.py`
- Output folder: `/home/vund/.svn/eda_classification/step5_upgraded`

Key findings from the upgraded EDA:

- Class balance is stable:
  - Train: `UP=51.3%`, `DOWN=48.7%`
  - Test: `UP=49.8%`, `DOWN=50.2%`
- Signal is present but weak:
  - KS significant features: `10/27`
  - Mann-Whitney significant features: `11/27`
  - Point-biserial significant features: `7/27`
  - `0/27` features with `|Cohen's d| > 0.2`
- Highest-ranked research features:
  1. `day_of_week_sin`
  2. `inventory_change_pct`
  3. `conflict_event_count_log1p`
  4. `fatalities_log1p`
  5. `gdelt_volume_lag1_log1p`
- Strongest train/test shift:
  - `yield_spread`
  - `gdelt_tone_30d`
  - `gdelt_tone_7d`
  - `conflict_intensity_7d_log1p`
- Collinearity is non-trivial:
  - `VIF > 10`: `3` features
  - `VIF 5-10`: `6` features
  - Most problematic:
    - `gdelt_tone_7d`
    - `conflict_intensity_7d_log1p`
    - `conflict_event_count_log1p`

Relevant EDA artifacts:

- Ranking: `step5_upgraded_feature_ranking_clf.csv`
- Full ranking table: `tables/step5_upgraded_feature_ranking_clf_full.csv`
- Shift table: `tables/step5_upgraded_distribution_shift_ks_test.csv`
- VIF table: `tables/step5_upgraded_vif_results.csv`
- Summary: `reports/step5_upgraded_eda_summary.md`

## Step-by-Step Results

### Step 1: Baselines

Best baseline:

- Model: `LogisticRegression`
- Test Accuracy: `0.5119`
- Test F1_macro: `0.5085`
- Test AUC: `0.5249`

Artifact:

- `step1_test_results.csv`
- `step1_selected_result.csv`

### Step 2: Fine-tune + Ensemble

Best ensemble:

- Model: `Stacking`
- Test Accuracy: `0.5333`
- Test F1_macro: `0.5306`
- Test AUC: `0.5411`

This is the strongest result before explicit feature-selection logic starts helping.

Artifact:

- `step2_test_results.csv`
- `step2_selected_result.csv`

### Step 3: Technical Features + Target Variants

Best target/model combination:

- Target: `1d_t03`
- Model: `LGBM`
- Test Accuracy: `0.5218`
- Test F1_macro: `0.5169`
- Test AUC: `0.5360`

Observations:

- Technical features increased the usable feature set to `56`.
- Thresholded targets did not clearly beat the plain `1d_raw` setup.
- This step improved the feature space more than it improved headline metrics.

Artifact:

- `step3_results.csv`
- `step3_test_results.csv`

### Step 4: Ranking-Based Feature Selection

Best train-CV subset family:

- `MI_TOP_20`

Step 4 test results:

| Model | Test Accuracy | Test F1_macro | Test AUC |
|---|---:|---:|---:|
| XGB | 0.5190 | 0.4681 | 0.5312 |
| GBM | 0.5131 | 0.4762 | 0.5317 |
| LGBM | 0.5083 | 0.4370 | 0.5479 |

Observations:

- Simple ranking-based top-N selection was not enough.
- The selected `MI_TOP_20` subset did not outperform the best ensemble from Step 2.

Artifacts:

- `step4_feature_ranking.csv`
- `step4_subset_comparison.csv`
- `step4_best_by_ranking.csv`
- `step4_results.csv`

### Step 5: Smart Selection (Correlation Clustering + Permutation Importance)

This is the best step overall in the current run.

Best model:

- Model: `LGBM`
- Feature set: `CLUSTER_POS_10`
- Test Accuracy: `0.5452`
- Test F1_macro: `0.5326`
- Test AUC: `0.5563`

Comparison within Step 5:

| Model | Test Accuracy | Test F1_macro | Test AUC |
|---|---:|---:|---:|
| LGBM | 0.5452 | 0.5326 | 0.5563 |
| XGB | 0.5345 | 0.5290 | 0.5498 |
| GBM | 0.5262 | 0.5153 | 0.5363 |

Selected `CLUSTER_POS_10` feature set:

1. `vix_return_slog1p`
2. `oil_return`
3. `sp500_return_lag1`
4. `ret_mean_5`
5. `momentum_10`
6. `macd_signal`
7. `gdelt_tone_lag1`
8. `gdelt_volume_lag1_log1p`
9. `yield_spread`
10. `net_imports_change_pct_slog1p`

Interpretation:

- Correlation clustering and permutation importance worked better than simple MI/Spearman ranking.
- The winning set mixes:
  - short-horizon technical momentum
  - market context
  - one macro spread (`yield_spread`)
  - one GDELT attention feature
  - one supply-side feature

Artifacts:

- `step5_selected_features.csv`
- `step5_perm_importance.csv`
- `step5_set_comparison.csv`
- `step5_results.csv`

### Step 6: Weight Decay / Recency Weighting

Best script-selected result:

- Model: `LGBM`
- Scheme: `uniform`
- Test Accuracy: `0.5238`
- Test F1_macro: `0.4742`
- Test AUC: `0.5548`

Observations:

- Recency weighting did not improve over Step 5.
- `step_50pct_2x` helped inner selection for `GBM`, but the final test result still lagged.
- This suggests the main bottleneck is feature stability / regime shift, not just stale samples.

Artifacts:

- `step6_selection_results.csv`
- `step6_results.csv`
- `step6_weight_schemes.png`
- `step6_comparison.png`

### Step 7: Extensive Tree Tuning

Saved Step 7 results come from a rerun with reduced budget:

- `STEP7_N_ITER=15`

Reason:

- The original full-budget `50`-iteration run completed `XGBoost`, but the `GBM` leg became too slow and effectively blocked the pipeline.
- The lighter rerun was used to produce final Step 7 artifacts.

Saved Step 7 winner:

- Model: `XGBoost`
- Test Accuracy: `0.5226`
- Test F1_macro: `0.4716`
- Test AUC: `0.5586`

Important nuance:

- Step 7 gives the best `AUC` among the saved final artifacts.
- But it does **not** beat Step 5 on `Accuracy` or `F1_macro`.

Partial full-budget note:

- In the interrupted heavier run, `XGBoost` had already reached:
  - Test Accuracy: `0.5202`
  - Test F1_macro: `0.4852`
  - Test AUC: `0.5585`
- So the main Step 7 conclusion is stable: better ranking/AUC behavior than Step 5, but weaker classification balance on `F1_macro`.

Artifacts:

- `step7_test_results.csv`
- `step7_results.csv`

## Overall Ranking

If the priority is `F1_macro` and balanced classification quality:

1. `Step 5 / LGBM / CLUSTER_POS_10`
   - Accuracy: `0.5452`
   - F1_macro: `0.5326`
   - AUC: `0.5563`
2. `Step 2 / Stacking`
   - Accuracy: `0.5333`
   - F1_macro: `0.5306`
   - AUC: `0.5411`
3. `Step 5 / XGB / CLUSTER_POS_10`
   - Accuracy: `0.5345`
   - F1_macro: `0.5290`
   - AUC: `0.5498`

If the priority is mainly `AUC`:

1. `Step 7 / XGBoost`
   - AUC: `0.5586`
2. `Step 5 / LGBM`
   - AUC: `0.5563`
3. `Step 6 / LGBM uniform`
   - AUC: `0.5548`

## Recommendation

Current recommended model:

- `Step 5 / LGBM / CLUSTER_POS_10`

Why:

- Best overall trade-off between `Accuracy`, `F1_macro`, and `AUC`
- Small feature set (`10` features)
- More interpretable than the broader technical-expanded sets
- Better generalization than weighting-heavy or brute-force tuning variants

## Caveats

- The current workflow uses `test` for model comparison and model choice.
- That means this is a practical selection run, not a strict final unbiased evaluation.
- `oil_return` is still in the selected Step 5 feature set and is marked `medium` EDA risk:
  - acceptable for `end-of-day T -> T+1`
  - not acceptable for stricter intraday/no-same-day forecasting
- Several GDELT/conflict features show both:
  - genuine signal
  - strong train/test drift
- `yield_spread` is one of the strongest shifted features between train and test.

## Next Actions

1. Freeze `Step 5 / LGBM / CLUSTER_POS_10` as the current baseline winner.
2. If a truly final score is needed, carve out a new untouched final holdout after model selection.
3. For stricter forecasting, rerun Step 5 without same-day `oil_return`.
4. Consider pruning or regularizing the most collinear GDELT/conflict block:
   - `gdelt_tone_7d`
   - `conflict_intensity_7d_log1p`
   - `conflict_event_count_log1p`
