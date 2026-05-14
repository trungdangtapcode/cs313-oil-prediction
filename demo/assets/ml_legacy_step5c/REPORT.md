# Step5c Training Report

Dataset used:
- `/home/vund/.svn/data/processed/dataset_final_noleak_step5c_scaler.csv`

Output directory:
- `/home/vund/.svn/ml/classification/results_step5c`

Run setup:
- Train period: target date `2015-01-08 -> 2022-12-30`
- Test period: target date `2023-01-02 -> 2026-03-20`
- `SearchCV` stays inside train only
- Model selection is based on test metrics, per current workflow

Scaler status:
- `step5c` is treated as baked-scaled input
- `step1` and `step2` detect schema `step5c_baked_scaled_passthrough`
- model-time scaler is skipped for `step5c`
- `step3 -> step7` are tree-heavy steps and do not add an extra scaler layer

Parallelism:
- search jobs: `8`
- model jobs: `4`

## Step Results

### Step 1
- Selected by script: `LogisticRegression`
- Test: `Accuracy=0.5202`, `F1_macro=0.5166`, `AUC=0.5253`
- Best single-model Accuracy/AUC inside step1: `LightGBM` with `Accuracy=0.5405`, `AUC=0.5603`

### Step 2
- Selected by script: `Stacking`
- Test: `Accuracy=0.5179`, `F1_macro=0.5150`, `AUC=0.5379`
- Highest Accuracy inside step2: `LGBM_v2` / `Voting` at `0.5274`
- Highest AUC inside step2: `XGB_v2` at `0.5518`

### Step 3
- Selected by script: `1d_raw / XGB`
- Test: `Accuracy=0.5179`, `F1_macro=0.4637`, `AUC=0.5623`
- This is the highest AUC across the full run

### Step 4
- Selected case: `MI_SPEARMAN_TOP_30`
- Best model in step4: `GBM`
- Test: `Accuracy=0.5071`, `F1_macro=0.4513`, `AUC=0.5258`

### Step 5
- Selected feature set path: cluster + permutation
- Best model in step5: `XGB`
- Test: `Accuracy=0.5333`, `F1_macro=0.5029`, `AUC=0.5446`
- This is the best result from steps `4 -> 7` by Accuracy

### Step 6
- Selected by script: `GBM + step_50pct_2x`
- Test: `Accuracy=0.5226`, `F1_macro=0.4790`, `AUC=0.5550`

### Step 7
- Winner: `XGBoost`
- Test: `Accuracy=0.5202`, `F1_macro=0.4852`, `AUC=0.5585`

## Practical Summary

If selecting by the current test-first workflow:
- Best overall `F1_macro`: `Step 1 / LogisticRegression` at `0.5166`
- Best overall `Accuracy`: `Step 1 / LightGBM` at `0.5405`
- Best overall `AUC`: `Step 3 / 1d_raw XGB` at `0.5623`
- Best result among later feature-selection/tuning steps: `Step 5 / XGB` at `Accuracy=0.5333`, `F1_macro=0.5029`, `AUC=0.5446`

Artifacts:
- step logs: `results_step5c/logs/`
- selected bundles:
  - `step1_selected_bundle.joblib`
  - `step2_selected_bundle.joblib`
