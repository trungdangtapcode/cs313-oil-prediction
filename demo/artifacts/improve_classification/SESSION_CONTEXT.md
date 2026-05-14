# Session Context: Improve Oil Direction Classification

Date: 2026-05-14  
Workspace: `/home/vund/.svn`  
Python env used: `.venv/bin/python` (`Python 3.8.20`)

## Original User Request

The user asked to:

1. Read `ml/classification/final/` to understand the current classification context.
2. Unzip `oil_direction_team_docs_md.zip` into `docs/improve`.
3. Improve results from `ml/classification/final/` into `ml/improve-classification/`.
4. Apply both machine learning and deep learning if needed.
5. Keep scripts, logs, results, and artifacts inside `ml/improve-classification/`.

## Important Environment Notes

- The default system Python under `/opt/miniconda3/bin/python` had dependency issues:
  - `pandas` attempted to import `pyarrow`, which failed due to NumPy 2.x compatibility.
  - `sklearn`, `xgboost`, `lightgbm`, `torch` were missing from that default environment.
- The repo `.venv` worked and was used for all successful ML/DL runs:
  - `pandas`: OK
  - `sklearn`: OK
  - `xgboost`: OK
  - `lightgbm`: OK
  - `torch`: OK
  - `catboost`: missing
  - `tensorflow`: missing

Use:

```bash
.venv/bin/python -u ml/improve-classification/run_improve_classification.py
```

## Docs Unzipped

`oil_direction_team_docs_md.zip` was unzipped into:

```text
docs/improve/
  oil_direction_all_in_one.md
  oil_direction_implementation_plan.md
  oil_direction_leakage_audit_checklist.md
  oil_direction_research_benchmark_brief.md
```

Key guidance from these docs:

- Daily next-day oil direction is noisy.
- Realistic near-term target is around `Accuracy >= 0.56`, `AUC >= 0.58`.
- Any result above `0.60-0.63` accuracy should be leakage-audited carefully.
- Use chronological train/validation/test only.
- Do not treat threshold-label or coverage-filtered accuracy as directly comparable to full-coverage raw UP/DOWN accuracy.

## Baseline Context From `ml/classification/final/`

Current final pipeline uses:

```text
data/processed/dataset_final_noleak_step5c_scaler.csv
```

Main target:

```text
oil_return_fwd1 > 0 => UP=1, DOWN=0
```

Split logic:

```text
target date < 2023-01-01       => train
target date >= 2023-01-01      => final test
```

`ml/config.py` also defines validation split:

```text
target date < 2022-01-01                         => train
2022-01-01 <= target date < 2023-01-01           => validation
target date >= 2023-01-01                        => test
```

Important saved final baselines:

```text
final_step6_selected:
  Model      = XGB
  Scheme     = linear_03
  Accuracy   = 0.5262
  F1_macro   = 0.4906
  AUC        = 0.5596

final_step6_all_schemes_best_f1:
  Model      = LGBM
  Scheme     = exp_hl100
  Accuracy   = 0.5345
  F1_macro   = 0.5341
  AUC        = 0.5234

final_step6_all_schemes_best_auc:
  Model      = XGB
  Scheme     = step_50pct_2x
  Accuracy   = 0.5345
  F1_macro   = 0.4889
  AUC        = 0.5608
```

## Work Completed In `ml/improve-classification/`

Created:

```text
ml/improve-classification/
  run_improve_classification.py
  REPORT.md
  SESSION_CONTEXT.md
  logs/
    run_20260514_152953.log
  models/
    best_ml_full_coverage.joblib
    best_ml_auc.joblib
    DL_GRU_L5.pt
    DL_GRU_L10.pt
    DL_GRU_L20.pt
    DL_GRU_L40.pt
    DL_MLP_L5.pt
    DL_MLP_L10.pt
    DL_MLP_L20.pt
    DL_MLP_L40.pt
  results/
    comparison.csv
    final_baselines.csv
    ml_results.csv
    dl_results.csv
    ml_test_predictions.csv
    dl_test_predictions.csv
    ml_thresholds.json
    metadata.json
    leaderboard_f1_macro.png
```

The first run failed only at report generation because `pandas.to_markdown()` required optional package `tabulate`. The script was patched to use an internal Markdown-table fallback. The successful run is:

```text
ml/improve-classification/logs/run_20260514_152953.log
```

## Successful Evaluation Design

The successful script uses:

```text
train for validation:
  target date < 2022-01-01

validation/model-threshold selection:
  2022-01-01 <= target date < 2023-01-01

final refit:
  target date < 2023-01-01

final test:
  target date >= 2023-01-01
```

Dataset size:

```text
Rows       = 2922
Features   = 27
Train      = 1822
Validation = 260
Train full = 2082
Test       = 840
Test UP rate = 0.4976
```

## Best Current Result

Best full-coverage row from `results/comparison.csv`:

```text
Model      = ENS_FINAL3_th05
Source     = improve_ml
Accuracy   = 0.5464
F1_macro   = 0.5395
AUC        = 0.5348
Threshold  = 0.5
Coverage   = 1.0
N          = 840
```

This improves the saved final best-F1 row:

```text
final LGBM exp_hl100:
  Accuracy = 0.5345
  F1_macro = 0.5341
  AUC      = 0.5234
```

The best full-coverage model bundle is:

```text
ml/improve-classification/models/best_ml_full_coverage.joblib
```

It is an average-probability ensemble over:

```text
LGBM_exp100
GBM_exp100
XGB_linear03
```

with fixed threshold:

```text
0.5
```

## AUC-Oriented Result

Best AUC still comes from the prior final saved all-schemes row:

```text
final_step6_all_schemes_best_auc:
  XGB step_50pct_2x
  Accuracy = 0.5345
  F1_macro = 0.4889
  AUC      = 0.5608
```

In the new run, strong AUC candidates include:

```text
XGB_linear03_th05:
  Accuracy = 0.5262
  F1_macro = 0.4906
  AUC      = 0.5596

LGBM_step50_3:
  Accuracy = 0.5286
  F1_macro = 0.5043
  AUC      = 0.5580
```

These are better viewed as probability/ranking candidates, not best full-coverage classifiers.

## Deep Learning Branch

Deep learning was included via PyTorch:

```text
MLP sequence models: lookback 5, 10, 20, 40
GRU sequence models: lookback 5, 10, 20, 40
```

Best DL full-coverage result:

```text
DL_GRU_L40_th05:
  Accuracy = 0.5190
  F1_macro = 0.5189
  AUC      = 0.5117
```

DL improved validation in places, but did not beat the best ML ensemble on the 2023+ test set.

## Key Files To Inspect First In A New Session

Read these in order:

```text
ml/improve-classification/REPORT.md
ml/improve-classification/results/comparison.csv
ml/improve-classification/results/ml_results.csv
ml/improve-classification/results/dl_results.csv
ml/improve-classification/run_improve_classification.py
ml/improve-classification/logs/run_20260514_152953.log
```

Useful commands:

```bash
sed -n '1,260p' ml/improve-classification/REPORT.md

.venv/bin/python - <<'PY'
import pandas as pd
c = pd.read_csv('ml/improve-classification/results/comparison.csv')
print(c[['Source','Model','Accuracy','F1_macro','AUC','Threshold','ThresholdMode']].head(15).to_string(index=False))
PY
```

## GPU / `nvitop` Process Context

After the successful improve run, no `run_improve_classification.py` process remained.

The process still visible in `nvitop` is a very old job from `ml/classification/final/`, not the new improve script:

```text
PID 1003980
Command: .venv/bin/python ml/classification/final/step6_xgb_vs_gbm.py
STAT: Sl
PPID: 1
Elapsed: 16+ days
cwd: /home/vund/.svn
```

Its child processes shown by `nvidia-smi` / `nvitop` are joblib/loky workers:

```text
1005121 ... LokyProcess-1 ... about 202 MiB
1005122 ... LokyProcess-2 ... about 202 MiB
1005128 ... LokyProcess-3 ... about 202 MiB
1005135 ... LokyProcess-4 ... about 202 MiB
1005146 ... LokyProcess-5 ... about 202 MiB
1005155 ... LokyProcess-6 ... about 202 MiB
1005161 ... LokyProcess-7 ... about 202 MiB
1005171 ... LokyProcess-8 ... about 202 MiB
```

This is not a zombie because state is `Sl` / `Rl`, not `Z`, and it still holds GPU memory. If the user wants to stop it:

```bash
kill 1003980
```

If children remain:

```bash
kill 1005121 1005122 1005128 1005135 1005146 1005155 1005161 1005171
```

Then verify:

```bash
nvidia-smi
ps -p 1003980 -o pid,stat,etime,cmd
```

## Git / Worktree Notes

Current relevant worktree state observed:

```text
 M .gitignore
?? docs/improve/
?? ml/improve-classification/
?? oil_direction_team_docs_md.zip
```

Important:

- `.gitignore` was already modified before/externally; do not revert it unless the user explicitly asks.
- `oil_direction_team_docs_md.zip` remains untracked at repo root.
- `docs/improve/` and `ml/improve-classification/` are new untracked outputs from this session.

## If Continuing The Work

Good next steps:

1. Decide whether to commit the generated artifacts or only the script/report/results.
2. Optionally add a small inference helper for `best_ml_full_coverage.joblib`.
3. If optimizing further, focus on validation-stable ensembles and leakage-safe feature engineering; prior broad feature engineering looked good on validation but weakened on test.
4. Be cautious with selective coverage rows. Some show higher accuracy at lower coverage, but they are not directly comparable to full-coverage classification.

