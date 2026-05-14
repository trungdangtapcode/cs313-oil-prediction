# Implementation Plan: Daily Next-Day Oil Price Direction

**Generated:** 2026-05-14  
**Format:** Markdown, coding-agent friendly  
**Audience:** ML / data engineering team

---

## 0. Objective

Build a clean, auditable pipeline for predicting crude oil next-day direction:

```text
Given features known as-of date t, predict whether crude oil close_{t+1} > close_t.
```

Primary target:

```text
y_t = 1[log(close_{t+1} / close_t) > 0]
```

Primary metrics:

```text
Accuracy
AUC
Balanced Accuracy
F1
MCC
Log-loss
Brier score
```

Secondary metrics:

```text
transaction-cost-adjusted PnL
Sharpe
max drawdown
turnover
probability calibration
```

---

## 1. Expected performance target

Current repo, based on the uploaded benchmark report:

```text
Accuracy = 0.5452
AUC      = 0.5586
```

Recommended target bands:

```text
Near-term:
  Accuracy >= 0.56
  AUC      >= 0.58

Good:
  Accuracy 0.58-0.60
  AUC      0.60-0.62

Stretch:
  Accuracy 0.61-0.63
  AUC      0.63-0.65

Above stretch:
  mandatory leakage audit before claiming success
```

Do not set `0.70-0.80` accuracy as a default KPI for the clean daily next-day classification setup.

---

## 2. Repository deliverables

The implementation should produce these artifacts:

```text
data/
  raw/
  interim/
  processed/
  release_calendars/

features/
  feature_registry.yml
  market_features.py
  curve_features.py
  macro_features.py
  supply_features.py
  sentiment_features.py
  asof_join.py

ml/
  classification/
    train_walkforward.py
    evaluate.py
    calibrate.py
    ablation.py
    leakage_checks.py
    results/

reports/
  benchmark_summary.md
  leakage_audit.md
  ablation_results.md
```

---

## 3. Phase 0: Audit current pipeline

### 3.1 Goal

Confirm that current `Accuracy = 0.5452` and `AUC = 0.5586` are clean and reproducible.

### 3.2 Checks

```text
[ ] Recompute target from raw close prices.
[ ] Confirm y_t uses close_{t+1}, not close_t or close_{t-1} by mistake.
[ ] Confirm all features at row t are available by prediction cutoff.
[ ] Confirm no random split is used.
[ ] Confirm scaler/imputer/feature selector is fit only on train.
[ ] Confirm test set is not used for feature selection/model selection.
[ ] Re-run best LGBM and XGBoost from scratch.
[ ] Store seed, parameters, split boundaries, feature list, and metrics.
```

### 3.3 Deliverable

```text
reports/current_pipeline_audit.md
```

Include:

```text
metric table
split dates
feature list
known risks
reproducibility command
```

---

## 4. Phase 1: Build feature registry

### 4.1 Feature registry schema

Create `features/feature_registry.yml`.

Example:

```yaml
features:
  - name: wti_ret_1d
    group: market
    source: FRED / exchange data
    frequency: daily
    availability: close_t
    allowed_for_close_to_next_close: true
    transformation: log_return_1d
    leakage_risk: low

  - name: eia_crude_inventory_surprise
    group: supply
    source: EIA + consensus provider
    frequency: weekly
    availability: release_timestamp
    allowed_for_close_to_next_close: only_if_released_before_cutoff
    transformation: actual_minus_consensus
    leakage_risk: high

  - name: ovx_level
    group: volatility
    source: Cboe / FRED OVXCLS
    frequency: daily
    availability: daily_close
    allowed_for_close_to_next_close: true_if_after_close_prediction
    transformation: level
    leakage_risk: medium
```

### 4.2 Required metadata

Every feature must have:

```text
name
group
source
frequency
raw timestamp column
availability rule
transformation
missing value policy
leakage risk level
```

---

## 5. Phase 2: Add data sources

### 5.1 Market features

Required:

```text
WTI close
Brent close
WTI returns: 1d, 2d, 5d, 10d, 20d
realized volatility: 5d, 10d, 20d
volume and open interest if available
```

### 5.2 Futures curve features

High priority if futures data is available:

```text
CL1, CL2, CL3, CL6, CL12 settlement
CL1-CL2 spread
CL1-CL3 spread
CL1-CL6 spread
slope_1_6
backwardation_flag
contango_flag
roll_yield_proxy
curve PCA components
```

### 5.3 Intermarket features

Add:

```text
S&P 500 return
DXY return
Gold return
Silver return
Heating oil return
Gasoline return
Natural gas return
Copper return
```

### 5.4 Volatility/risk features

Add:

```text
VIX level/change
OVX level/change
MOVE level/change
credit spread if available
```

### 5.5 Macro features

Add as-of release values:

```text
US 2Y yield
US 10Y yield
10Y-2Y curve
CPI surprise
PMI surprise
ISM surprise
NFP surprise
Fed event flag
```

### 5.6 Supply/inventory features

Add:

```text
EIA crude inventory change
EIA crude inventory surprise
EIA Cushing inventory change
EIA gasoline inventory change
EIA distillate inventory change
EIA production
refinery utilization
crude imports
crude exports
API crude inventory surprise if available
Baker Hughes oil rig count
Baker Hughes oil rig change
days_since_eia_release
days_since_rig_release
```

### 5.7 OPEC/supply event features

Add:

```text
opec_meeting_day
opec_plus_meeting_day
opec_announcement_day
opec_cut_flag
opec_hike_flag
opec_hold_flag
opec_cut_size_bpd
opec_surprise_vs_expectation
days_since_opec_event
opec_news_sentiment
opec_policy_index
```

### 5.8 Sentiment features

Start simple:

```text
news_count
sentiment_mean
sentiment_std
positive_ratio
negative_ratio
```

Then topic-level:

```text
topic_opec_sentiment
topic_inventory_sentiment
topic_demand_sentiment
topic_macro_sentiment
topic_geopolitical_sentiment
sentiment_decay_3d
sentiment_decay_5d
```

---

## 6. Phase 3: Implement as-of joins

### 6.1 Core rule

```text
A feature value is allowed only if it was known before the prediction cutoff.
```

### 6.2 Generic as-of join interface

Pseudo-code:

```python
def asof_join_daily(base_df, event_df, date_col, release_ts_col, cutoff_ts_col):
    """
    base_df: one row per trading date, with prediction cutoff timestamp
    event_df: macro/supply/news events with actual release timestamp
    Only attach events where event.release_ts <= base.cutoff_ts.
    """
    pass
```

### 6.3 Weekly EIA example

```text
If EIA is released Wednesday 10:30 ET:
  - If prediction cutoff is Wednesday after close, EIA may be used for Thursday prediction.
  - If prediction cutoff is Wednesday before release, it cannot be used.
  - If holidays shift release day, use actual release calendar.
```

### 6.4 Macro example

```text
CPI / PMI / ISM:
  use actual release timestamp
  store surprise on release day
  forward-fill latest known value only after release
  avoid revised data unless vintage data is available
```

### 6.5 News example

```text
Only aggregate news published before cutoff.
Do not aggregate all headlines by calendar date if some are published after the trading close.
```

---

## 7. Phase 4: Label experiments

Run labels as separate experiments.

### 7.1 Raw direction label

```text
y_raw = 1[ret_next > 0]
```

This is the main benchmark.

### 7.2 Threshold labels

```text
y_10bp = 1 if ret_next > 0.001
       = 0 if ret_next < -0.001
       = neutral otherwise
```

Report:

```text
accuracy
AUC
coverage
class balance
ignored neutral share
```

### 7.3 Triple-barrier label

For trading-style labels:

```text
upper_barrier = +k * volatility
lower_barrier = -k * volatility
max_horizon = H days
```

Report separately from raw next-day direction.

### 7.4 Meta-labeling

Use only after a base signal exists:

```text
base signal proposes trade direction
meta model predicts whether trade should be taken
```

---

## 8. Phase 5: Model matrix

### 8.1 Baseline models

```text
majority class
sign persistence
logistic regression
elastic-net logistic
```

### 8.2 Main tabular models

```text
LightGBM
XGBoost
CatBoost
Random Forest
ExtraTrees
```

### 8.3 Stacking

Only after base models are stable:

```text
base models: LGBM, XGBoost, CatBoost, Logistic
meta model: logistic regression or ridge
out-of-fold predictions only
no test leakage
```

### 8.4 Sequence models

Research track:

```text
GRU
LSTM
TCN
Transformer
```

### 8.5 Graph models

Advanced research track:

```text
ST-GNN
MTGNN
ASTGCN
attention graph model
```

Do not prioritize until tabular pipeline is clean.

---

## 9. Phase 6: Evaluation design

### 9.1 Walk-forward split

Example:

```text
Fold 1:
  train: 2015-2018
  valid: 2019
  test : 2020

Fold 2:
  train: 2015-2019
  valid: 2020
  test : 2021

Fold 3:
  train: 2015-2020
  valid: 2021
  test : 2022
```

### 9.2 Avoid repeated test tuning

```text
Use validation for model/feature choices.
Touch test once per locked experiment.
Log every experiment.
```

### 9.3 Required report table

For every experiment:

```text
experiment_id
label_type
feature_set
model
train_period
valid_period
test_period
accuracy
balanced_accuracy
auc
f1
mcc
log_loss
brier
turnover
cost_adjusted_pnl
sharpe
max_drawdown
```

---

## 10. Phase 7: Ablation matrix

Run these in order:

| Experiment | Feature set | Purpose |
|---:|---|---|
| A0 | baseline/persistence only | no-skill comparison |
| A1 | market + technical | current-style benchmark |
| A2 | A1 + futures curve | test curve signal |
| A3 | A2 + VIX/OVX/MOVE | test risk/vol signal |
| A4 | A3 + intermarket | test Gold/DXY/SPX/energy complex |
| A5 | A4 + macro | test macro release signal |
| A6 | A5 + supply/inventory | test EIA/API/rig signal |
| A7 | A6 + OPEC events | test policy/supply shock signal |
| A8 | A7 + sentiment | test news signal |
| A9 | A8 + stacking | test ensemble gain |

Acceptance rule:

```text
A new feature group is useful only if it improves average walk-forward metrics and does not only improve one lucky fold.
```

---

## 11. Phase 8: Explainability and diagnostics

Required diagnostics:

```text
SHAP summary per fold
permutation importance per fold
feature stability across folds
calibration curve
probability bucket hit rate
regime-wise results
```

Suggested regimes:

```text
2014-2016 oil crash if data available
2020 COVID shock
2022 geopolitical / inflation shock
2023-2026 recent regime
high volatility vs low volatility
backwardation vs contango
inventory release days vs non-release days
OPEC event days vs non-event days
```

---

## 12. Weekly side experiment

### 12.1 Why test weekly

```text
Daily labels are noisy.
Macro/supply/sentiment effects may materialize over multiple days.
Weekly may improve signal-to-noise.
```

### 12.2 Risks

```text
fewer samples
greater fold variance
higher overfit risk
```

### 12.3 Recommended weekly setup

```text
Target: weekly log return sign
Features: weekly aggregated/as-of features
Models: logistic, LGBM, XGBoost, CatBoost
Validation: walk-forward by year
Report: do not compare directly to daily accuracy
```

---

## 13. Acceptance gates

Before a model is accepted:

```text
[ ] Target alignment checked.
[ ] As-of feature availability checked.
[ ] Walk-forward split used.
[ ] Preprocessing fit only on train.
[ ] Hyperparameters tuned on validation only.
[ ] Test set not used repeatedly.
[ ] Ablation shows contribution of feature group.
[ ] Metrics stable across folds.
[ ] Result not driven by one regime.
[ ] If Accuracy >0.63 or AUC >0.64, leakage audit completed.
```

---

## 14. Suggested first sprint

### Sprint 1 goal

Move from:

```text
Accuracy 0.5452
AUC      0.5586
```

toward:

```text
Accuracy >= 0.56
AUC      >= 0.58
```

### Sprint 1 tasks

```text
1. Reproduce current results.
2. Build feature registry.
3. Add OVX, MOVE, Gold, S&P 500, DXY.
4. Add clean walk-forward evaluation script.
5. Run A0-A4 ablation.
6. Produce report with metrics and leakage checklist.
```

### Sprint 1 non-goals

```text
No graph neural network yet.
No complex LLM sentiment yet.
No weekly result mixed into daily benchmark.
No 0.70+ KPI.
```
