# Leakage & Validation Audit Checklist: Daily Oil Direction

**Generated:** 2026-05-14  
**Format:** Markdown, coding-agent friendly

---

## 0. When to run this checklist

Run this checklist:

```text
before accepting any new best model
whenever Accuracy > 0.60
mandatory if Accuracy > 0.63
mandatory if AUC > 0.64
mandatory if any paper-like 0.70+ result appears
```

---

## 1. Target alignment

```text
[ ] y_t = 1[close_{t+1} > close_t] is implemented exactly.
[ ] No feature uses close_{t+1} or return_{t+1}.
[ ] The final row with missing t+1 target is removed.
[ ] Date index is sorted before shifting.
[ ] Asset-specific shifting is used if multiple assets exist.
[ ] Holidays/weekends are handled as next trading day, not next calendar day.
[ ] If threshold label is used, neutral rows are clearly tracked and coverage is reported.
```

Quick test:

```python
# pseudo-code
check = df[['date', 'close']].copy()
check['ret_next_manual'] = np.log(check['close'].shift(-1) / check['close'])
assert np.allclose(df['ret_next'], check['ret_next_manual'], equal_nan=True)
```

---

## 2. Feature timing

```text
[ ] Every feature has an availability rule.
[ ] Every feature has a raw source timestamp if possible.
[ ] Daily close features are used only if prediction occurs after close.
[ ] No same-day macro/supply release is used before actual release time.
[ ] No future revised macro data is used unless vintage data is available.
[ ] News published after cutoff is excluded from date t aggregation.
[ ] Feature engineering windows are backward-looking only.
```

Red flags:

```text
rolling(..., center=True)
shift(-1) in feature code
using entire dataset to compute normalization
using official revised historical macro data without vintage handling
forward-filling weekly data from period end rather than release date
```

---

## 3. EIA/API inventory leakage

```text
[ ] EIA weekly release timestamp is used.
[ ] Holiday-adjusted EIA release schedule is used.
[ ] EIA value is not available before release timestamp.
[ ] Inventory surprise is actual - consensus, not actual - future revised estimate.
[ ] API report is handled separately from EIA report.
[ ] days_since_eia_release is computed from release date, not inventory week-ending date.
[ ] If consensus is missing, feature is marked missing rather than backfilled from future.
```

Correct pattern:

```text
inventory week ending Friday is not the feature availability date.
EIA release date/time is the feature availability date.
```

---

## 4. Macro release leakage

```text
[ ] CPI/PMI/ISM/NFP release timestamps are used.
[ ] Surprises are computed from actual - consensus available before release.
[ ] Revised values are not used as if known at release.
[ ] Macro value is forward-filled only after release timestamp.
[ ] days_since_release is included for lower-frequency variables.
```

---

## 5. News/sentiment leakage

```text
[ ] News items have publication timestamps.
[ ] Time zones are normalized.
[ ] News after market close is assigned to the next eligible prediction window if needed.
[ ] News category/sentiment model is trained only on training data if supervised.
[ ] Topic model / embedding reducer is fit only on training data.
[ ] Duplicate syndicated headlines are de-duplicated or controlled.
[ ] Article updates are not treated as if original timestamp contained final content.
```

Red flags:

```text
aggregating all news by calendar date without timestamp cutoff
training topic model on full corpus including test period
using future headlines to define vocabulary/topics
```

---

## 6. Train/validation/test split

```text
[ ] No random split.
[ ] Splits are chronological.
[ ] Validation is before test.
[ ] Hyperparameter tuning uses validation only.
[ ] Test period is touched once after experiment is locked.
[ ] Walk-forward results are reported across multiple folds.
[ ] No overlapping target leakage if using multi-day labels.
[ ] Purging/embargo is used for overlapping labels or triple-barrier labels.
```

Recommended:

```text
expanding window or rolling window
```

Not recommended:

```text
random train_test_split
stratified random split
cross-validation that shuffles time
```

---

## 7. Preprocessing leakage

```text
[ ] Imputer fit on train only.
[ ] Scaler fit on train only.
[ ] PCA fit on train only.
[ ] Feature selector fit on train only.
[ ] Outlier winsorization thresholds fit on train only.
[ ] Clustering fit on train only.
[ ] SMOTE/class balancing, if used, applied only to train.
```

Quick test:

```text
All preprocessing objects should be stored inside fold-specific pipeline.
No preprocessing should be fit globally before split.
```

---

## 8. Model-selection leakage

```text
[ ] Test set is not used to choose model family.
[ ] Test set is not used to choose feature groups.
[ ] Test set is not used to choose threshold.
[ ] Test set is not used to choose probability calibration.
[ ] Test set is not used repeatedly across hundreds of experiments without correction.
[ ] Best model is selected based on validation or nested CV, then evaluated on test.
```

Red flag:

```text
"We tried 200 configurations and report the best test result."
```

---

## 9. Feature importance sanity check

```text
[ ] Top features are economically plausible.
[ ] No target-derived feature appears in top importance.
[ ] No future timestamp feature appears in top importance.
[ ] Feature importance is stable across folds.
[ ] Removing a suspicious feature does not collapse performance from 0.70 to 0.52.
```

Common suspicious features:

```text
next_return
future_return
target_lead
close_shift_minus_1
pct_change_next
label_encoded_date_leak
post_release_inventory_value_before_release
```

---

## 10. Regime robustness

```text
[ ] Metrics are reported by fold/year.
[ ] Model does not win only in one crisis year.
[ ] Performance is checked in high-vol and low-vol regimes.
[ ] Performance is checked in contango and backwardation regimes.
[ ] Inventory release days and non-release days are evaluated separately.
[ ] OPEC event days and non-event days are evaluated separately.
```

Minimum report:

```text
accuracy_by_year
auc_by_year
class_balance_by_year
number_of_trades_by_year
```

---

## 11. Probability calibration

```text
[ ] Calibration curve is plotted or tabulated.
[ ] Brier score is reported.
[ ] Probability buckets have sufficient sample size.
[ ] 0.55 probability bucket does not imply 0.75 realized hit rate due to small samples.
[ ] Calibration model is fit on validation only.
```

---

## 12. Trading sanity check

```text
[ ] Directional accuracy is not the only metric.
[ ] Transaction costs are included.
[ ] Turnover is reported.
[ ] PnL is not dominated by a few days.
[ ] Max drawdown is reported.
[ ] Sharpe is not annualized incorrectly.
[ ] Long/short exposure is measured.
[ ] Strategy is compared with always-long / no-trade benchmark.
```

---

## 13. Extreme result protocol

If any result exceeds:

```text
Accuracy > 0.63
AUC > 0.64
```

Run:

```text
[ ] Recompute labels manually.
[ ] Drop top 10 suspicious features and rerun.
[ ] Shift all features by +1 day backward/forward as a falsification test.
[ ] Replace target with shuffled target; model should fall to no-skill.
[ ] Train on earlier period, test on latest untouched period.
[ ] Run simple logistic model to see if signal remains.
[ ] Check top SHAP features for timestamp leakage.
[ ] Re-run with completely independent script if possible.
```

If any result exceeds:

```text
Accuracy > 0.70
```

Then classify it as:

```text
unverified high-claim result
```

until independently replicated.

---

## 14. Required audit report template

```markdown
# Leakage Audit Report

## Experiment
- experiment_id:
- model:
- feature_set:
- label:
- train/valid/test periods:

## Metrics
- accuracy:
- auc:
- balanced_accuracy:
- f1:
- mcc:
- log_loss:
- brier:

## Target validation
- passed / failed
- notes:

## Feature timing validation
- passed / failed
- high-risk features:

## Preprocessing validation
- passed / failed
- notes:

## Split validation
- passed / failed
- notes:

## Top features
| rank | feature | group | risk | comment |

## Falsification tests
| test | expected | observed | pass/fail |

## Decision
- accepted / rejected / needs further audit
```

---

## 15. Final rule

```text
In daily oil direction, a modest clean edge is valuable.
A spectacular edge is suspicious until audited.
```
