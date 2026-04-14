# Problems in `other_eda_preprocess`

## Scope

This document summarizes the main problems in the `other_eda_preprocess`
pipeline.

Important distinction:

- if the branch is used only for descriptive EDA, some of these issues are less
  severe
- if it is used as a modeling or forecasting dataset, these issues become
  material

---

## 1. No Dedicated Leakage-Cleanup Stage

The branch goes from feature engineering in
`other_eda_preprocess/scripts/step4_transformation.py` straight into reduction in
`other_eda_preprocess/scripts/step5_reduction.py`.

There is no equivalent of the current pipeline's `step4b_fix_leakage.py`.

Consequence:

- columns with known timing or preprocessing contamination remain in the final
  dataset
- the final dataset is not a conservative modeling-safe export

---

## 2. Direct Target Leakage Risk If Used for Classification Training

In the EDA runner, the target is defined as:

- `direction = (oil_return > 0).astype(int)`
- file: [eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:83)

But in the reduced final dataset, `oil_return` is still kept:

- `oil_return` is not dropped in
  [step5_reduction.py](/home/vund/.svn/other_eda_preprocess/scripts/step5_reduction.py:55)

Why this is a problem:

- if someone takes `dataset_final.csv` and trains a classifier for `direction`
  without carefully removing `oil_return`, the target is effectively encoded in
  a feature
- that is a direct leakage path

Practical impact:

- this is one of the most serious problems if the dataset is reused outside the
  original EDA notebook/script assumptions

---

## 3. Macro / FRED Release-Timing Leakage

The branch builds FRED-derived features in
[step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:122).

The key issue is that CPI-derived features are forward-filled directly onto the
daily timeline:

- `cpi_yoy` is reindexed with `method="ffill"` at
  [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:143)

Related features are then derived from monthly macro columns:

- `fed_rate_change`
- `fed_rate_regime`
- `real_rate`

Why this is a problem:

- monthly macro data are treated as available too early on the daily timeline
- the dataset assumes the new monthly value is visible before the real release
  timing is fully respected

Affected final columns:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`

Consequence:

- if the dataset is used for forecasting, these macro columns are not timing-safe

---

## 4. Split Leakage in `geopolitical_stress_index`

The stress block is created in
[step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:291).

The code fits `MinMaxScaler` on all rows before `2023-01-01`:

- see [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:303)
- scaler fit happens at [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:319)

Why this is a problem:

- the transformation is not fit strictly on a pure training fold
- if 2022 is later used as validation, the feature has already absorbed
  validation-period distribution information

Affected columns:

- `stress_tone`
- `stress_volume`
- `stress_goldstein`
- `geopolitical_stress_index`

Consequence:

- this is not row-level future leakage
- but it is still a train/validation contamination risk

---

## 5. Full-Series Winsorization Leakage

`oil_volatility_7d` is built from rolling volatility and then clipped using the
99th percentile of the full series:

- volatility feature creation:
  [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:95)
- clipping:
  [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:389)

Why this is a problem:

- early rows are clipped using a threshold learned from future data
- this introduces full-series preprocessing leakage

Affected final column:

- `oil_volatility_7d`

Consequence:

- the feature is not strictly train-only transformed

---

## 6. Same-Day Feature Availability Risk

Market returns are computed from current-day closes:

- `oil_return`
- `usd_return`
- `sp500_return`
- `vix_return`
- file:
  [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:50)

The reduction step drops same-day close levels:

- `oil_close`, `usd_close`, `sp500_close`, `vix_close`
- file:
  [step5_reduction.py](/home/vund/.svn/other_eda_preprocess/scripts/step5_reduction.py:85)

But it keeps same-day returns:

- `oil_return`
- `usd_return`
- `sp500_return`
- `vix_return`

Why this is a problem:

- for same-day descriptive EDA, this is fine
- for a strict forecasting setup, these are only valid if prediction happens
  after day-end data are fully known

Consequence:

- the branch does not enforce a clear feature-availability contract
- it is easy to misuse the data in an intraday or pre-close setting

---

## 7. Aggressive Forward-Fill Across All Non-Date Columns

Final cleanup forward-fills all non-date columns with `limit=3`:

- [step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:422)

Why this is a problem:

- the fill is broad and not column-specific
- it can smear previous values into rows where a feature may be temporarily
  unavailable
- this is not classical future leakage, but it can still distort feature meaning

Consequence:

- feature semantics become less strict
- data quality becomes more assumption-heavy

---

## 8. No Deterministic Post-Processing Layer for Heavy-Tail Features

The branch stops after reduction and does not add a deterministic processing
stage like the current pipeline's `step5b`.

So raw heavy-tailed columns remain in the final dataset, for example:

- `gdelt_events`
- `conflict_event_count`
- `conflict_intensity_7d`
- `fatalities`
- `fatalities_7d`
- `gdelt_volume_lag1`
- `net_imports_change_pct`
- `production_change_pct`
- `vix_return`
- `day_of_week`
- `month`

Why this is a problem:

- these distributions are harder for many models
- skew and tail behavior remain untreated
- the dataset is less modeling-friendly out of the box

Consequence:

- extra preprocessing is still needed before many ML experiments

---

## 9. No Separate Modeling-Safe Export

The branch has one main reduced dataset:

- `other_eda_preprocess/data/processed/dataset_final.csv`

But it does not separate:

- descriptive EDA dataset
- strict modeling dataset
- deterministic processed dataset
- baked-scaled convenience dataset

Why this is a problem:

- one file is asked to serve too many purposes
- downstream users can easily assume it is safe for training when it is not

Consequence:

- higher risk of accidental misuse

---

## 10. Same-Day EDA Orientation Instead of Forecasting Orientation

The branch is built around same-day direction analysis:

- target is derived from `oil_return` itself
- file:
  [eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:83)

It does not create a forward target like:

- `oil_return_fwd1`
- `oil_return_fwd1_date`

Why this is a problem:

- it is fine for descriptive same-day EDA
- but it is not aligned with a real `T -> T+1` forecasting workflow

Consequence:

- the branch is structurally weaker for predictive ML than the current pipeline

---

## Severity Summary

### High severity for modeling

- direct target leakage risk through `oil_return`
- macro/FRED release-timing leakage
- no dedicated modeling-safe export

### Medium severity

- stress-feature split leakage
- full-series winsorization of `oil_volatility_7d`
- unclear same-day feature availability assumptions

### Lower severity but still important

- broad forward-fill assumptions
- no deterministic post-processing layer
- EDA-first structure encourages reuse in settings it was not designed for

---

## Final Takeaway

`other_eda_preprocess` is acceptable as an exploratory branch for descriptive
EDA, but it has several structural problems if reused as a forecasting or model
training dataset:

- it keeps leak-prone macro/stress/preprocessed features
- it mixes same-day target logic with reusable feature exports
- it lacks a dedicated leakage-cleanup step
- it does not separate EDA data from model-safe data

In short:

- good for exploration
- unsafe to treat as the final forecasting dataset without extra cleanup
