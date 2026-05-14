# `other_eda_preprocess` Final vs Current `step5b` / `step5c`

## Scope

This note compares the final dataset from `other_eda_preprocess` with the two
current end-stage datasets:

- `other_eda_preprocess/data/processed/dataset_final.csv`
- `data/processed/dataset_final_noleak_processed.csv` (`step5b`)
- `data/processed/dataset_final_noleak_step5c_scaler.csv` (`step5c`)

---

## Quick Summary

| Dataset | Shape | Date range | Missing | Main purpose |
|---|---:|---|---:|---|
| `other_final` | `2923 x 33` | `2015-01-07 -> 2026-03-20` | `0` | EDA / same-day classification snapshot |
| `step5b` | `2922 x 30` | `2015-01-07 -> 2026-03-19` | `0` | deterministic processed, cleaner for modeling |
| `step5c` | `2922 x 30` | `2015-01-07 -> 2026-03-19` | `0` | same schema as `step5b`, but already scaled |

Important note:

- `step5b` and `step5c` have the same schema
- `step5c` differs only in values, because the features are imputed/scaled
- `step5b/step5c` have one fewer row because they include a forward target and
  therefore drop the last row with no `T+1` target

---

## 1. Schema Comparison

### `other_final`

File:

- [dataset_final.csv](/home/vund/.svn/other_eda_preprocess/data/processed/dataset_final.csv)

Columns:

- `date`
- `yield_spread`
- `cpi_lag`
- `unemployment_lag`
- `inventory_change_pct`
- `gdelt_goldstein`
- `gdelt_events`
- `gdelt_tone_7d`
- `gdelt_tone_30d`
- `gdelt_goldstein_7d`
- `conflict_event_count`
- `fatalities`
- `oil_return`
- `usd_return`
- `sp500_return`
- `vix_return`
- `oil_volatility_7d`
- `fed_rate_change`
- `fed_rate_regime`
- `real_rate`
- `inventory_zscore`
- `production_change_pct`
- `net_imports_change_pct`
- `conflict_intensity_7d`
- `fatalities_7d`
- `geopolitical_stress_index`
- `oil_return_lag1`
- `oil_return_lag2`
- `vix_lag1`
- `gdelt_tone_lag1`
- `gdelt_volume_lag1`
- `day_of_week`
- `month`

### `step5b`

File:

- [dataset_final_noleak_processed.csv](/home/vund/.svn/data/processed/dataset_final_noleak_processed.csv)

Columns:

- `date`
- `oil_return_fwd1`
- `oil_return_fwd1_date`
- `yield_spread`
- `inventory_change_pct`
- `gdelt_goldstein`
- `gdelt_tone_7d`
- `gdelt_tone_30d`
- `gdelt_goldstein_7d`
- `oil_return`
- `usd_return`
- `sp500_return`
- `inventory_zscore`
- `oil_return_lag1`
- `oil_return_lag2`
- `gdelt_tone_lag1`
- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`
- `gdelt_events_log1p`
- `conflict_event_count_log1p`
- `conflict_intensity_7d_log1p`
- `fatalities_log1p`
- `fatalities_7d_log1p`
- `gdelt_volume_lag1_log1p`
- `vix_lag1_log1p`
- `net_imports_change_pct_slog1p`
- `production_change_pct_slog1p`
- `vix_return_slog1p`

### `step5c`

File:

- [dataset_final_noleak_step5c_scaler.csv](/home/vund/.svn/data/processed/dataset_final_noleak_step5c_scaler.csv)

Columns:

- identical to `step5b`

---

## 2. Columns Present Only in `other_final`

These columns exist in `other_final` but not in `step5b/step5c`:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `geopolitical_stress_index`
- `oil_volatility_7d`
- `conflict_event_count`
- `conflict_intensity_7d`
- `fatalities`
- `fatalities_7d`
- `gdelt_events`
- `gdelt_volume_lag1`
- `vix_lag1`
- `net_imports_change_pct`
- `production_change_pct`
- `vix_return`
- `day_of_week`
- `month`

These columns fall into two groups:

1. leak-prone or contamination-prone columns:
   - `cpi_lag`
   - `unemployment_lag`
   - `real_rate`
   - `fed_rate_change`
   - `fed_rate_regime`
   - `geopolitical_stress_index`
   - `oil_volatility_7d`

2. raw columns that were deliberately transformed in `step5b/step5c`:
   - `conflict_event_count`
   - `conflict_intensity_7d`
   - `fatalities`
   - `fatalities_7d`
   - `gdelt_events`
   - `gdelt_volume_lag1`
   - `vix_lag1`
   - `net_imports_change_pct`
   - `production_change_pct`
   - `vix_return`
   - `day_of_week`
   - `month`

---

## 3. Columns Present Only in `step5b/step5c`

These columns do not exist in `other_final`:

- `oil_return_fwd1`
- `oil_return_fwd1_date`
- `conflict_event_count_log1p`
- `conflict_intensity_7d_log1p`
- `fatalities_log1p`
- `fatalities_7d_log1p`
- `gdelt_events_log1p`
- `gdelt_volume_lag1_log1p`
- `vix_lag1_log1p`
- `net_imports_change_pct_slog1p`
- `production_change_pct_slog1p`
- `vix_return_slog1p`
- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`

These reflect the current pipeline design:

1. forward target support:
   - `oil_return_fwd1`
   - `oil_return_fwd1_date`

2. deterministic feature reshaping:
   - `log1p` for heavy-tailed positive features
   - `signed_log1p` for signed heavy-tailed features
   - cyclical encoding for calendar features

---

## 4. Direct Mapping: Raw vs Transformed

`other_final` keeps several raw features that are replaced in `step5b/step5c`:

| `other_final` raw column | `step5b/step5c` replacement |
|---|---|
| `gdelt_events` | `gdelt_events_log1p` |
| `conflict_event_count` | `conflict_event_count_log1p` |
| `conflict_intensity_7d` | `conflict_intensity_7d_log1p` |
| `fatalities` | `fatalities_log1p` |
| `fatalities_7d` | `fatalities_7d_log1p` |
| `gdelt_volume_lag1` | `gdelt_volume_lag1_log1p` |
| `vix_lag1` | `vix_lag1_log1p` |
| `net_imports_change_pct` | `net_imports_change_pct_slog1p` |
| `production_change_pct` | `production_change_pct_slog1p` |
| `vix_return` | `vix_return_slog1p` |
| `day_of_week` | `day_of_week_sin`, `day_of_week_cos` |
| `month` | `month_sin`, `month_cos` |

Meaning:

- `other_final` is more raw and more presentation-friendly
- `step5b/step5c` are more model-oriented

---

## 5. Shared Columns

These columns are shared between `other_final` and `step5b`:

- `yield_spread`
- `inventory_change_pct`
- `gdelt_goldstein`
- `gdelt_tone_7d`
- `gdelt_tone_30d`
- `gdelt_goldstein_7d`
- `oil_return`
- `usd_return`
- `sp500_return`
- `inventory_zscore`
- `oil_return_lag1`
- `oil_return_lag2`
- `gdelt_tone_lag1`

For these shared columns:

- `other_final` and `step5b` keep roughly the same raw values
- `step5c` has the same columns, but values are already scaled

Examples:

- `yield_spread`: raw mean in `other_final` and `step5b` is the same order;
  `step5c` mean is near `0`
- `gdelt_goldstein`: raw in `other_final/step5b`; centered in `step5c`
- `gdelt_tone_7d`: raw in `other_final/step5b`; transformed/scaled in `step5c`

---

## 6. Leakage / Safety Comparison

### `other_final`

Still keeps leak-prone or contamination-prone columns:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `geopolitical_stress_index`
- `oil_volatility_7d`

Warning when training:

- `oil_return`

Interpretation:

- useful for EDA
- not a conservative modeling-safe final export

### `step5b`

Compared with `other_final`, `step5b`:

- removes the leak-prone macro/stress/oil-volatility block
- keeps forward-target columns
- converts many heavy-tailed features into safer deterministic forms
- remains stricter than `step5c` because it does not fit scalers on the full
  dataset

Interpretation:

- the best of the three if the goal is honest downstream evaluation

### `step5c`

Compared with `step5b`, `step5c`:

- keeps the same schema
- applies imputation and curated scaling on the full exported feature matrix

Interpretation:

- best for convenience research or quick training
- not as strict as `step5b` for leakage-sensitive evaluation

---

## 7. Conceptual Difference

### `other_final`

Conceptually tied to:

- same-day direction EDA
- presentation outputs
- descriptive classification analysis

### `step5b / step5c`

Conceptually tied to:

- forward-target forecasting
- explicit leakage cleanup
- deterministic processing
- optional scaled export

This is why the datasets feel different even when many base features overlap.

---

## 8. Recommended Usage

Use `other_final` when:

- you want rawer EDA
- you want to inspect the old presentation-first branch
- you are not treating it as the final forecasting dataset

Use `step5b` when:

- you want the cleanest current modeling dataset
- you want deterministic processing without baking scaler statistics into the
  export

Use `step5c` when:

- you want the same schema as `step5b`
- but you want a ready-to-train scaled dataset for convenience

---

## Bottom Line

`other_final` is richer in raw EDA-facing columns, but it still carries:

- leak-prone macro features
- preprocessing-contaminated features
- same-day EDA assumptions

`step5b` is the clean modeling-oriented replacement.

`step5c` is the scaled convenience version of `step5b`.
