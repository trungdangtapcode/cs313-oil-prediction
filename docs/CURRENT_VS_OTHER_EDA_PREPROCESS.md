# Current Pipeline vs `other_eda_preprocess`

## Purpose

This document compares the current forecasting-oriented pipeline with the older
`other_eda_preprocess` branch.

Short version:

- `other_eda_preprocess` is closer to an `EDA-first / same-day classification`
  pipeline.
- The current pipeline is a `forecasting / modeling-first` pipeline with extra
  leakage cleanup and additional processing stages.

---

## Executive Summary

| Area | `other_eda_preprocess` | Current pipeline |
|---|---|---|
| Core `step4` feature engineering | Almost the same | Almost the same |
| Target definition | Same-day `direction = oil_return > 0` | Forward target `oil_return_fwd1 = oil_return.shift(-1)` |
| Explicit leakage cleanup | No dedicated step | Yes, `step4b_fix_leakage.py` |
| Deterministic post-processing | No | Yes, `step5b_processing.py` |
| Baked scaled export | No | Yes, `step5c_processing.py` |
| Modeling suitability | Moderate | Higher |
| EDA readability of raw features | Higher | Slightly lower after transforms |

---

## Pipeline Mapping

| Stage | `other_eda_preprocess` | Current pipeline | Main difference |
|---|---|---|---|
| Raw transform | `other_eda_preprocess/scripts/step4_transformation.py` | `scripts/step4_transformation.py` | Nearly identical backbone |
| Reduction | `other_eda_preprocess/scripts/step5_reduction.py` | `scripts/step5_reduction.py` | Same reduction style |
| Leakage cleanup | None | `scripts/step4b_fix_leakage.py` | Current pipeline drops contaminated columns before modeling |
| Deterministic processed export | None | `scripts/step5b_processing.py` | Current pipeline adds `log1p`, `signed_log1p`, `sin/cos` |
| Full-data scaled export | None | `scripts/step5c_processing.py` | Current pipeline can export a baked-scaled dataset |

---

## 1. `step4` Backbone: Mostly the Same

Both branches share the same general `step4` logic:

- Market returns
- Rolling windows
- FRED features
- EIA features
- GDELT features
- ACLED features
- Stress index with `MinMaxScaler`
- Lag features
- Time features
- Winsorization
- Final cleanup

Relevant files:

- Current: `scripts/step4_transformation.py`
- Other: `other_eda_preprocess/scripts/step4_transformation.py`

Important note:

- The similarity at `step4` means the same macro timing and stress-scaling risks
  exist in both branches at this stage.

---

## 2. Target Definition: This Is a Major Difference

### `other_eda_preprocess`

The EDA runner defines the target as same-day direction:

- `direction = (oil_return > 0).astype(int)`
- File: `other_eda_preprocess/scripts/eda_runner.py`

Interpretation:

- the branch is analyzing whether the current day is `UP` or `DOWN`
- this is descriptive / same-day classification oriented

### Current pipeline

The current branch builds an explicit forward target:

- `oil_return_fwd1 = oil_return.shift(-1)`
- `oil_return_fwd1_date = date.shift(-1)`
- File: `scripts/step4_transformation.py`

Interpretation:

- features at time `T`
- target at time `T+1`
- this is forecasting oriented

Practical effect:

- `other_eda_preprocess` is easier to use for same-day EDA
- the current pipeline is structurally more appropriate for predictive modeling

---

## 3. Leakage Handling: The Current Pipeline Adds a Dedicated Cleanup Stage

### `other_eda_preprocess`

There is no dedicated leakage-cleaning step between raw `step4` output and final
dataset export.

Its final dataset still keeps several columns that were flagged later as risky:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `oil_volatility_7d`
- `geopolitical_stress_index`

### Current pipeline

The current pipeline inserts `step4b_fix_leakage.py` before the final modeling
dataset is built.

File:

- `scripts/step4b_fix_leakage.py`

This step removes columns contaminated by:

- monthly release timing leakage
- derived-from-leaky-macro features
- split leakage from stress scaling
- global preprocessing leakage from winsorization

Columns dropped in `step4b`:

- `cpi_lag`
- `unemployment_lag`
- `fed_funds_rate_lag`
- `cpi_yoy`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `stress_tone`
- `stress_volume`
- `stress_goldstein`
- `geopolitical_stress_index`
- `oil_volatility_7d`

Practical effect:

- the current pipeline is more conservative
- it loses some macro/stress features
- but it is cleaner for downstream modeling

---

## 4. Reduction Stage: Similar Style, Different Input Quality

Both branches use a similar `step5_reduction` philosophy:

- drop intermediate columns
- drop weak binary flags
- drop highly collinear feature groups
- remove same-day close levels

Files:

- Current: `scripts/step5_reduction.py`
- Other: `other_eda_preprocess/scripts/step5_reduction.py`

Same-day market levels dropped by both reduction scripts:

- `oil_close`
- `usd_close`
- `sp500_close`
- `vix_close`

But the key difference is upstream:

- `other_eda_preprocess` runs reduction directly on the original transformed data
- the current pipeline runs reduction after `step4b`, on a cleaner input

So even if the reduction logic looks similar, the current final dataset is built
from a stricter pre-filtered base.

---

## 5. `step5b`: Deterministic Processing Exists Only in the Current Pipeline

The current pipeline adds a deterministic processed dataset layer:

- File: `scripts/step5b_processing.py`
- Output: `data/processed/dataset_final_noleak_processed.csv`

This stage intentionally uses only transforms that do not require fitting on the
full dataset:

- cyclical encoding for calendar features
- `log1p` for heavy-tailed positive features
- `signed_log1p` for selected signed heavy-tailed features

### Current `step5b` transforms

Calendar:

- `day_of_week -> day_of_week_sin, day_of_week_cos`
- `month -> month_sin, month_cos`

`log1p`:

- `gdelt_events -> gdelt_events_log1p`
- `conflict_event_count -> conflict_event_count_log1p`
- `conflict_intensity_7d -> conflict_intensity_7d_log1p`
- `fatalities -> fatalities_log1p`
- `fatalities_7d -> fatalities_7d_log1p`
- `gdelt_volume_lag1 -> gdelt_volume_lag1_log1p`
- `vix_lag1 -> vix_lag1_log1p`

`signed_log1p`:

- `net_imports_change_pct -> net_imports_change_pct_slog1p`
- `production_change_pct -> production_change_pct_slog1p`
- `vix_return -> vix_return_slog1p`

The `other_eda_preprocess` branch has no equivalent stage. It keeps those
features in their rawer form.

Practical effect:

- `other_eda_preprocess` is easier to inspect visually in raw EDA
- `step5b` is usually better for modeling because the feature distributions are
  less skewed

---

## 6. `step5c`: Full-Data Scaled Export Exists Only in the Current Pipeline

The current pipeline also adds a convenience scaled export:

- File: `scripts/step5c_processing.py`
- Output: `data/processed/dataset_final_noleak_step5c.csv`

This step:

1. rebuilds the `step5b`-style processed data
2. applies imputation + scaling on the full feature matrix
3. saves both the scaled CSV and a preprocessor artifact

This uses groups defined in:

- `ml/model_preprocessing.py`

Current curated scaling groups:

Standard:

- `yield_spread`
- `gdelt_goldstein`
- `gdelt_goldstein_7d`
- `usd_return`
- `inventory_zscore`

Robust:

- `inventory_change_pct`
- `oil_return`
- `oil_return_lag1`
- `oil_return_lag2`
- `sp500_return`
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

Power:

- `gdelt_tone_7d`
- `gdelt_tone_30d`
- `gdelt_tone_lag1`

Passthrough:

- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`

The `other_eda_preprocess` branch has no equivalent full curated scaling layer.

Important caveat:

- `step5c` is convenient for research and fixed offline experiments
- but it is less strict than `step5b` for honest evaluation because scaling is
  fit on the full dataset

---

## 7. Dataset-Level Comparison

### `other_eda_preprocess`

File:

- `other_eda_preprocess/data/processed/dataset_final.csv`

Shape:

- total columns: `33`
- features excluding `date`: `32`

Still includes:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `oil_volatility_7d`
- `geopolitical_stress_index`

### Current `dataset_final_noleak.csv`

File:

- `data/processed/dataset_final_noleak.csv`

Shape:

- total columns: `28`
- features excluding `date`, `oil_return_fwd1`, `oil_return_fwd1_date`: `25`

Compared with `other_eda_preprocess`, it:

- removes the leak-prone macro/stress/winsorized columns
- adds forward target columns:
  - `oil_return_fwd1`
  - `oil_return_fwd1_date`

### Current `step5b`

File:

- `data/processed/dataset_final_noleak_processed.csv`

Shape:

- total columns: `30`
- modeling features: `27`

Compared with `other_eda_preprocess`, it replaces raw columns with transformed
versions:

- `gdelt_events -> gdelt_events_log1p`
- `conflict_event_count -> conflict_event_count_log1p`
- `conflict_intensity_7d -> conflict_intensity_7d_log1p`
- `fatalities -> fatalities_log1p`
- `fatalities_7d -> fatalities_7d_log1p`
- `gdelt_volume_lag1 -> gdelt_volume_lag1_log1p`
- `vix_lag1 -> vix_lag1_log1p`
- `net_imports_change_pct -> net_imports_change_pct_slog1p`
- `production_change_pct -> production_change_pct_slog1p`
- `vix_return -> vix_return_slog1p`
- `day_of_week -> day_of_week_sin/day_of_week_cos`
- `month -> month_sin/month_cos`

### Current `step5c`

File:

- `data/processed/dataset_final_noleak_step5c_scaler.csv`

Shape:

- total columns: `30`
- modeling features: `27`

Schema:

- same columns as `step5b`
- values already imputed/scaled

---

## 8. Direct Column Differences

### Present in `other_eda_preprocess` final, absent in current `dataset_final_noleak`

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `oil_volatility_7d`
- `geopolitical_stress_index`

### Present in current `dataset_final_noleak`, absent in `other_eda_preprocess` final

- `oil_return_fwd1`
- `oil_return_fwd1_date`

### Present in current `step5b/step5c`, replacing raw forms from `other_eda_preprocess`

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

---

## 9. Leakage and Modeling Consequences

### `other_eda_preprocess`

Pros:

- easier to inspect as a raw EDA-oriented dataset
- preserves more original-looking features

Cons:

- keeps macro/FRED timing-risk columns
- keeps `geopolitical_stress_index` built from stress components scaled before
  strict train-only preprocessing
- keeps `oil_volatility_7d` that was clipped using full-series statistics
- uses same-day direction logic in EDA, not forward forecasting

### Current pipeline

Pros:

- explicitly designed for `T -> T+1` forecasting
- has a dedicated leakage cleanup stage
- separates deterministic processing from fit-based scaling
- supports two final forms:
  - `step5b` for stricter evaluation
  - `step5c` for convenience research

Cons:

- more conservative feature set
- some raw EDA intuition is lost after `log1p/slog1p/sin-cos`
- `step5c` is convenient but not the strictest leakage-safe evaluation dataset

---

## 10. Recommended Usage

Use `other_eda_preprocess` when:

- you want descriptive EDA
- you want to inspect rawer features quickly
- you are not treating that dataset as the final forecasting dataset

Use the current pipeline when:

- you want a forecasting-oriented dataset
- you want explicit leakage cleanup
- you want a cleaner training path

Recommended current exports:

- stricter evaluation: `data/processed/dataset_final_noleak_processed.csv`
- convenience scaled research/training: `data/processed/dataset_final_noleak_step5c_scaler.csv`

---

## Final Takeaway

`other_eda_preprocess` is essentially the older `step4 + step5` lineage, aimed
more at interpretability and same-day EDA.

The current pipeline starts from the same backbone, but adds:

- `step4b`: leakage cleanup by dropping contaminated columns
- `step5b`: deterministic feature reshaping
- `step5c`: optional baked scaling export
- forward target construction for real `T -> T+1` forecasting

That is why the current branch is materially better suited for machine learning,
even though the older branch may still look richer or more intuitive for raw EDA.
