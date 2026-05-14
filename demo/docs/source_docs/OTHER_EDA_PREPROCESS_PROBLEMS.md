# Problems in `other_eda_preprocess` (Full 6-Step Review)

## Scope

This document reviews the full six-script workflow inside
`other_eda_preprocess`, not just its final CSV.

Reviewed scripts:

1. `step4_transformation.py`
2. `step5_reduction.py`
3. `step6_quality_check.py`
4. `eda_runner.py`
5. `create_presentation_plots.py`
6. `create_eda_notebook.py`

Important framing:

- the branch is explicitly EDA-focused
- many of the issues below become severe only when the branch is reused as a
  forecasting or model-training pipeline

Reference:

- [other_eda_preprocess/README.md](/home/vund/.svn/other_eda_preprocess/README.md:9)

---

## Problem 1. The Branch Starts Too Late in the Pipeline

The README says the authoritative input is:

- `data/processed/dataset_final.csv`

Reference:

- [other_eda_preprocess/README.md](/home/vund/.svn/other_eda_preprocess/README.md:34)

The branch includes:

- `step4_transformation.py`
- `step5_reduction.py`
- `step6_quality_check.py`
- EDA/presentation scripts

But it does not expose a full step1-step3 raw-to-integrated pipeline within that
branch.

Why this is a problem:

- provenance is incomplete inside the branch
- reproducing the entire final dataset from raw sources is less transparent
- it is harder to audit upstream assumptions

Severity:

- medium for EDA-only use
- high for research reproducibility

---

## Problem 2. Same-Day Target Design Is Easy to Misuse

The branch defines the target as:

- `direction = (oil_return > 0).astype(int)`

References:

- [other_eda_preprocess/README.md](/home/vund/.svn/other_eda_preprocess/README.md:5)
- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:83)

This is a same-day target, not a forward target.

Why this is a problem:

- it is acceptable for descriptive EDA
- but it is not a proper `T -> T+1` forecasting target
- downstream users can easily assume the dataset is forecasting-ready when it is
  not

Severity:

- low for descriptive EDA
- high for predictive ML

---

## Problem 3. `step4_transformation.py` Keeps Macro / FRED Timing Leakage

The FRED feature construction happens in:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:122)

`cpi_yoy` is built from raw monthly FRED data and forward-filled onto the daily
timeline:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:143)

Derived macro features are then built from those monthly columns:

- `fed_rate_change`
- `fed_rate_regime`
- `real_rate`

Why this is a problem:

- monthly macro values are treated as available too early on the daily axis
- this creates temporal / feature leakage risk

Affected final columns:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`

Severity:

- high for forecasting or honest test evaluation

---

## Problem 4. `step4_transformation.py` Has Split Leakage in Stress Features

The stress block is built in:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:291)

It fits `MinMaxScaler` on all rows before `2023-01-01`:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:303)
- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:319)

Why this is a problem:

- the scaler is not fit inside a train-only modeling pipeline
- if 2022 is used as validation later, that validation distribution already
  influenced the transformed features

Affected columns:

- `stress_tone`
- `stress_volume`
- `stress_goldstein`
- `geopolitical_stress_index`

Severity:

- medium for EDA
- medium/high for train/validation integrity

---

## Problem 5. `step4_transformation.py` Uses Full-Series Winsorization

`oil_volatility_7d` is created as a rolling volatility feature and then clipped
using the 99th percentile of the full series:

- feature creation:
  [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:95)
- clipping:
  [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:389)

Why this is a problem:

- early rows are clipped using future distribution information
- this is a form of preprocessing leakage

Affected final column:

- `oil_volatility_7d`

Severity:

- medium

---

## Problem 6. `step4_transformation.py` Does Broad Forward-Fill Across All Features

Final cleanup forward-fills all non-date columns:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:422)

Why this is a problem:

- the fill policy is very broad
- it is not feature-specific
- it can smear stale values across multiple feature types

This is not classical future leakage, but it weakens feature semantics.

Severity:

- medium

---

## Problem 7. `step5_reduction.py` Does Not Produce a Modeling-Safe Final Dataset

Reduction happens in:

- [other_eda_preprocess/scripts/step5_reduction.py](/home/vund/.svn/other_eda_preprocess/scripts/step5_reduction.py:41)

This script drops:

- intermediate columns
- same-day close levels
- some collinear groups

But it still leaves important risky columns in `dataset_final.csv`, including:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `oil_volatility_7d`
- `geopolitical_stress_index`

Why this is a problem:

- the final export is not conservative enough for modeling
- there is no separate no-leak final dataset

Severity:

- high

---

## Problem 8. `step5_reduction.py` Still Leaves `oil_return` in the Final Dataset

`oil_return` is not removed by reduction:

- [other_eda_preprocess/scripts/step5_reduction.py](/home/vund/.svn/other_eda_preprocess/scripts/step5_reduction.py:55)

But the EDA target is derived directly from `oil_return`:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:83)

Why this is a problem:

- if someone trains a classifier on `dataset_final.csv` for `direction` and
  forgets to drop `oil_return`, the target source is present in the feature set
- this is a direct target leakage risk

Severity:

- very high

---

## Problem 9. `step6_quality_check.py` Under-Detects Leakage

Quality check file:

- [other_eda_preprocess/scripts/step6_quality_check.py](/home/vund/.svn/other_eda_preprocess/scripts/step6_quality_check.py:1)

Its leakage rule only checks for same-day columns like:

- `vix_close`
- `usd_close`
- `sp500_close`

Reference:

- [other_eda_preprocess/scripts/step6_quality_check.py](/home/vund/.svn/other_eda_preprocess/scripts/step6_quality_check.py:69)

It then prints:

- `No obvious data leakage detected`

Reference:

- [other_eda_preprocess/scripts/step6_quality_check.py](/home/vund/.svn/other_eda_preprocess/scripts/step6_quality_check.py:85)

Why this is a problem:

- it misses the macro timing issue
- it misses stress-feature split leakage
- it misses full-series winsorization leakage

Severity:

- high, because it can create false confidence

---

## Problem 10. `eda_runner.py` Reinforces the Same-Day Framing

The main analysis file:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:1)

does good descriptive EDA, but it is built around:

- same-day target definition
- same-day feature interpretation
- same-day split narrative

Its leakage table marks the following as low risk:

- `oil_volatility_7d`
- `geopolitical_stress_index`
- `cpi_lag`
- `unemployment_lag`
- `fed_rate_change`

Reference:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:665)

Why this is a problem:

- those labels are too optimistic for a strict forecasting setup
- the script acknowledges same-day market returns as medium risk, but is too lax
  on macro/stress/preprocessing issues

Severity:

- medium/high

---

## Problem 11. `eda_runner.py` Uses Contaminated Features in Recommendations

The summary section recommends ideas such as:

- `real_rate × geopolitical_stress_index`
- `high_vol_regime = (oil_volatility_7d > Q75)`

Reference:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:827)

Why this is a problem:

- the recommendations are built on features that are already risky
- this can push future modeling work toward contaminated features instead of
  cleaner alternatives

Severity:

- medium

---

## Problem 12. `create_presentation_plots.py` Freezes Misleading Leakage Labels

Presentation plot generator:

- [other_eda_preprocess/scripts/create_presentation_plots.py](/home/vund/.svn/other_eda_preprocess/scripts/create_presentation_plots.py:1)

Its leakage-risk table labels:

- `oil_volatility_7d` as low
- `cpi_lag` as low / safe
- `geopolitical_stress` as low

Reference:

- [other_eda_preprocess/scripts/create_presentation_plots.py](/home/vund/.svn/other_eda_preprocess/scripts/create_presentation_plots.py:530)

Why this is a problem:

- those labels are then turned into presentation-ready visuals
- the slide deck can unintentionally communicate incorrect safety claims

This is worse than a code-only issue, because it affects narrative and decision
making.

Severity:

- high for communication quality

---

## Problem 13. `create_eda_notebook.py` Contains a Target-Definition Mismatch

Notebook generator:

- [other_eda_preprocess/scripts/create_eda_notebook.py](/home/vund/.svn/other_eda_preprocess/scripts/create_eda_notebook.py:1)

The title says:

- prediction of next-day oil direction

Reference:

- [other_eda_preprocess/scripts/create_eda_notebook.py](/home/vund/.svn/other_eda_preprocess/scripts/create_eda_notebook.py:18)

But the notebook code defines:

- `direction = (oil_return > 0).astype(int)`

Reference:

- [other_eda_preprocess/scripts/create_eda_notebook.py](/home/vund/.svn/other_eda_preprocess/scripts/create_eda_notebook.py:71)

Why this is a problem:

- the narrative says "next day"
- the code implements same-day direction
- this creates conceptual inconsistency for anyone reading the notebook without
  tracing the code carefully

Severity:

- medium

---

## Problem 14. One Final CSV Is Asked to Serve Too Many Roles

The branch centers everything on:

- `other_eda_preprocess/data/processed/dataset_final.csv`

That single dataset is used as:

- final reduced dataset
- quality-check input
- EDA input
- presentation input
- notebook input

Why this is a problem:

- there is no separate:
  - raw transformed dataset
  - no-leak modeling dataset
  - deterministic processed dataset
  - baked-scaled convenience dataset

This makes misuse more likely.

Severity:

- high from an engineering perspective

---

## Severity by Script

| Script | Main issue |
|---|---|
| `step4_transformation.py` | macro timing leakage, stress split leakage, full-series winsorization, broad forward-fill |
| `step5_reduction.py` | final dataset still not modeling-safe, leaves `oil_return` in export |
| `step6_quality_check.py` | leakage checks are too weak and can produce false confidence |
| `eda_runner.py` | strong descriptive EDA, but same-day framing and optimistic leakage labels |
| `create_presentation_plots.py` | presentation layer freezes misleading risk labels |
| `create_eda_notebook.py` | notebook says next-day prediction but implements same-day target |

---

## Final Dataset Risk Table

The final file used by the branch is:

- `other_eda_preprocess/data/processed/dataset_final.csv`

Below is the practical risk table for the final dataset itself.

| Column | Leak type | Severity | Recommended action | Why |
|---|---|---|---|---|
| `oil_return` | `target_source_risk` | `very_high` | `must_drop_if_target_is_direction` | Target direction is derived directly from `oil_return`. |
| `cpi_lag` | `macro_release_timing` | `high` | `drop` | Monthly CPI is exposed too early on the daily timeline. |
| `unemployment_lag` | `macro_release_timing` | `high` | `drop` | Monthly unemployment is exposed too early on the daily timeline. |
| `real_rate` | `derived_macro_timing` | `high` | `drop` | Derived from monthly macro inputs that are visible too early. |
| `fed_rate_change` | `derived_macro_timing` | `medium_high` | `drop_or_retime` | Derived from `fed_funds_rate_lag` timing assumptions. |
| `fed_rate_regime` | `derived_macro_timing` | `medium_high` | `drop_or_retime` | Derived from `fed_funds_rate_lag` timing assumptions. |
| `geopolitical_stress_index` | `split_leakage` | `medium` | `drop_for_strict_eval` | Built from stress features scaled outside a pure train-only modeling fold. |
| `oil_volatility_7d` | `global_preprocessing` | `medium` | `drop_for_strict_eval` | Clipped using a full-series quantile, so early rows use future distribution information. |
| `usd_return` | `same_day_availability` | `medium` | `keep_only_for_eod` | Same-day return is only valid if prediction happens after close. |
| `sp500_return` | `same_day_availability` | `medium` | `keep_only_for_eod` | Same-day return is only valid if prediction happens after close. |
| `vix_return` | `same_day_availability` | `medium` | `keep_only_for_eod` | Same-day return is only valid if prediction happens after close. |

---

## What `other_eda_preprocess` Is Still Good For

Despite the problems above, the branch is still useful for:

- fast descriptive EDA
- generating presentation-ready figures
- communicating exploratory findings
- notebook-based review

It becomes problematic when treated as:

- a clean forecasting dataset
- a strict leakage-safe modeling dataset
- a final ML pipeline

---

## Final Takeaway

`other_eda_preprocess` is not broken as an EDA branch.

Its real problem is that:

- it is easy to mistake an EDA/presentation workflow for a modeling-safe data
  pipeline

Across the full six-step workflow, the main issues are:

- same-day target framing
- macro timing leakage
- stress split leakage
- full-series preprocessing leakage
- weak final leakage checks
- misleading risk communication in presentation artifacts
- no separate model-safe export

So the right interpretation is:

- acceptable for exploration and slides
- unsafe to treat as the final forecasting pipeline without additional cleanup
