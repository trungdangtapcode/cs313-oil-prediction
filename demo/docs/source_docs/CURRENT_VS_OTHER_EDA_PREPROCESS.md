# Current Pipeline vs `other_eda_preprocess` (Full 6-Step View)

## Scope

This document compares the current repository with the full six-script workflow
inside `other_eda_preprocess`.

Important framing:

- `other_eda_preprocess` is an `EDA / presentation` repository snapshot
- the current repository is a broader `data pipeline + modeling` repository

So the comparison must cover:

1. transformation
2. reduction
3. quality check
4. EDA runner
5. presentation plotting
6. notebook generation

and not just the final CSV.

---

## High-Level Difference

### `other_eda_preprocess`

The branch is explicitly documented as:

- an EDA-focused project
- stopping at EDA / presentation-ready analysis
- not containing a completed modeling pipeline

Reference:

- [other_eda_preprocess/README.md](/home/vund/.svn/other_eda_preprocess/README.md:9)

Its six active scripts are:

1. `step4_transformation.py`
2. `step5_reduction.py`
3. `step6_quality_check.py`
4. `eda_runner.py`
5. `create_presentation_plots.py`
6. `create_eda_notebook.py`

### Current repository

The current repository is broader:

- raw-data ingestion scripts exist
- cleaning and integration scripts exist
- there is an explicit leakage-cleanup step
- there is a deterministic processed export
- there is a baked-scaled convenience export
- there is a full `ml/classification` training pipeline

Reference:

- [scripts/PIPELINE_STEPS.md](/home/vund/.svn/scripts/PIPELINE_STEPS.md:1)

---

## Full Flow Mapping

| `other_eda_preprocess` step | Purpose in that repo | Current counterpart | Main difference |
|---|---|---|---|
| `step4_transformation.py` | Build transformed features | `scripts/step4_transformation.py` | Backbone is almost the same |
| `step5_reduction.py` | Build `dataset_final.csv` | `scripts/step5_reduction.py` and `scripts/step4b_fix_leakage.py` | Current repo adds leakage cleanup before final modeling export |
| `step6_quality_check.py` | Console-only final checks | `scripts/step4b_fix_leakage.py`, `step6_quality_check.py`, upgraded EDA checks | Current repo is stricter on leakage, not just NaN/INF |
| `eda_runner.py` | Main EDA reports / tables / figures | `eda_classification/eda_clf.py` and upgraded EDA batches | Current repo uses forward target and modeling-oriented EDA |
| `create_presentation_plots.py` | Slide-ready plots | no single identical script; current repo uses EDA outputs and docs | Current repo does not freeze the same presentation assumptions |
| `create_eda_notebook.py` | Generate notebook from final dataset | no direct equivalent | Current repo is more script-first than notebook-template-first |

---

## 0. Upstream Coverage

### `other_eda_preprocess`

The README says the authoritative input is:

- `data/processed/dataset_final.csv`

Reference:

- [other_eda_preprocess/README.md](/home/vund/.svn/other_eda_preprocess/README.md:34)

That means the repo snapshot begins effectively from:

- transformed / reduced / EDA-ready data

It does not include a full visible step1-step3 raw-to-integrated pipeline inside
that branch.

### Current repository

The current repository has explicit upstream stages:

- raw ingestion
- cleaning
- integration
- transformation
- leakage cleanup
- reduction
- processed exports
- training

Reference:

- [scripts/PIPELINE_STEPS.md](/home/vund/.svn/scripts/PIPELINE_STEPS.md:1)

Main implication:

- `other_eda_preprocess` is easier to use as a self-contained EDA snapshot
- the current repo is much better if you need end-to-end provenance

---

## 1. Transformation Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/step4_transformation.py](/home/vund/.svn/other_eda_preprocess/scripts/step4_transformation.py:1)

This script does:

- market returns
- rolling windows
- FRED-derived features
- EIA-derived features
- GDELT-derived features
- ACLED-derived features
- `geopolitical_stress_index`
- lag features
- time features
- winsorization
- final cleanup

### Current repository

File:

- [scripts/step4_transformation.py](/home/vund/.svn/scripts/step4_transformation.py:1)

The backbone is almost identical.

Important difference:

- the current script also creates a forward target:
  - `oil_return_fwd1`
  - `oil_return_fwd1_date`

Reference:

- [scripts/step4_transformation.py](/home/vund/.svn/scripts/step4_transformation.py:446)

### Consequence

At pure `step4` level:

- both repos have almost the same feature-engineering DNA
- the big structural change is target design:
  - `other_eda_preprocess`: same-day direction analysis
  - current repo: `T -> T+1` forecasting

---

## 2. Reduction Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/step5_reduction.py](/home/vund/.svn/other_eda_preprocess/scripts/step5_reduction.py:41)

This script builds:

- `dataset_final.csv`

It drops:

- intermediate stress columns
- weak binary columns
- some macro collinearity columns
- some raw supply columns
- some GDELT redundancy
- same-day close levels

### Current repository

File:

- [scripts/step5_reduction.py](/home/vund/.svn/scripts/step5_reduction.py:41)

The reduction logic is largely the same.

But the current repo adds:

- `step4b_fix_leakage.py`

before reduction is used for modeling.

Reference:

- [scripts/step4b_fix_leakage.py](/home/vund/.svn/scripts/step4b_fix_leakage.py:3)

### Consequence

Reduction style is similar, but the input quality is different:

- `other_eda_preprocess` reduces the original transformed dataset directly
- the current repo can reduce a leakage-scrubbed dataset first

That makes the current downstream exports materially cleaner.

---

## 3. Quality Check Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/step6_quality_check.py](/home/vund/.svn/other_eda_preprocess/scripts/step6_quality_check.py:1)

This script checks:

- NaN
- INF
- a small same-day leakage rule set
- target distribution
- train/test split
- sample rows

What it does well:

- simple sanity check
- quick shape / missing / split validation

What it does not catch:

- macro/FRED timing leakage
- stress-feature split leakage
- full-series winsorization leakage

Its leakage check only looks for:

- `vix_close`
- `usd_close`
- `sp500_close`

Reference:

- [other_eda_preprocess/scripts/step6_quality_check.py](/home/vund/.svn/other_eda_preprocess/scripts/step6_quality_check.py:69)

### Current repository

The current repo handles this differently:

- leakage cleanup is done structurally in
  [step4b_fix_leakage.py](/home/vund/.svn/scripts/step4b_fix_leakage.py:36)
- deterministic processing is isolated in
  [step5b_processing.py](/home/vund/.svn/scripts/step5b_processing.py:3)
- model-time preprocessing groups are explicit in
  [ml/model_preprocessing.py](/home/vund/.svn/ml/model_preprocessing.py:1)

### Consequence

`other_eda_preprocess/step6_quality_check.py` is useful as a basic sanity pass,
but it is not a strong leakage audit.

---

## 4. Main EDA Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:1)

This is the core analysis script.

It:

- loads `dataset_final.csv`
- derives `direction = (oil_return > 0)`
- splits train/test at `2023-01-01`
- runs data quality, target analysis, feature distributions, time-series tests,
  feature-target tests, leakage/split checks, and summary export

Target definition:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:83)

Its leakage table explicitly classifies some features as low/medium/high:

- [other_eda_preprocess/scripts/eda_runner.py](/home/vund/.svn/other_eda_preprocess/scripts/eda_runner.py:665)

### Current repository

The current repo's EDA is now centered around:

- `eda_classification/eda_clf.py`

and recent upgraded runs such as:

- `eda_classification/step5_upgraded`

The current EDA uses:

- forward target semantics
- target-date-based splitting
- richer feature ranking
- train/test shift analysis
- leakage-oriented interpretation tied to the current no-leak datasets

### Consequence

`other_eda_preprocess/eda_runner.py` is strong for descriptive same-day EDA, but
it is not aligned with the current forecasting-oriented data contracts.

---

## 5. Presentation Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/create_presentation_plots.py](/home/vund/.svn/other_eda_preprocess/scripts/create_presentation_plots.py:1)

This script turns the EDA into slide-ready plots.

It is valuable because:

- it packages the narrative cleanly
- it forces a concrete summary
- it makes the repo presentation-friendly

But it also freezes some assumptions into visuals and labels, including a
leakage-risk table:

- [other_eda_preprocess/scripts/create_presentation_plots.py](/home/vund/.svn/other_eda_preprocess/scripts/create_presentation_plots.py:527)

### Current repository

The current repo does not have one single identical presentation script.

Instead it has:

- richer EDA batches
- markdown reports
- training result reports

### Consequence

`other_eda_preprocess` is better if the immediate goal is slide generation.

The current repo is better if the immediate goal is research that must stay
consistent with the final modeling pipeline.

---

## 6. Notebook Generation Stage

### `other_eda_preprocess`

File:

- [other_eda_preprocess/scripts/create_eda_notebook.py](/home/vund/.svn/other_eda_preprocess/scripts/create_eda_notebook.py:1)

This script generates a notebook template around `dataset_final.csv`.

It is useful because:

- it standardizes notebook structure
- it makes review easier for a human analyst

But it inherits the same assumptions as the EDA runner.

### Current repository

The current repo is more script/report-first than notebook-template-first.

There is no direct one-to-one notebook generator that anchors the full current
pipeline.

### Consequence

`other_eda_preprocess` is more polished as an EDA presentation package.

The current repo is more polished as an engineering pipeline.

---

## Final Dataset Comparison

### `other_eda_preprocess`

Main final file:

- [other_eda_preprocess/data/processed/dataset_final.csv](/home/vund/.svn/other_eda_preprocess/data/processed/dataset_final.csv)

Shape:

- `33` total columns

Still keeps final columns such as:

- `cpi_lag`
- `unemployment_lag`
- `real_rate`
- `fed_rate_change`
- `fed_rate_regime`
- `oil_volatility_7d`
- `geopolitical_stress_index`

### Current repository

Main modeling-oriented exports:

- [dataset_final_noleak.csv](/home/vund/.svn/data/processed/dataset_final_noleak.csv)
- [dataset_final_noleak_processed.csv](/home/vund/.svn/data/processed/dataset_final_noleak_processed.csv)
- [dataset_final_noleak_step5c_scaler.csv](/home/vund/.svn/data/processed/dataset_final_noleak_step5c_scaler.csv)

The current repo can therefore separate:

- no-leak reduced export
- deterministic processed export
- baked-scaled convenience export

`other_eda_preprocess` has only one main final dataset.

---

## Strengths of `other_eda_preprocess`

It is worth being precise here. The branch is not simply "bad".

It is strong at:

- fast EDA onboarding
- self-contained presentation outputs
- clear same-day target narrative
- notebook generation for review
- concise exploratory reporting

If the task is:

- explain the dataset
- produce EDA figures
- create presentation material

then `other_eda_preprocess` is a good branch.

---

## Strengths of the Current Repository

The current repository is stronger when the task is:

- trace provenance from raw data
- build forecasting targets
- control leakage more explicitly
- separate deterministic processing from fit-based preprocessing
- train actual models on curated exports

That is why the current branch is more appropriate for machine learning.

---

## Bottom Line

`other_eda_preprocess` is best understood as:

- a six-script EDA and presentation workflow built around
  `dataset_final.csv`

The current repository is best understood as:

- a broader engineering pipeline that starts earlier, cleans more aggressively,
  supports no-leak exports, and continues into training

So the two repos are not direct substitutes:

- `other_eda_preprocess` is stronger as an EDA package
- the current repository is stronger as a forecasting / ML pipeline
