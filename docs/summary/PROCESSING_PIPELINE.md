# Comprehensive Data Processing Pipeline

This document meticulously details the end-to-end data processing pipeline for the Oil Price Prediction project. The pipeline encompasses everything from raw data ingestion to advanced feature engineering, rigorous mitigation of data leakage, and scaling formatting suitable for final model consumption.

---

## Step 1: Load and Inspect (`step1_load_inspect.py`)
- **Objective**: Retrieve raw data files (`market_data.csv`, `fred_data.csv`, `eia_data.csv`, `gdelt_data.csv`, `ACLED`) and execute preliminary sanity checks.
- **Key Operations**:
  - Validity checks for raw dataset shapes, essential columns, and data types constraints.
  - Identification and broad quantification of missing values prior to formalized imputation.

---

## Step 2: Data Cleaning (`step2_cleaning.py`)
- **Objective**: Address missing values, tackle noise and inconsistencies, and rectify specific reporting irregularities like the "Weekend Problem".
- **Key Operations**:
  - **GDELT Refinement**: Generates imputation markers (e.g., `gdelt_data_imputed`) and forward-fills missing observations (`limit=3` days) for core indicators: tone, goldstein scaler, volume, and event counts.
  - **ACLED Integration (Weekend Algorithm Fix)**: ACLED conflict occurrences reported over non-trading weekends are deterministically shifted to the subsequent financial trading day (typically Monday). Event quantities and fatalities are rigorously aggregated to align with the structured multi-market financial calendar.
  - **FRED Structuring**: Standardizes monthly macroeconomic indicators, verifying that prior publication lag structures (applied concurrently during raw crawling) are enforced safely (`fed_funds_rate_lag`, `cpi_lag`, `unemployment_lag`).
  - **Market Refinement**: Eliminates duplicated dates chronologically, strictly enforcing a unified temporal order, and applying a forward fill exclusively to `oil_close` to prevent `NaN`-induced calculation disruptions downstream.
  - Enforces mandatory Business Day isolation limits to rigorously scrub non-market-trading dates.

---

## Step 3: Data Integration (`step3_integration.py`)
- **Objective**: Synthesize disparate data sources into a harmonized cross-domain dataset strictly mapped to United States Financial Business Days (`Mon-Fri`).
- **Key Operations**:
  - Establishes a resolute `DatetimeIndex` scoped strictly by valid Business Days (`freq="B"`).
  - Merges datasets natively, running short-term chronological gap approximations by applying forward fill (`ffill`, `limit=3`) safely. Financial differentials/returns (`oil_return`, `usd_return`, `sp500_return`, `vix_return`) are expressly excluded from any artificial algorithmic imputation.
  - Implements redundancy tracking (e.g., verifying empirical covariance distributions between `wti_fred` against `oil_close` proxies).

---

## Step 4: Feature Transformation (`step4_transformation.py`)
- **Objective**: Execute domain-specific aggregations establishing predictive signals and complex attribute construction.
- **Key Operations**:
  - **Return Computations**: Generates daily Percentage Returns explicitly for target indices (`oil_return`, `usd_return`, `sp500_return`, `vix_return`).
  - **Temporal Aggregations (Rolling Windows)**: Computes dynamic time-series smoothing mechanisms (7-day, 30-day moving averages) explicitly against `gdelt_tone`, `gdelt_goldstein`, and constructs volatility tracking vectors (`oil_volatility_7d`).
  - **FRED Target Logic**: Integrates Year-over-Year (YoY) Inflation indicators (`cpi_yoy`), delineates True Real-Rate derivations (`fed_funds_rate_lag` - `cpi_yoy`), extracts rate pacing vectors (`fed_rate_change`), isolates recession signals (`yield_spread` < 0), and constructs `fed_rate_regime` categorically.
  - **EIA Macro Logic**: Transforms lagging petroleum constraints into percentage supply-chain differences (`production_change_pct`, `net_imports_change_pct` over 5-day windows). Formulates rolling normalized parameters (`inventory_zscore`).
  - **Multi-domain Global Stress Indexes**: Produces aggregated weighted benchmarks dynamically fusing `stress_tone`, `stress_volume`, and `stress_goldstein` attributes using variance scaling (`geopolitical_stress_index`).
  - **Chronological Features**: Engenders intrinsic memory through Lag features (t-1, t-2).

---

## Step 4B: Leakage Fix (`step4b_fix_leakage.py`)
- **Objective**: Curate conservative "No-Leak" pipeline derivatives dynamically dropping features highly prone to target contamination, split leakage, or timeline paradox biases.
- **Key Operations**:
  - **Release Timing Exclusions**: Aggressively strips monthly variables mapped excessively early into the daily continuum (`cpi_lag`, `unemployment_lag`, `fed_funds_rate_lag`).
  - **Derived Ancestry Cascades**: Strips parameters polluted downstream from release bias origins (`cpi_yoy`, `real_rate`, `fed_rate_change`, `fed_rate_regime`).
  - **Scaling Window Paradox Bias**: Eliminates indicators inherently poisoned via standard transformation mechanisms operating comprehensively against unified train-validation partitions simultaneously (`stress_tone`, `stress_volume`, `stress_goldstein`, `geopolitical_stress_index`, `oil_volatility_7d`).
  - Produces clean variants targeting subsequent model architectures securely devoid of timeline data leakage: `dataset_step4_noleak.csv`.

---

## Step 5: Data Reduction & Model Selection (`step5_reduction.py`)
- **Objective**: Synthesize a model-ready export pruning intermediate noise and extreme collinearity variables.
- **Key Operations**:
  - Prunes highly collinear or functionally intermediate values comprehensively mapped during preceding aggregation efforts.
  - Discards binary threshold triggers representing weak signal-to-noise ratios (`gdelt_tone_spike`, `recession_signal`).
  - Explicitly strips absolute pricing dimensions mapped structurally into respective fractional return paradigms (`vix_close`, `usd_close`, `sp500_close`, `oil_close`).
  - Formats canonical exports designed cleanly for inference usage architectures (`dataset_final.csv`).

---

## Step 5B: No-Leak Deterministic Processing (`step5b_processing.py`)
- **Objective**: Standardize feature distributions globally while maintaining stringent mathematical boundaries segregating valid train-time transformations. 
- **Key Operations**:
  - **Cyclical Calendric Algorithms**: Synthesizes periodic sine/cosine transformations directly against boundary date thresholds mapping continuous temporal shifts safely (`day_of_week`, `month`).
  - **Heavy-Tail Compressors**: Subjects positively skewed density paradigms rigidly through optimized natural logarithms (`log1p`)—focusing particularly on aggregated casualty distributions, volume vectors, and generalized categorical incidents.
  - **Bidirectional Logarithmic Smoothing**: Enforces `signed_log1p` processing cleanly mitigating severe skew anomalies simultaneously spanning both negative and positive variance planes against critical derivatives (`net_imports_change_pct`, `production_change_pct`, `vix_return`).
  - Deferral Strategy: Explicitly defers Train-Time distribution transformations (`StandardScaler`, `RobustScaler`, `PowerTransformer`) preventing future-information leakage, restricting parameter applications exclusively through explicit modular modeling blocks.

---

## Step 5C: Full-Data Scaling Initialization (`step5c_processing.py`) 
*(Research & Experimental Use-Case)*
- **Objective**: Produce fixed, structurally uniform, ready-transformed data matrixes specifically designated for static experimental EDA configurations or non-production back-testing.
- **Key Operations**:
  - Leverages feature sets rigorously derived from Step 5B principles.
  - Enacts cross-dataset overarching pipeline scaling operations utilizing predefined model schema groups:
    - **Standard Set**: `yield_spread`, `gdelt_goldstein`, `usd_return`, `inventory_zscore`.
    - **Robust Set**: `inventory_change_pct`, `oil_return`, `conflict` logistics, `net_imports_change_pct`. 
    - **Power / Yeo-Johnson Set**: `gdelt_tone_7d`, `gdelt_tone_30d`, `gdelt_tone_lag1`.
  - Produces `dataset_final_noleak_step5c.csv` alongside canonical scaler memory files (`dataset_final_noleak_step5c_preprocessor.joblib`).
  - *Disclaimer: Explicitly contraindicates production evaluation reliance strictly citing validation inclusion spanning full-frame normalization.*

---

## Note on Usage
In production modeling or canonical evaluation pipelines involving strict time-series cross validation (TSCV), it is required to ingest exclusively from **`dataset_final_noleak_processed.csv`** (Step 5B) coupling dynamic step-by-step Standard/Robust scaling exclusively executed internally isolated within each validation fold (e.g., configurations tracked natively via `ml/model_preprocessing.py`).