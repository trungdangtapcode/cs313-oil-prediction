# Classification EDA Pipeline Map

This document thoroughly details the Exploratory Data Analysis (EDA) pipeline operating strictly upon the binary classification objective (predicting daily oil price direction: UP vs DOWN). The environment primarily orbits the `eda_clf.py` generator, utilized to run extensive statistical evaluations against various stages of processed pipeline datasets. 

---

## Core Engine: `eda_clf.py`
- **Objective**: Execute a rigorous Data Quality, Feature Significance, and Validation Stability Assessment. 
- **Target Variable**: Continuous leading values natively partitioned into strict binary paradigms (`oil_return_fwd1 > 0` → `UP`=1, `DOWN`=0).
- **Key Analytical Operations**:
  - **Data Quality & Safety Protocol**: Scans global outlier ratios (via IQR limits), records explicit null occurrences, computes strict timeline shifts (Train vs. Test Kolmogorov-Smirnov), and enforces a documented leakage safety flag protocol detecting timing risks.
  - **Target Class & Timeline Exploration**: Maps base class constraints bounding model metrics via specific distributions:
    - Overall Train/Test Class Split Imbalances (`01_class_distribution.png`).
    - Temporal breakdown delineating up/down ratios strictly per calendar year (`02_class_by_year.png`).
    - Moving average proportions identifying streak-length structures and market density persistence (`03_target_over_time.png` / `08_class_over_time.png`).
  - **Feature Discriminative Power Analysis**: Isolates empirical structural variation between the two price trajectory distributions utilizing:
    - *Kolmogorov-Smirnov (KS)* & *Mann-Whitney U (MWU)* testing bounding univariant distribution overlaps (`03_dist_by_class_all.png`).
    - *Cohen's d* tracking absolute normalized effect-size differences between classification boundaries (`04_cohens_d.png`).
  - **Probabilistic Signal Estimation**: Discovers explicit ranking correlation metrics bounding model performance boundaries:
    - *Point-Biserial Correlations* tracking dense linear continuous/binary constraints (`05_pointbiserial.png`).
    - *Mutual Information (MI)* evaluating robust non-linear boundaries natively separated from monotonic correlation constraints (`06_mutual_info.png`).
  - **Topological Visuals**: Plots distribution density maps comparing raw distributions for top features (`09_violin_top12.png`), dense quantile bounding (`07_boxplot_top20.png`), and hierarchical feature pair isolation (`11_scatter_pairs.png`).
  - **Compound Feature Ranking Export**: Formulates the master `feature_ranking_clf.csv` index natively grouping features across a `research_score` dynamically calculating an optimization ratio bridging observed target signals versus explicit longitudinal stability (decay bounds).

---

## Segregated Sub-Executions
The `eda_classification/` directory operates specifically structured sub-directories mapping directly across numerous pipeline feature permutation models. The primary workflow systematically runs the `eda_clf.py` suite against these derived outputs generating discrete visualization hubs:

- **`drop_macro/`**
  - **Focus**: Evaluates feature densities completely stripping external Federal Reserve Macroeconomic (FRED) series data to analyze baseline dataset integrity safely removed from external fundamental reliance and rigid publication delay variables.

- **`step5_preprocess/`**
  - **Focus**: Computes baseline classification densities utilizing the pre-processed unscaled `dataset_final.csv` variants extracted immediately following intermediate dropping rules formalized in string-oriented `step5`.

- **`step5b_processing/`**
  - **Focus**: Runs rigorous signal mappings capturing deterministic feature permutations. It plots evaluations across cyclical continuous bounds (Sine/Cosine vectors) and bounded symmetric modifications (`log1p` / `signed_log1p`) assuring downstream signal densities persist uninterrupted post-transformation.

- **`step5_upgraded/`**
  - **Focus**: Reflects a modified intermediary structural evaluation testing topological feature additions strictly targeting multi-variable variance before funneling data frames further toward the rigorous training logic map architectures. 

---

## Notable File Generations (Outputs)
Within the root and structured sub-directories, executing the EDA suite routinely produces (but is not isolated strictly to):
  - `eda_log.txt`: Standard out recording structural console discoveries.
  - `feature_ranking_clf.csv` / `feature_ranking_clf_full.csv`: Tabular rankings combining scoring formulas driving terminal feature subset configurations downstream.
  - **PNG Suite**: Canonical correlation heatmaps (`10_corr_by_class.png`, `10_corr_top15.png`), multicollinearity tracking constraints (`06c_multicollinearity.png`), and dynamic structural seasonality heatmaps (`06d_calendar_patterns.png`).