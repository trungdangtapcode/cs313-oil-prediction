# Classification Final Pipeline Map

This document thoroughly details the end-to-end execution pipeline for the final classification training and optimization workflow, located in `ml/classification/final/`. The pipeline progresses from establishing clean baseline models to highly advanced GPU-accelerated hyperparameter searches, feature clustering, and temporal weight decay.

---

## Step 1: Baseline Classification (`step1_train_baseline.py`)
- **Objective**: Establish a robust, clean baseline benchmark for predicting daily oil direction using exclusively the original feature set (without extra technical indicators or specialized feature selection).
- **Target**: `oil_return_fwd1 > 0` (UP=1, DOWN=0).
- **Key Operations**:
  - Ingests the base feature set from `dataset_step4_transformed.csv` (properly stripping known leakage parameters).
  - Trains a broad spectrum of fundamental classifiers: `LogisticRegression`, `SVC` (linear/rbf), `RandomForest`, `GradientBoosting`, `XGBClassifier`, `LGBMClassifier`, and `MLPClassifier`.
  - Executes isolated Hyperparameter Tuning via `GridSearchCV` inside the strictly defined training window utilizing a `TimeSeriesSplit` cross-validation strategy.
  - Final candidate models are validated dynamically on a fixed holdout final test window to ascertain baseline generalized accuracy, F1, and AUC.
  - **Outputs**: Baseline leaderboard, feature importance metrics, and comprehensive reports (`results/step1_test_results.csv`, `results/step1_feature_importance.csv`).

---

## Step 2: Fine-tune & Ensemble (`step2_finetune_ensemble.py`)
- **Objective**: Elevate the baseline performance by aggressively fine-tuning the leading classifiers and hybridizing them utilizing Ensemble Learning topologies.
- **Target**: Continuous binary direction.
- **Key Operations**:
  - Dynamically isolates a `TOP_25` feature subset computed via internal Mutual Information (MI) and Spearman rankings derived strictly from the training cohort.
  - Extensively tunes leading standalone algorithms: `XGBClassifier`, `GradientBoostingClassifier`, `LGBMClassifier`, and `SVC`.
  - Constructs meta-estimators natively mapping temporal boundaries:
    - **Soft Voting**: `VotingClassifier`.
    - **Stacking**: Configures a bespoke `TimeSeriesStackingClassifier`, engineered intentionally to preserve the chronological chronological causality strictly required for time-series fold stacking (`OOF` prediction matrices cross-validated via logical `TimeSeriesSplit`).
  - **Outputs**: Ensemble convergence metrics and chosen optimized models (`results/step2_test_results.csv`).

---

## Step 3: Feature Selection and Retraining (`step3_select_and_train.py`)
*(Internally labeled as Step 4 in file docstrings following technical injection)*
- **Objective**: Evaluate whether rigidly ranking a highly saturated 81-feature dataset (inclusive of specialized technical indicators) accurately escalates validation stability. 
- **Target**: Binary daily direction.
- **Key Operations**:
  - Imports expanded technical and lag features synthesized natively via `step3_technical_improve.add_technical_features()`.
  - Performs a Tri-Signal Feature Ranking framework assessing:
    1. Mutual Information (`MI`) alone.
    2. Spearman correlation (`|Sp|`) alone.
    3. Merged ensemble signal (`MI + Spearman`).
  - Systematically constructs 28 unique subset cardinality structures (ranging Top `10, 15... 70` against all three ranking schemes).
  - Uses `LGBMClassifier` as a high-speed validation proxy traversing those 28 subsets securely within `TimeSeriesSplit` cross-validation boundaries.
  - Top identified subset undergoes exhaustive randomized retraining schemas targeting `XGBClassifier`, `GradientBoosting`, and `LGBMClassifier`.
  - **Outputs**: Ranked signal parameters, subset benchmarks (`results/step4_feature_ranking.csv`, `results/step4_subset_comparison.csv`).

---

## Step 4: Smart Feature Selection (`step4_group_selection.py`)
*(Internally labeled as Step 5 in file docstrings)*
- **Objective**: Implement sophisticated multicollinearity reduction through topological hierarchical clustering rather than relying simply on naive top-N ranking logic.
- **Target**: Binary daily direction.
- **Key Operations**:
  - **Correlation Clustering**: Computes an absolute Spearman Distance Matrix (`1 - |rho|`) constructing hierarchical clusters utilizing `scipy`'s Average Linkage methodology.
  - Establishes a variance threshold (`t=0.3` distance) actively suppressing redundant indicators displaying high correlative densities (`|rho| > 0.7`).
  - **Permutation Importance**: Synthesizes empirical feature disruption impacts directly inside internal validation holdouts leveraging a baseline `LGBMClassifier` cross-validated across `STEP5_PERM_REPEATS`.
  - **Reconciliation**: Traverses all identified clusters, surgically extracting exclusively the single highest-performing feature (dictated by its isolated permutation impact score) while explicitly terminating its redundant cluster peers.
  - **Outputs**: Topographical structure manifests, dropped matrices, permutation summaries (`results/step5_perm_importance.csv`, `results/step5_selected_features.csv`).

---

## Step 5: Weight Decay Experiment (`step5_weight_decay.py`)
*(Internally labeled as Step 6/7 in file docstrings)*
- **Objective**: Investigate potential accuracy drift stemming from dynamic market regime shifts by structurally up-weighting chronologically recent training observations over historic samples.
- **Target**: Binary daily direction.
- **Key Operations**:
  - Dynamically synthesizes a `TOP_50` feature selection layer strictly bound within the training split map.
  - Systematizes diverse mathematical decay paradigms ensuring comprehensive normalization (`mean_weight = 1.0` per distribution logic):
    - `uniform`: Equal weight.
    - `exp_hl`: Exponential decays parameterized across specific half-lives (`100, 250, 500, 1000` days).
    - `linear`: Linearly degrading weighting scales (`0.1, 0.3, 0.5` minimums).
    - `step_pct`: Hard step boundaries tripling or doubling terminal training percentages enforcing pure recent regime dominance.
  - Fits temporal schemes internally to validation constraints targeting gradient models before cementing a formalized decay parameter to the isolated final target timeline.
  - **Outputs**: Visual degradation arrays (`results/step6_weight_schemes.png`), comparative validation matrixes.

---

## Step 6: XGB vs GBM Extensive Tuning (`step6_xgb_vs_gbm.py`)
*(Internally labeled as Step 7/8 in file docstrings)*
- **Objective**: Perform a massively scoped algorithmic tuning tournament expressly tailored toward ultra-optimized gradient boosting frameworks deployed on precisely restricted top-tier subsets.
- **Target**: Binary daily direction.
- **Key Operations**:
  - Identifies a rigorous `TOP_50` feature map via continuous `MI + Spearman` validations.
  - Defines excessively dense, combinatorial `ParameterGrid` search planes for `XGBClassifier` and `GradientBoostingClassifier` natively via `GridSearchCV`.
  - Searches extensive combinatorics involving structural depths (`2-8`), sampling limits, absolute alpha/lambda regularizations, explicit scale weight structures, and algorithmic learning thresholds.
  - Extracts absolute winner distributions exclusively matching performance against internal cross-validation bounds securely projected immediately onto terminal Test distributions.
  - **Outputs**: Broad exhaustive parameter manifests (`results/step7_test_results.csv`).

---

## Step 6B: XGBoost GPU Ultra-Deep Grid Search (`step6b_xgboost_gpu_grid.py`)
- **Objective**: Execute a structurally unconstrained hardware-accelerated grid search concentrating fundamentally on algorithmic depth strictly limited to the `XGBoost` engine ecosystem natively ported to local GPUs.
- **Target**: Binary daily direction.
- **Key Operations**:
  - Sets up native device routing directly engaging `device='cuda'` alongside hardware-centric structures (`tree_method='hist'`, legacy `gpu_hist`).
  - Eliminates logic crossovers by modularizing isolated search architectures natively tailored to discrete XGBoost frameworks ensuring zero illegal parameter mapping:
    - `gbtree_depthwise_core`: Standard dense depth limitations + gamma regularizations.
    - `gbtree_regularized_cuda`: Deep matrix regularization loops traversing massive bin architectures `(max_bin: 256, 512)`.
    - `gbtree_lossguide_cuda`: Node expansion natively controlled explicitly by threshold gradients (`max_depth=0`) dictating independent leaf development boundaries.
  - Continuously executes exhaustive cross-validation combinations, dynamically formulating `scale_pos_weight` ratios dynamically computed balancing local distribution anomalies natively.
  - **Outputs**: Ultimate deterministic hardware-accelerated configuration manifests bridging directly into extreme inference capability parameters (`results/step6b_results.csv`, `results/step6b_test_results.csv`).
