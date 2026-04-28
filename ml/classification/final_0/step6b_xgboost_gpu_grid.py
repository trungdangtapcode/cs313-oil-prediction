"""
STEP 6B: XGBoost GPU Ultra-Deep Grid Search

This step is a dedicated XGBoost-only search designed for GPU execution.
Instead of splitting the search budget across multiple tree models, it
focuses on a deeper and more XGBoost-specific exploration.

Goal of this step:
  - Use GPU-accelerated XGBoost as the only learner
  - Search multiple XGBoost families: gbtree/depthwise, gbtree/lossguide, and dart
  - Exploit XGBoost-specific knobs such as grow_policy, max_leaves, rate_drop,
    skip_drop, sampling_method, max_bin, and scale_pos_weight
  - Select the best configuration by CV accuracy, then evaluate once on test

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base features from the process pipeline
  - Technical features added by the shared helper
    ml/classification/step3_technical_improve.py
  - The same technical-feature pool convention used by
    final/step3_select_and_train.py, final/step4_group_selection.py,
    final/step5_weight_decay.py, and final/step6_xgb_vs_gbm.py
  - A TOP_50 subset is rebuilt inside this script using MI + Spearman ranking

Search design:
  - GridSearchCV only
  - TimeSeriesSplit for time-aware cross-validation
  - Multiple focused sub-grids to search XGBoost deeply without mixing invalid
    parameter combinations across booster families

Outputs:
  - results/step6b_feature_ranking.csv
  - results/step6b_search_manifest.csv
  - results/step6b_cv_results.csv
  - results/step6b_top_configs.csv
  - results/step6b_test_results.csv
  - results/step6b_results.csv

Usage:
  python ml/classification/final/step6b_xgboost_gpu_grid.py
"""
import os
import sys
import time

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
CLASSIFICATION_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
ML_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, CLASSIFICATION_DIR)
sys.path.insert(0, ML_DIR)

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    OUT_DIR,
    RANDOM_STATE as RS,
    TARGET,
    TARGET_DATE_COL,
    get_tscv,
    get_train_test_masks,
    set_global_seed,
)
# NOTE:
# `ml/classification/final/` does not have its own `step3_technical_improve.py`.
# The final-step scripts reuse the shared helper from the parent
# `ml/classification/` directory, which is why `CLASSIFICATION_DIR` is added
# to `sys.path` above.
from step3_technical_improve import add_technical_features

P = '=' * 90
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))
STEP6B_VERBOSE = max(0, int(os.getenv('STEP6B_VERBOSE', '1')))
STEP6B_EXPORT_TOPN = max(1, int(os.getenv('STEP6B_EXPORT_TOPN', '50')))
STEP6B_SCORING = os.getenv('STEP6B_SCORING', 'accuracy')
STEP6B_DEVICE = os.getenv('STEP6B_XGB_DEVICE', 'cuda')
STEP6B_LEGACY_GPU = os.getenv('STEP6B_XGB_LEGACY_GPU', '0') == '1'


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        'Accuracy': accuracy_score(y, pred),
        'F1_macro': f1_score(y, pred, average='macro'),
        'AUC': roc_auc_score(y, prob),
    }


def log(message=''):
    print(message, flush=True)


def build_rankings(X_train, y_train, features):
    mi = mutual_info_classif(X_train.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train.corrwith(y_train, method='spearman').abs()

    rank = pd.DataFrame({'feature': features, 'MI': mi, 'abs_sp': sp.values})
    for col in ['MI', 'abs_sp']:
        mx = rank[col].max()
        rank[f'{col}_n'] = rank[col] / mx if mx > 0 else 0
    rank['mix_score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2
    rank.sort_values(['mix_score', 'MI', 'abs_sp'], ascending=False, inplace=True)
    rank.reset_index(drop=True, inplace=True)
    return rank


def build_scale_pos_weight_grid(y_train):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    ratio = neg / max(pos, 1)
    candidates = {
        round(max(0.5, ratio * 0.85), 3),
        round(ratio, 3),
        round(max(0.75, ratio * 1.15), 3),
    }
    return sorted(candidates)


def build_xgb_base_estimator():
    base = {
        'random_state': RS,
        'verbosity': 0,
        'n_jobs': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'enable_categorical': False,
    }

    # XGBoost >= 2.0 prefers device='cuda' + tree_method='hist'.
    # If the training environment uses an older XGBoost build, set:
    #   STEP6B_XGB_LEGACY_GPU=1
    # and the script will switch to tree_method='gpu_hist'.
    if STEP6B_LEGACY_GPU:
        base.update({
            'tree_method': 'gpu_hist',
        })
    else:
        base.update({
            'tree_method': 'hist',
            'device': STEP6B_DEVICE,
        })

    return XGBClassifier(**base)


def build_grid_specs(scale_pos_weight_grid):
    return [
        (
            'gbtree_depthwise_core',
            {
                'booster': ['gbtree'],
                'grow_policy': ['depthwise'],
                'n_estimators': [300, 600, 900],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.02, 0.05, 0.08],
                'min_child_weight': [1, 4, 8],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.6, 0.85, 1.0],
                'gamma': [0.0, 0.2, 0.6],
                'sampling_method': ['uniform', 'gradient_based'],
                'scale_pos_weight': scale_pos_weight_grid,
            },
        ),
        (
            'gbtree_regularized_cuda',
            {
                'booster': ['gbtree'],
                'grow_policy': ['depthwise'],
                'n_estimators': [600],
                'max_depth': [4, 6],
                'learning_rate': [0.03, 0.06],
                'min_child_weight': [1, 5],
                'subsample': [0.85],
                'colsample_bytree': [0.85],
                'colsample_bynode': [0.7, 1.0],
                'reg_alpha': [0.0, 0.01, 0.1, 0.5],
                'reg_lambda': [1.0, 3.0, 8.0],
                'max_delta_step': [0, 2],
                'scale_pos_weight': scale_pos_weight_grid,
                'max_bin': [256, 512],
            },
        ),
        (
            'gbtree_lossguide_cuda',
            {
                'booster': ['gbtree'],
                'grow_policy': ['lossguide'],
                'max_depth': [0],
                'max_leaves': [31, 63, 127, 255],
                'n_estimators': [300, 600, 900],
                'learning_rate': [0.02, 0.05, 0.08],
                'min_child_weight': [1, 4, 8],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.6, 0.85, 1.0],
                'gamma': [0.0, 0.2],
                'reg_alpha': [0.0, 0.1],
                'reg_lambda': [1.0, 4.0],
                'max_bin': [256, 512],
            },
        ),
        (
            'dart_depthwise_cuda',
            {
                'booster': ['dart'],
                'grow_policy': ['depthwise'],
                'n_estimators': [300, 600, 900],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.02, 0.05, 0.08],
                'min_child_weight': [1, 5],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.7, 1.0],
                'gamma': [0.0, 0.2],
                'rate_drop': [0.0, 0.05, 0.1],
                'skip_drop': [0.0, 0.1],
                'normalize_type': ['tree', 'forest'],
                'sample_type': ['uniform', 'weighted'],
            },
        ),
    ]


def main():
    seed = set_global_seed()
    log(f'\n{P}\n STEP 6B: XGBOOST GPU ULTRA-DEEP GRID SEARCH\n{P}')
    log(f'  Seed: {seed}')
    log(f'  Parallelism: search_jobs={SEARCH_N_JOBS}')
    log(f'  Scoring: {STEP6B_SCORING}')
    log(f'  XGBoost GPU mode: {"legacy gpu_hist" if STEP6B_LEGACY_GPU else f"device={STEP6B_DEVICE}, tree_method=hist"}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    all_features = [c for c in df.columns if c not in exclude]

    train_mask, test_mask, _ = get_train_test_masks(df)

    X_train_full = df.loc[train_mask, all_features]
    X_test_full = df.loc[test_mask, all_features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    rank = build_rankings(X_train_full, y_train, all_features)
    rank.to_csv(os.path.join(OUT, 'step6b_feature_ranking.csv'), index=False)
    top50 = rank.head(50)['feature'].tolist()

    X_train = X_train_full[top50]
    X_test = X_test_full[top50]
    tscv = get_tscv()

    scale_pos_weight_grid = build_scale_pos_weight_grid(y_train)
    grid_specs = build_grid_specs(scale_pos_weight_grid)

    manifest_rows = []
    for grid_name, grid in grid_specs:
        manifest_rows.append({
            'Grid': grid_name,
            'Combinations': len(list(ParameterGrid(grid))),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.loc[len(manifest_df)] = {
        'Grid': 'TOTAL',
        'Combinations': int(manifest_df['Combinations'].sum()),
    }
    manifest_df.to_csv(os.path.join(OUT, 'step6b_search_manifest.csv'), index=False)
    progress_rows = [{
        'Grid': row['Grid'],
        'Combinations': row['Combinations'],
        'Status': 'pending',
        'Best_CV_Acc': np.nan,
        'SearchTime_s': np.nan,
    } for row in manifest_rows]
    progress_path = os.path.join(OUT, 'step6b_progress.csv')
    pd.DataFrame(progress_rows).to_csv(progress_path, index=False)

    log(f'  Features selected: {len(top50)} (TOP_50 by MI + Spearman)')
    log(f'  Train: {len(X_train)} | Test: {len(X_test)}')
    log(f'  scale_pos_weight grid: {scale_pos_weight_grid}')
    log(f'\n Grid manifest:')
    log(manifest_df.to_string(index=False))

    base_estimator = build_xgb_base_estimator()

    all_cv_rows = []
    best_overall = None
    best_overall_score = -np.inf

    for grid_name, param_grid in grid_specs:
        n_comb = len(list(ParameterGrid(param_grid)))
        progress_df = pd.read_csv(progress_path)
        progress_df.loc[progress_df['Grid'] == grid_name, 'Status'] = 'running'
        progress_df.to_csv(progress_path, index=False)

        log(f'\n{P}\n {grid_name} | full GridSearchCV | {n_comb:,} combinations\n{P}')
        t0 = time.time()

        gs = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=tscv,
            scoring=STEP6B_SCORING,
            refit=True,
            n_jobs=SEARCH_N_JOBS,
            verbose=max(3, STEP6B_VERBOSE),
            return_train_score=False,
            error_score=np.nan,
        )
        gs.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)

        cv_results = pd.DataFrame(gs.cv_results_).sort_values(
            ['mean_test_score', 'std_test_score'],
            ascending=[False, True],
        )
        cv_results['Grid'] = grid_name
        cv_results['SearchTime_s'] = elapsed
        all_cv_rows.append(cv_results)
        cv_results.to_csv(os.path.join(OUT, f'step6b_{grid_name}_cv_results.csv'), index=False)

        if pd.notna(gs.best_score_) and gs.best_score_ > best_overall_score:
            best_overall_score = gs.best_score_
            best_overall = {
                'Grid': grid_name,
                'Estimator': gs.best_estimator_,
                'Params': gs.best_params_,
                'CV_Acc': gs.best_score_,
                'SearchTime_s': elapsed,
                'Combinations': n_comb,
            }

        progress_df = pd.read_csv(progress_path)
        progress_df.loc[progress_df['Grid'] == grid_name, 'Status'] = 'done'
        progress_df.loc[progress_df['Grid'] == grid_name, 'Best_CV_Acc'] = gs.best_score_
        progress_df.loc[progress_df['Grid'] == grid_name, 'SearchTime_s'] = elapsed
        progress_df.to_csv(progress_path, index=False)

        log(f'  Best params: {gs.best_params_}')
        log(f'  Best CV score ({STEP6B_SCORING}): {gs.best_score_:.4f}')
        log(f'  Search time: {elapsed}s')

        log(f'\n  Top 10 configs from {grid_name}:')
        log(f'  {"Rank":<6} {"CV_Acc":>8} {"Std":>8}')
        top_view = cv_results.head(10)
        for i, (_, row) in enumerate(top_view.iterrows(), start=1):
            log(f'  {i:<6} {row["mean_test_score"]:>8.4f} {row["std_test_score"]:>8.4f}')

    if best_overall is None:
        raise RuntimeError('No valid XGBoost grid search result was produced.')

    cv_df = pd.concat(all_cv_rows, ignore_index=True)
    cv_df.sort_values(['mean_test_score', 'std_test_score'], ascending=[False, True], inplace=True)
    cv_df.to_csv(os.path.join(OUT, 'step6b_cv_results.csv'), index=False)

    top_cols = [
        'Grid',
        'mean_test_score',
        'std_test_score',
        'rank_test_score',
        'mean_fit_time',
        'mean_score_time',
        'params',
    ]
    export_cols = [c for c in top_cols if c in cv_df.columns]
    cv_df.head(STEP6B_EXPORT_TOPN)[export_cols].to_csv(
        os.path.join(OUT, 'step6b_top_configs.csv'),
        index=False,
    )

    test_metrics = evaluate(best_overall['Estimator'], X_test, y_test)

    test_row = {
        'Model': 'XGBoost',
        'Grid': best_overall['Grid'],
        'Selected_On': STEP6B_SCORING,
        'Feature_Set': 'TOP_50_MI_SPEARMAN',
        'CV_Acc': best_overall['CV_Acc'],
        'Test_Accuracy': test_metrics['Accuracy'],
        'Test_F1_macro': test_metrics['F1_macro'],
        'Test_AUC': test_metrics['AUC'],
        'SearchTime_s': best_overall['SearchTime_s'],
        'Combinations': best_overall['Combinations'],
        'Params': str(best_overall['Params']),
        'GPU_Mode': 'gpu_hist' if STEP6B_LEGACY_GPU else f'device={STEP6B_DEVICE},tree_method=hist',
    }

    test_df = pd.DataFrame([test_row])
    test_df.to_csv(os.path.join(OUT, 'step6b_test_results.csv'), index=False)
    test_df.to_csv(os.path.join(OUT, 'step6b_results.csv'), index=False)

    log(f'\n{P}\n BEST OVERALL XGBOOST CONFIG\n{P}')
    log(f'  Grid:      {best_overall["Grid"]}')
    log(f'  CV Acc:    {best_overall["CV_Acc"]:.4f}')
    log(f'  Test Acc:  {test_metrics["Accuracy"]:.4f}')
    log(f'  Test F1m:  {test_metrics["F1_macro"]:.4f}')
    log(f'  Test AUC:  {test_metrics["AUC"]:.4f}')
    log(f'  Params:    {best_overall["Params"]}')

    log(f'\n Saved:')
    log(f'  - {os.path.join(OUT, "step6b_feature_ranking.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_search_manifest.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_progress.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_cv_results.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_top_configs.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_test_results.csv")}')
    log(f'  - {os.path.join(OUT, "step6b_results.csv")}')
    log(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
