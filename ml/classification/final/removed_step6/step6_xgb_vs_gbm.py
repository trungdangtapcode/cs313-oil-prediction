"""
STEP 7: XGBoost vs GBM Extensive Tuning

This step is a heavier tuning experiment focused on the strongest
tree-based models after technical features and feature selection are in place.

Goal of this step:
  - Compare whether wide hyperparameter search improves over simpler settings
  - Test XGBoost, GBM, and LightGBM under full hyperparameter grids
  - Select the winner on test under the current workflow

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base features plus no-leakage technical features from step 3
  - A TOP_50 subset is rebuilt inside this script using MI + Spearman ranking

Models trained in this file:
  - XGBClassifier
  - GradientBoostingClassifier
  - LGBMClassifier

Model selection:
  - GridSearchCV with full hyperparameter grids on the training window
  - TimeSeriesSplit for time-aware cross-validation
  - Final model choice happens on test under the current workflow

Outputs:
  - Console test comparison of the tuned models
  - results/step7_results.csv

Usage:
  python ml/classification/step7_xgb_vs_gbm.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DATA_PATH, DROP_COLS, OUT_DIR, RANDOM_STATE as RS, TARGET, TARGET_DATE_COL, get_tscv, get_train_test_masks, set_global_seed

from metrics import evaluate, get_scores, METRIC_COLS, SORT_COLS

P = '=' * 90
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))




def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 6: XGBoost vs GBM - FULL GRID SEARCH\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: search_jobs={SEARCH_N_JOBS}')
    print('  Search mode: full GridSearchCV')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    features = [c for c in df.columns if c not in DROP_COLS and c != TARGET]

    train_mask, test_mask, _ = get_train_test_masks(df)

    X_train = df.loc[train_mask, features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    X_test = df.loc[test_mask, features]
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)
    tscv = get_tscv()

    print(f'  Features: {len(features)} | Train: {len(X_train)} | Test: {len(X_test)}')

    search_specs = [
        (
            'XGBoost',
            XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
            {
                'n_estimators': [100, 200, 300, 500, 800],
                'max_depth': [2, 3, 4, 5, 6, 7, 8],
                'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                'min_child_weight': [1, 3, 5, 7, 10],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 3.0, 5.0],
                'gamma': [0, 0.01, 0.1, 0.5, 1.0],
                'scale_pos_weight': [0.9, 1.0, 1.1],
            },
        ),
        (
            'GBM',
            GradientBoostingClassifier(random_state=RS),
            {
                'n_estimators': [100, 200, 300, 500, 800],
                'max_depth': [2, 3, 4, 5, 6, 7, 8],
                'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                'min_samples_leaf': [1, 3, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9, None],
            },
        ),
        # (
        #     'LightGBM',
        #     LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
        #     {
        #         'n_estimators': [100, 200, 300, 500, 800],
        #         'max_depth': [2, 3, 4, 5, 6, 7, 8, -1],
        #         'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        #         'num_leaves': [7, 15, 20, 31, 50, 63],
        #         'min_child_samples': [1, 3, 5, 10, 20, 30],
        #         'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        #         'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #         'reg_alpha': [0, 0.01, 0.1, 1.0],
        #         'reg_lambda': [0, 0.1, 1.0, 5.0],
        #     },
        # ),
    ]

    val_rows = []
    for name, estimator, grid in search_specs:
        total_combinations = len(list(ParameterGrid(grid)))
        print(f'\n{P}\n {name} - Full GridSearch ({total_combinations:,} combinations)\n{P}')
        t0 = time.time()
        gs = GridSearchCV(
            estimator,
            grid,
            cv=tscv,
            scoring='accuracy',
            refit=True,
            n_jobs=SEARCH_N_JOBS,
        )
        gs.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)
        pred = gs.best_estimator_.predict(X_test)
        score = get_scores(gs.best_estimator_, X_test)
        val_metrics = evaluate(name, y_test.values, pred, score)
        val_rows.append({
            **val_metrics,
            'CV_Acc': gs.best_score_,
            'Time_s': elapsed,
            'Params': str(gs.best_params_),
            'Estimator': gs.best_estimator_,
        })

        print(f'  Best params: {gs.best_params_}')
        print(f'  CV Acc:  {gs.best_score_:.4f}')
        print(f'  Test:    Acc={val_metrics["Accuracy"]:.4f} '
            f'F1m={val_metrics["F1_macro"]:.4f} AUC={val_metrics["AUC"]:.4f}')
        print(f'  Time:    {elapsed}s')

        cv_results = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
        print(f'\n  Top 10 {name} configs:')
        print(f'  {"Rank":<6} {"CV_Acc":>8} {"Std":>8}')
        for _, row in cv_results.head(10).iterrows():
            print(f'  {int(row["rank_test_score"]):<6} {row["mean_test_score"]:>8.4f} {row["std_test_score"]:>8.4f}')

    vdf = pd.DataFrame(val_rows).sort_values(SORT_COLS, ascending=False)
    vdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT, 'step7_test_results.csv'), index=False)

    print(f'\n{P}\n TEST COMPARISON\n{P}')
    print(vdf[METRIC_COLS + ['CV_Acc', 'Time_s']].to_string(index=False))

    best = vdf.iloc[0]
    winner = best['Estimator']

    holdout_row = {
        'Model': best['Model'],
        'Selected_On': 'test',
        'CV_Acc': best['CV_Acc'],
        'Accuracy': best['Accuracy'],
        'F1_binary': best['F1_binary'],
        'F1_macro': best['F1_macro'],
        'AUC': best['AUC'],
        'Precision_UP': best['Precision_UP'],
        'Recall_UP': best['Recall_UP'],
        'TP': best['TP'],
        'FP': best['FP'],
        'TN': best['TN'],
        'FN': best['FN'],
        'Params': best['Params'],
    }
    pd.DataFrame([holdout_row]).to_csv(os.path.join(OUT, 'step7_results.csv'), index=False)

    print(f'\n Winner selected on test: {best["Model"]}')
    print(
        f' Test: Acc={best["Accuracy"]:.4f} '
        f'F1m={best["F1_macro"]:.4f} AUC={best["AUC"]:.4f}'
    )

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
