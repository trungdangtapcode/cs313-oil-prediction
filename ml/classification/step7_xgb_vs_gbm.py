"""
STEP 7: XGBoost vs GBM Extensive Tuning

This step is a heavier tuning experiment focused on the strongest
tree-based models after technical features and feature selection are in place.

Goal of this step:
  - Compare whether wide hyperparameter search improves over simpler settings
  - Test XGBoost, GBM, and LightGBM under larger search spaces
  - Select the winner on validation before evaluating the final holdout test

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
  - RandomizedSearchCV with large hyperparameter grids on the training window
  - TimeSeriesSplit for time-aware cross-validation
  - Final model choice happens on validation only
  - The chosen winner is refit on train+validation and evaluated once on holdout test

Outputs:
  - Console validation comparison of the tuned models
  - results/step7_validation_results.csv
  - results/step7_results.csv

Usage:
  python ml/classification/step7_xgb_vs_gbm.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE as RS, DATA_PATH, TARGET, TARGET_DATE_COL, get_tscv, get_train_val_test_masks, set_global_seed
from step3_technical_improve import add_technical_features

P = '=' * 90
OUT = os.path.join(os.path.dirname(__file__), 'results')


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        'Accuracy': accuracy_score(y, pred),
        'F1_macro': f1_score(y, pred, average='macro'),
        'AUC': roc_auc_score(y, prob),
    }


def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 8: XGBoost vs GBM - EXTENSIVE TUNING\n{P}')
    print(f'  Seed: {seed}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    all_features = [c for c in df.columns if c not in exclude]

    train_mask, val_mask, test_mask, _ = get_train_val_test_masks(df)

    X_train_full = df.loc[train_mask, all_features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    X_val_full = df.loc[val_mask, all_features]
    y_val = (df.loc[val_mask, TARGET] > 0).astype(int)
    X_test_full = df.loc[test_mask, all_features]
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    mi = mutual_info_classif(X_train_full.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train_full.corrwith(df.loc[train_mask, TARGET], method='spearman').abs()
    rank = pd.DataFrame({'feature': all_features, 'MI': mi, 'abs_sp': sp.values})
    for c in ['MI', 'abs_sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2
    rank.sort_values('score', ascending=False, inplace=True)
    top50 = rank.head(50)['feature'].tolist()

    X_train = X_train_full[top50]
    X_val = X_val_full[top50]
    X_test = X_test_full[top50]
    tscv = get_tscv()

    print(f'  Features: {len(top50)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')

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
        (
            'LightGBM',
            LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
            {
                'n_estimators': [100, 200, 300, 500, 800],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, -1],
                'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                'num_leaves': [7, 15, 20, 31, 50, 63],
                'min_child_samples': [1, 3, 5, 10, 20, 30],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.01, 0.1, 1.0],
                'reg_lambda': [0, 0.1, 1.0, 5.0],
            },
        ),
    ]

    val_rows = []
    for name, estimator, grid in search_specs:
        print(f'\n{P}\n {name} - 50 iterations RandomizedSearch\n{P}')
        t0 = time.time()
        gs = RandomizedSearchCV(
            estimator,
            grid,
            n_iter=50,
            cv=tscv,
            scoring='accuracy',
            refit=True,
            n_jobs=1,
            random_state=RS,
        )
        gs.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)
        val_metrics = evaluate(gs.best_estimator_, X_val, y_val)
        val_rows.append({
            'Model': name,
            'CV_Acc': gs.best_score_,
            'Val_Accuracy': val_metrics['Accuracy'],
            'Val_F1_macro': val_metrics['F1_macro'],
            'Val_AUC': val_metrics['AUC'],
            'Time_s': elapsed,
            'Params': str(gs.best_params_),
            'Estimator': clone(gs.best_estimator_),
        })

        print(f'  Best params: {gs.best_params_}')
        print(f'  CV Acc:  {gs.best_score_:.4f}')
        print(
            f'  Val:     Acc={val_metrics["Accuracy"]:.4f} '
            f'F1m={val_metrics["F1_macro"]:.4f} AUC={val_metrics["AUC"]:.4f}'
        )
        print(f'  Time:    {elapsed}s')

        cv_results = pd.DataFrame(gs.cv_results_).sort_values('rank_test_score')
        print(f'\n  Top 10 {name} configs:')
        print(f'  {"Rank":<6} {"CV_Acc":>8} {"Std":>8}')
        for _, row in cv_results.head(10).iterrows():
            print(f'  {int(row["rank_test_score"]):<6} {row["mean_test_score"]:>8.4f} {row["std_test_score"]:>8.4f}')

    vdf = pd.DataFrame(val_rows).sort_values(['Val_Accuracy', 'Val_F1_macro', 'Val_AUC'], ascending=False)
    vdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT, 'step7_validation_results.csv'), index=False)

    print(f'\n{P}\n VALIDATION COMPARISON\n{P}')
    print(vdf[['Model', 'CV_Acc', 'Val_Accuracy', 'Val_F1_macro', 'Val_AUC', 'Time_s']].to_string(index=False))

    best = vdf.iloc[0]
    winner = clone(best['Estimator'])
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    winner.fit(X_train_val, y_train_val)
    holdout = evaluate(winner, X_test, y_test)

    holdout_row = {
        'Model': best['Model'],
        'Selected_On': 'validation',
        'CV_Acc': best['CV_Acc'],
        'Val_Accuracy': best['Val_Accuracy'],
        'Val_F1_macro': best['Val_F1_macro'],
        'Val_AUC': best['Val_AUC'],
        'Holdout_Accuracy': holdout['Accuracy'],
        'Holdout_F1_macro': holdout['F1_macro'],
        'Holdout_AUC': holdout['AUC'],
        'Params': best['Params'],
    }
    pd.DataFrame([holdout_row]).to_csv(os.path.join(OUT, 'step7_results.csv'), index=False)

    print(f'\n Winner selected on validation: {best["Model"]}')
    print(
        f' Validation: Acc={best["Val_Accuracy"]:.4f} '
        f'F1m={best["Val_F1_macro"]:.4f} AUC={best["Val_AUC"]:.4f}'
    )
    print(
        f' Holdout:    Acc={holdout["Accuracy"]:.4f} '
        f'F1m={holdout["F1_macro"]:.4f} AUC={holdout["AUC"]:.4f}'
    )

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
