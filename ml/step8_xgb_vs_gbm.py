"""
STEP 8: XGBoost vs GBM - extensive tuning
Usage: python ml/step8_xgb_vs_gbm.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE as RS, DATA_PATH, SPLIT_DATE, OUT_DIR, get_tscv
from step4_improve import add_technical_features

P = '=' * 90


def main():
    print(f'\n{P}\n STEP 8: XGBoost vs GBM — EXTENSIVE TUNING\n{P}')

    # Load + technicals + TOP_50
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    exclude = {'date', 'oil_return', 'oil_close'}
    all_features = [c for c in df.columns if c not in exclude]

    train_mask = df['date'] < SPLIT_DATE
    test_mask = df['date'] >= SPLIT_DATE

    X_train_full = df.loc[train_mask, all_features]
    y_train = (df.loc[train_mask, 'oil_return'] > 0).astype(int)
    X_test_full = df.loc[test_mask, all_features]
    y_test = (df.loc[test_mask, 'oil_return'] > 0).astype(int)

    # TOP_50 selection
    mi = mutual_info_classif(X_train_full.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train_full.corrwith(df.loc[train_mask, 'oil_return'], method='spearman').abs()
    rank = pd.DataFrame({'feature': all_features, 'MI': mi, 'abs_sp': sp.values})
    for c in ['MI', 'abs_sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2
    rank.sort_values('score', ascending=False, inplace=True)
    top50 = rank.head(50)['feature'].tolist()

    X_train = X_train_full[top50]
    X_test = X_test_full[top50]
    tscv = get_tscv()

    print(f'  Features: {len(top50)} | Train: {len(X_train)} | Test: {len(X_test)}')

    # ═══════════════════════════════════════
    # XGBoost — WIDE GRID
    # ═══════════════════════════════════════
    print(f'\n{P}\n XGBoost — 50 iterations RandomizedSearch\n{P}')

    xgb_grid = {
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
    }

    t0 = time.time()
    xgb_gs = RandomizedSearchCV(
        XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
        xgb_grid, n_iter=50, cv=tscv, scoring='accuracy',
        refit=True, n_jobs=1, random_state=RS
    )
    xgb_gs.fit(X_train, y_train)
    xgb_time = round(time.time() - t0, 1)

    xgb_pred = xgb_gs.best_estimator_.predict(X_test)
    xgb_prob = xgb_gs.best_estimator_.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='macro')
    xgb_auc = roc_auc_score(y_test, xgb_prob)

    print(f'  Best params: {xgb_gs.best_params_}')
    print(f'  CV Acc:  {xgb_gs.best_score_:.4f}')
    print(f'  Test:    Acc={xgb_acc:.4f} F1m={xgb_f1:.4f} AUC={xgb_auc:.4f}')
    print(f'  Time:    {xgb_time}s')

    # Top 10 CV results
    cv_results = pd.DataFrame(xgb_gs.cv_results_)
    cv_results.sort_values('rank_test_score', inplace=True)
    print(f'\n  Top 10 XGB configs:')
    print(f'  {"Rank":<6} {"CV_Acc":>8} {"Std":>8}')
    for _, r in cv_results.head(10).iterrows():
        print(f'  {int(r["rank_test_score"]):<6} {r["mean_test_score"]:>8.4f} {r["std_test_score"]:>8.4f}')

    # ═══════════════════════════════════════
    # GBM — WIDE GRID
    # ═══════════════════════════════════════
    print(f'\n{P}\n GBM (sklearn) — 50 iterations RandomizedSearch\n{P}')

    gbm_grid = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'min_samples_leaf': [1, 3, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9, None],
    }

    t0 = time.time()
    gbm_gs = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RS),
        gbm_grid, n_iter=50, cv=tscv, scoring='accuracy',
        refit=True, n_jobs=1, random_state=RS
    )
    gbm_gs.fit(X_train, y_train)
    gbm_time = round(time.time() - t0, 1)

    gbm_pred = gbm_gs.best_estimator_.predict(X_test)
    gbm_prob = gbm_gs.best_estimator_.predict_proba(X_test)[:, 1]
    gbm_acc = accuracy_score(y_test, gbm_pred)
    gbm_f1 = f1_score(y_test, gbm_pred, average='macro')
    gbm_auc = roc_auc_score(y_test, gbm_prob)

    print(f'  Best params: {gbm_gs.best_params_}')
    print(f'  CV Acc:  {gbm_gs.best_score_:.4f}')
    print(f'  Test:    Acc={gbm_acc:.4f} F1m={gbm_f1:.4f} AUC={gbm_auc:.4f}')
    print(f'  Time:    {gbm_time}s')

    cv_results2 = pd.DataFrame(gbm_gs.cv_results_)
    cv_results2.sort_values('rank_test_score', inplace=True)
    print(f'\n  Top 10 GBM configs:')
    print(f'  {"Rank":<6} {"CV_Acc":>8} {"Std":>8}')
    for _, r in cv_results2.head(10).iterrows():
        print(f'  {int(r["rank_test_score"]):<6} {r["mean_test_score"]:>8.4f} {r["std_test_score"]:>8.4f}')

    # ═══════════════════════════════════════
    # LightGBM — WIDE GRID (for reference)
    # ═══════════════════════════════════════
    print(f'\n{P}\n LightGBM — 50 iterations RandomizedSearch\n{P}')

    lgbm_grid = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, -1],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'num_leaves': [7, 15, 20, 31, 50, 63],
        'min_child_samples': [1, 3, 5, 10, 20, 30],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0, 5.0],
    }

    t0 = time.time()
    lgbm_gs = RandomizedSearchCV(
        LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
        lgbm_grid, n_iter=50, cv=tscv, scoring='accuracy',
        refit=True, n_jobs=1, random_state=RS
    )
    lgbm_gs.fit(X_train, y_train)
    lgbm_time = round(time.time() - t0, 1)

    lgbm_pred = lgbm_gs.best_estimator_.predict(X_test)
    lgbm_prob = lgbm_gs.best_estimator_.predict_proba(X_test)[:, 1]
    lgbm_acc = accuracy_score(y_test, lgbm_pred)
    lgbm_f1 = f1_score(y_test, lgbm_pred, average='macro')
    lgbm_auc = roc_auc_score(y_test, lgbm_prob)

    print(f'  Best params: {lgbm_gs.best_params_}')
    print(f'  CV Acc:  {lgbm_gs.best_score_:.4f}')
    print(f'  Test:    Acc={lgbm_acc:.4f} F1m={lgbm_f1:.4f} AUC={lgbm_auc:.4f}')
    print(f'  Time:    {lgbm_time}s')

    # ═══════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════
    print(f'\n{P}\n FINAL COMPARISON\n{P}')
    print(f'\n {"Model":<10} {"CV_Acc":>8} {"Test_Acc":>10} {"F1_macro":>10} {"AUC":>8} {"Time":>8}')
    print(f' {"-"*58}')
    print(f' {"XGBoost":<10} {xgb_gs.best_score_:>8.4f} {xgb_acc:>10.4f} {xgb_f1:>10.4f} {xgb_auc:>8.4f} {xgb_time:>7.0f}s')
    print(f' {"GBM":<10} {gbm_gs.best_score_:>8.4f} {gbm_acc:>10.4f} {gbm_f1:>10.4f} {gbm_auc:>8.4f} {gbm_time:>7.0f}s')
    print(f' {"LightGBM":<10} {lgbm_gs.best_score_:>8.4f} {lgbm_acc:>10.4f} {lgbm_f1:>10.4f} {lgbm_auc:>8.4f} {lgbm_time:>7.0f}s')

    winner = max([('XGBoost', xgb_acc), ('GBM', gbm_acc), ('LightGBM', lgbm_acc)], key=lambda x: x[1])
    print(f'\n Winner: {winner[0]} (Acc={winner[1]:.4f})')

    # Save
    pd.DataFrame([
        {'Model': 'XGBoost', 'CV_Acc': xgb_gs.best_score_, 'Test_Acc': xgb_acc, 'F1m': xgb_f1, 'AUC': xgb_auc,
         'Params': str(xgb_gs.best_params_)},
        {'Model': 'GBM', 'CV_Acc': gbm_gs.best_score_, 'Test_Acc': gbm_acc, 'F1m': gbm_f1, 'AUC': gbm_auc,
         'Params': str(gbm_gs.best_params_)},
        {'Model': 'LightGBM', 'CV_Acc': lgbm_gs.best_score_, 'Test_Acc': lgbm_acc, 'F1m': lgbm_f1, 'AUC': lgbm_auc,
         'Params': str(lgbm_gs.best_params_)},
    ]).to_csv(os.path.join(OUT_DIR, 'step8_results.csv'), index=False)

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
