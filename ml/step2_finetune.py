"""
STEP 2: Fine-tune top models (regression + classification)
Usage: python ml/step2_finetune.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from config import load_data, get_tscv, OUT_DIR, RANDOM_STATE as RS

P = '=' * 90


def main():
    print(f'\n{P}\n STEP 2: FINE-TUNE TOP MODELS\n{P}')
    data = load_data()
    tscv = get_tscv()

    # Load best subsets from step1
    # Read subset_comparison.csv to pick best
    comp = pd.read_csv(os.path.join(OUT_DIR, 'subset_comparison.csv'))
    best_reg_name = comp.loc[comp['RMSE'].idxmin(), 'Subset'].split('(')[0].strip()
    best_cls_name = comp.loc[comp['F1m'].idxmax(), 'Subset'].split('(')[0].strip()
    print(f'  Best reg subset: {best_reg_name} | Best cls subset: {best_cls_name}')

    reg_feats_file = os.path.join(OUT_DIR, f'subset_{best_reg_name}.csv')
    cls_feats_file = os.path.join(OUT_DIR, f'subset_{best_cls_name}.csv')

    if os.path.exists(reg_feats_file):
        reg_feats = pd.read_csv(reg_feats_file, header=None)[0].tolist()
    else:
        reg_feats = data['features']  # fallback

    if os.path.exists(cls_feats_file):
        cls_feats = pd.read_csv(cls_feats_file, header=None)[0].tolist()
    else:
        cls_feats = data['features']

    # ═══════════════════════════════════════════════════════
    # REGRESSION FINE-TUNE (TOP_10)
    # ═══════════════════════════════════════════════════════
    print(f'\n{P}\n 2A: REGRESSION ({len(reg_feats)} features)\n{P}')
    print(f'  Features: {reg_feats}')

    X_tr = data['X_train'][reg_feats]
    X_te = data['X_test'][reg_feats]
    y_tr, y_te = data['y_train'], data['y_test']

    reg_models = {
        'LightGBM_v2': (
            LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31],
             'min_child_samples': [5, 20], 'reg_alpha': [0, 0.1]}
        ),
        'XGBoost_v2': (
            XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'reg_alpha': [0, 0.1],
             'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
        ),
        'GBM_v2': (
            GradientBoostingRegressor(random_state=RS),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'min_samples_leaf': [5, 10],
             'subsample': [0.8, 1.0]}
        ),
    }

    reg_results = []
    reg_preds = {}
    for name, (model, grid) in reg_models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv,
                                scoring='neg_root_mean_squared_error',
                                refit=True, n_jobs=-1, random_state=RS)
        gs.fit(X_tr, y_tr)
        pred = gs.best_estimator_.predict(X_te)
        reg_preds[name] = pred

        mae = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2 = r2_score(y_te, pred)
        da = np.mean(np.sign(y_te) == np.sign(pred)) * 100
        elapsed = round(time.time() - t0, 1)

        reg_results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'DA%': da, 'Time_s': elapsed})
        print(f'  Best: {gs.best_params_}')
        print(f'  RMSE={rmse:.6f}  R2={r2:.6f}  DA={da:.1f}%  ({elapsed}s)')

    rdf = pd.DataFrame(reg_results).sort_values('RMSE')
    rdf.index = range(1, len(rdf) + 1)
    print(f'\n Regression Results:')
    print(rdf.to_string())
    rdf.to_csv(os.path.join(OUT_DIR, 'finetune_regression.csv'), index=False)

    # ═══════════════════════════════════════════════════════
    # CLASSIFICATION FINE-TUNE (TOP_20)
    # ═══════════════════════════════════════════════════════
    print(f'\n{P}\n 2B: CLASSIFICATION ({len(cls_feats)} features)\n{P}')
    print(f'  Features: {cls_feats}')

    X_tr_c = data['X_train'][cls_feats]
    X_te_c = data['X_test'][cls_feats]
    y_tr_c = (data['y_train'] > 0).astype(int)
    y_te_c = (data['y_test'] > 0).astype(int)

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_c), columns=cls_feats, index=X_tr_c.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te_c), columns=cls_feats, index=X_te_c.index)

    cls_models = {
        'GBM_cls_v2': (
            GradientBoostingClassifier(random_state=RS),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'min_samples_leaf': [5, 10],
             'subsample': [0.8, 1.0]}, False
        ),
        'XGB_cls_v2': (
            XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1, eval_metric='logloss'),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'reg_alpha': [0, 0.1],
             'subsample': [0.8, 1.0]}, False
        ),
        'LGBM_cls_v2': (
            LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
            {'n_estimators': [200, 500], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31],
             'min_child_samples': [5, 20]}, False
        ),
        'SVM_RBF_v2': (
            SVC(random_state=RS, probability=True, kernel='rbf'),
            {'C': [0.1, 1.0, 5.0, 10.0], 'gamma': ['scale', 'auto', 0.01]}, True
        ),
    }

    cls_results = []
    cls_preds = {}
    for name, (model, grid, use_sc) in cls_models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        Xtr = X_tr_sc if use_sc else X_tr_c
        Xte = X_te_sc if use_sc else X_te_c

        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=-1, random_state=RS)
        gs.fit(Xtr, y_tr_c)
        pred = gs.best_estimator_.predict(Xte)
        cls_preds[name] = pred
        prob = gs.best_estimator_.predict_proba(Xte)[:, 1] if hasattr(gs.best_estimator_, 'predict_proba') else None

        acc = accuracy_score(y_te_c, pred)
        f1m = f1_score(y_te_c, pred, average='macro')
        auc = roc_auc_score(y_te_c, prob) if prob is not None else np.nan
        elapsed = round(time.time() - t0, 1)

        cls_results.append({'Model': name, 'Accuracy': acc, 'F1_macro': f1m, 'AUC': auc, 'Time_s': elapsed})
        print(f'  Best: {gs.best_params_}')
        print(f'  Acc={acc:.4f}  F1m={f1m:.4f}  AUC={auc:.4f}  ({elapsed}s)')

    cdf = pd.DataFrame(cls_results).sort_values('F1_macro', ascending=False)
    cdf.index = range(1, len(cdf) + 1)
    print(f'\n Classification Results:')
    print(cdf.to_string())
    cdf.to_csv(os.path.join(OUT_DIR, 'finetune_classification.csv'), index=False)

    # ═══════════════════════════════════════════════════════
    # BACKTEST
    # ═══════════════════════════════════════════════════════
    print(f'\n{P}\n BACKTEST\n{P}')
    y_actual = data['y_test'].values
    dates = data['dates_test']
    bh = np.cumprod(1 + y_actual) - 1

    # Regression backtest (long/short)
    print(f'\n Regression (Long/Short):')
    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*50}')
    print(f' {"Buy&Hold":<20} {bh[-1]*100:>9.2f}')
    for name, pred in reg_preds.items():
        sig = np.sign(pred)
        sr = sig * y_actual
        cum = np.cumprod(1 + sr) - 1
        pk = np.maximum.accumulate(np.cumprod(1 + sr))
        dd = (np.cumprod(1 + sr) - pk) / pk
        total = cum[-1] * 100
        sharpe = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(252)
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {dd.min()*100:>9.2f}')

    # Classification backtest (long if UP)
    print(f'\n Classification (Long if UP, flat if DOWN):')
    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*50}')
    print(f' {"Buy&Hold":<20} {bh[-1]*100:>9.2f}')
    for name, pred in cls_preds.items():
        sr = pred * y_actual
        cum = np.cumprod(1 + sr) - 1
        pk = np.maximum.accumulate(np.cumprod(1 + sr))
        dd = (np.cumprod(1 + sr) - pk) / pk
        total = cum[-1] * 100
        sharpe = np.mean(sr) / (np.std(sr) + 1e-10) * np.sqrt(252)
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {dd.min()*100:>9.2f}')

    print(f'\n{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
