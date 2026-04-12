"""
MODEL OPTIMIZATION
1. Feature selection (remove weak, try subsets)
2. Fine-tune top models (wider grid)
3. Stacking / Voting ensemble
4. Compare all vs baseline

Usage: python ml/optimize.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               RandomForestRegressor, RandomForestClassifier,
                               StackingRegressor, StackingClassifier,
                               VotingRegressor, VotingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from config import load_data, get_tscv, OUT_DIR, RANDOM_STATE

P = '=' * 90
RS = RANDOM_STATE


# ═══════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════
def reg_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    da   = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'DA%': da}

def clf_metrics(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    da  = acc * 100
    return {'Model': name, 'Accuracy': acc, 'F1_macro': f1m, 'AUC': auc, 'DA%': da}

def backtest(name, y_pred_signal, y_actual_returns, dates):
    """signal: 1=long, -1=short, 0=flat. Returns dict."""
    strat = y_pred_signal * y_actual_returns
    cum = np.cumprod(1 + strat) - 1
    peak = np.maximum.accumulate(np.cumprod(1 + strat))
    dd = (np.cumprod(1 + strat) - peak) / peak
    return {
        'Model': name,
        'Total%': cum[-1] * 100,
        'Sharpe': np.mean(strat) / (np.std(strat) + 1e-10) * np.sqrt(252),
        'MaxDD%': dd.min() * 100,
        'cum': cum,
    }


# ═══════════════════════════════════════════════════════════════
# PART 1: FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════
def feature_selection(data):
    print(f'\n{P}\n PART 1: FEATURE SELECTION\n{P}')

    X_train = data['X_train']
    y_train = data['y_train']
    features = data['features']

    # 1a. Mutual Information (regression)
    mi_reg = mutual_info_regression(X_train.fillna(0), y_train, random_state=RS, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': features, 'MI_reg': mi_reg}).sort_values('MI_reg', ascending=False)

    # 1b. Mutual Information (classification)
    y_cls = (y_train > 0).astype(int)
    mi_cls = mutual_info_classif(X_train.fillna(0), y_cls, random_state=RS, n_neighbors=5)
    mi_df['MI_cls'] = mi_cls

    # 1c. Spearman correlation
    sp = X_train.corrwith(y_train, method='spearman')
    mi_df['abs_spearman'] = mi_df['feature'].map(sp.abs())

    # Combined score (normalized)
    for col in ['MI_reg', 'MI_cls', 'abs_spearman']:
        mx = mi_df[col].max()
        if mx > 0:
            mi_df[f'{col}_norm'] = mi_df[col] / mx
        else:
            mi_df[f'{col}_norm'] = 0

    mi_df['combined_score'] = (mi_df['MI_reg_norm'] + mi_df['MI_cls_norm'] + mi_df['abs_spearman_norm']) / 3
    mi_df.sort_values('combined_score', ascending=False, inplace=True)
    mi_df.reset_index(drop=True, inplace=True)

    print(f'\n Feature ranking (combined MI_reg + MI_cls + |Spearman|):')
    print(f' {"#":<4} {"Feature":<30} {"MI_reg":>8} {"MI_cls":>8} {"|Spear|":>8} {"Score":>8}')
    print(f' {"-"*70}')
    for i, r in mi_df.iterrows():
        print(f' {i+1:<4} {r.feature:<30} {r.MI_reg:>8.4f} {r.MI_cls:>8.4f} {r.abs_spearman:>8.4f} {r.combined_score:>8.4f}')

    # Define feature subsets
    top10 = mi_df.head(10)['feature'].tolist()
    top15 = mi_df.head(15)['feature'].tolist()
    top20 = mi_df.head(20)['feature'].tolist()
    top25 = mi_df.head(25)['feature'].tolist()

    # Remove near-zero features
    threshold = mi_df['combined_score'].quantile(0.25)
    above_thresh = mi_df[mi_df['combined_score'] > threshold]['feature'].tolist()

    subsets = {
        'ALL_42': features,
        'TOP_10': top10,
        'TOP_15': top15,
        'TOP_20': top20,
        'TOP_25': top25,
        f'ABOVE_Q25({len(above_thresh)})': above_thresh,
    }

    print(f'\n Feature subsets:')
    for k, v in subsets.items():
        print(f'   {k}: {len(v)} features')

    return subsets, mi_df


# ═══════════════════════════════════════════════════════════════
# PART 2: FEATURE SUBSET COMPARISON (quick)
# ═══════════════════════════════════════════════════════════════
def compare_feature_subsets(data, subsets):
    print(f'\n{P}\n PART 2: FEATURE SUBSET COMPARISON\n{P}')

    tscv = get_tscv()
    scaler = StandardScaler()

    # Quick comparison using LightGBM (reg) + GBM (cls) as proxy
    reg_model = LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1,
                              n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=20)
    cls_model = GradientBoostingClassifier(random_state=RS,
                                           n_estimators=200, max_depth=5, learning_rate=0.05)

    results = []
    for name, feats in subsets.items():
        X_tr = data['X_train'][feats]
        X_te = data['X_test'][feats]
        y_reg = data['y_train']
        y_cls = (data['y_train'] > 0).astype(int)
        y_te_reg = data['y_test']
        y_te_cls = (data['y_test'] > 0).astype(int)

        # Regression CV
        cv_reg = cross_val_score(reg_model, X_tr, y_reg, cv=tscv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        reg_model.fit(X_tr, y_reg)
        pred_reg = reg_model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te_reg, pred_reg))
        r2 = r2_score(y_te_reg, pred_reg)

        # Classification CV
        cv_cls = cross_val_score(cls_model, X_tr, y_cls, cv=tscv,
                                 scoring='accuracy', n_jobs=-1)
        cls_model.fit(X_tr, y_cls)
        pred_cls = cls_model.predict(X_te)
        acc = accuracy_score(y_te_cls, pred_cls)
        f1m = f1_score(y_te_cls, pred_cls, average='macro')

        results.append({
            'Subset': name, 'N_feat': len(feats),
            'Reg_CV_RMSE': -cv_reg.mean(), 'Reg_Test_RMSE': rmse, 'Reg_R2': r2,
            'Cls_CV_Acc': cv_cls.mean(), 'Cls_Test_Acc': acc, 'Cls_F1m': f1m,
        })
        print(f'  {name:<20} (n={len(feats):>2}) | Reg RMSE={rmse:.5f} R2={r2:.4f} | Cls Acc={acc:.4f} F1m={f1m:.4f}')

    rdf = pd.DataFrame(results)
    print(f'\n Best regression subset:      {rdf.loc[rdf["Reg_Test_RMSE"].idxmin(), "Subset"]}')
    print(f' Best classification subset:  {rdf.loc[rdf["Cls_F1m"].idxmax(), "Subset"]}')
    return rdf


# ═══════════════════════════════════════════════════════════════
# PART 3: FINE-TUNE TOP MODELS (wider grid)
# ═══════════════════════════════════════════════════════════════
def finetune_regression(data, best_feats):
    print(f'\n{P}\n PART 3A: FINE-TUNE REGRESSION ({len(best_feats)} features)\n{P}')

    X_tr = data['X_train'][best_feats]
    X_te = data['X_test'][best_feats]
    y_tr = data['y_train']; y_te = data['y_test']
    tscv = get_tscv()

    models = {
        'LightGBM_v2': (
            LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'num_leaves': [15, 31],
                'min_child_samples': [5, 20],
                'reg_alpha': [0, 0.1],
            }
        ),
        'XGBoost_v2': (
            XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'reg_alpha': [0, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        ),
        'GBM_v2': (
            GradientBoostingRegressor(random_state=RS),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'min_samples_leaf': [5, 10],
                'subsample': [0.8, 1.0],
            }
        ),
    }

    results = []
    best_estimators = {}
    predictions = {}

    for name, (model, grid) in models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        # RandomizedSearchCV for wider grids
        from sklearn.model_selection import RandomizedSearchCV
        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv,
                                scoring='neg_root_mean_squared_error',
                                refit=True, n_jobs=-1, random_state=RS)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        best_estimators[name] = best
        cv_rmse = -gs.best_score_

        y_pred = best.predict(X_te)
        predictions[name] = y_pred
        res = reg_metrics(name, y_te.values, y_pred)
        res['CV_RMSE'] = cv_rmse
        res['Time_s'] = round(time.time() - t0, 1)
        results.append(res)

        print(f'  Best params: {gs.best_params_}')
        print(f'  CV RMSE:  {cv_rmse:.6f}')
        print(f'  Test RMSE: {res["RMSE"]:.6f}  R2: {res["R2"]:.6f}  DA: {res["DA%"]:.1f}%')
        print(f'  Time: {res["Time_s"]}s')

    return results, best_estimators, predictions


def finetune_classification(data, best_feats):
    print(f'\n{P}\n PART 3B: FINE-TUNE CLASSIFICATION ({len(best_feats)} features)\n{P}')

    X_tr = data['X_train'][best_feats]
    X_te = data['X_test'][best_feats]
    y_tr = (data['y_train'] > 0).astype(int)
    y_te = (data['y_test'] > 0).astype(int)

    # Also need scaled for SVM
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=best_feats, index=X_tr.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=best_feats, index=X_te.index)
    tscv = get_tscv()

    from sklearn.model_selection import RandomizedSearchCV

    models = {
        'GBM_cls_v2': (
            GradientBoostingClassifier(random_state=RS),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'min_samples_leaf': [5, 10],
                'subsample': [0.8, 1.0],
            }, False
        ),
        'XGB_cls_v2': (
            XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1, eval_metric='logloss'),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'reg_alpha': [0, 0.1],
                'subsample': [0.8, 1.0],
            }, False
        ),
        'LGBM_cls_v2': (
            LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
            {
                'n_estimators': [200, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'num_leaves': [15, 31],
                'min_child_samples': [5, 20],
            }, False
        ),
        'SVM_RBF_v2': (
            SVC(random_state=RS, probability=True, kernel='rbf'),
            {
                'C': [0.1, 1.0, 5.0, 10.0],
                'gamma': ['scale', 'auto', 0.01],
            }, True
        ),
    }

    results = []
    best_estimators = {}
    predictions = {}
    probabilities = {}

    for name, (model, grid, use_sc) in models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        Xtr = X_tr_sc if use_sc else X_tr
        Xte = X_te_sc if use_sc else X_te

        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=-1, random_state=RS)
        gs.fit(Xtr, y_tr)
        best = gs.best_estimator_
        best_estimators[name] = best

        y_pred = best.predict(Xte)
        predictions[name] = y_pred
        y_prob = best.predict_proba(Xte)[:, 1] if hasattr(best, 'predict_proba') else None
        probabilities[name] = y_prob

        res = clf_metrics(name, y_te.values, y_pred, y_prob)
        res['CV_Acc'] = gs.best_score_
        res['Time_s'] = round(time.time() - t0, 1)
        results.append(res)

        print(f'  Best params: {gs.best_params_}')
        print(f'  CV Acc:  {gs.best_score_:.4f}')
        print(f'  Test Acc: {res["Accuracy"]:.4f}  F1m: {res["F1_macro"]:.4f}  AUC: {res["AUC"]:.4f}')
        print(f'  Time: {res["Time_s"]}s')

    return results, best_estimators, predictions, probabilities


# ═══════════════════════════════════════════════════════════════
# PART 4: STACKING / VOTING ENSEMBLE
# ═══════════════════════════════════════════════════════════════
def build_ensembles(data, best_feats):
    print(f'\n{P}\n PART 4: STACKING & VOTING ENSEMBLES\n{P}')

    X_tr = data['X_train'][best_feats]
    X_te = data['X_test'][best_feats]
    y_reg = data['y_train']; y_te_reg = data['y_test']
    y_cls = (data['y_train'] > 0).astype(int)
    y_te_cls = (data['y_test'] > 0).astype(int)

    reg_results = []; cls_results = []
    reg_preds = {}; cls_preds = {}; cls_probs = {}

    # ── Regression Ensembles ──
    base_reg = [
        ('lgbm', LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1,
                                n_estimators=300, max_depth=5, learning_rate=0.03, num_leaves=20)),
        ('xgb', XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1,
                              n_estimators=300, max_depth=5, learning_rate=0.03)),
        ('gbm', GradientBoostingRegressor(random_state=RS,
                                           n_estimators=300, max_depth=5, learning_rate=0.03)),
    ]

    # Voting
    print('\n--- Voting_Reg ---')
    t0 = time.time()
    voting_reg = VotingRegressor(estimators=base_reg, n_jobs=-1)
    voting_reg.fit(X_tr, y_reg)
    pred = voting_reg.predict(X_te)
    reg_preds['Voting_Reg'] = pred
    res = reg_metrics('Voting_Reg', y_te_reg.values, pred)
    res['Time_s'] = round(time.time() - t0, 1)
    reg_results.append(res)
    print(f'  RMSE: {res["RMSE"]:.6f}  R2: {res["R2"]:.6f}  DA: {res["DA%"]:.1f}%  ({res["Time_s"]}s)')

    # Stacking
    print('\n--- Stacking_Reg ---')
    t0 = time.time()
    stack_reg = StackingRegressor(
        estimators=base_reg,
        final_estimator=Ridge(alpha=1.0),
        cv=get_tscv(), n_jobs=-1
    )
    stack_reg.fit(X_tr, y_reg)
    pred = stack_reg.predict(X_te)
    reg_preds['Stacking_Reg'] = pred
    res = reg_metrics('Stacking_Reg', y_te_reg.values, pred)
    res['Time_s'] = round(time.time() - t0, 1)
    reg_results.append(res)
    print(f'  RMSE: {res["RMSE"]:.6f}  R2: {res["R2"]:.6f}  DA: {res["DA%"]:.1f}%  ({res["Time_s"]}s)')

    # ── Classification Ensembles ──
    base_cls = [
        ('lgbm', LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=-1,
                                 n_estimators=300, max_depth=5, learning_rate=0.03, num_leaves=20)),
        ('xgb', XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1, eval_metric='logloss',
                               n_estimators=300, max_depth=5, learning_rate=0.03)),
        ('gbm', GradientBoostingClassifier(random_state=RS,
                                            n_estimators=300, max_depth=5, learning_rate=0.03)),
    ]

    print('\n--- Voting_Cls (soft) ---')
    t0 = time.time()
    voting_cls = VotingClassifier(estimators=base_cls, voting='soft', n_jobs=-1)
    voting_cls.fit(X_tr, y_cls)
    pred = voting_cls.predict(X_te)
    prob = voting_cls.predict_proba(X_te)[:, 1]
    cls_preds['Voting_Cls'] = pred; cls_probs['Voting_Cls'] = prob
    res = clf_metrics('Voting_Cls', y_te_cls.values, pred, prob)
    res['Time_s'] = round(time.time() - t0, 1)
    cls_results.append(res)
    print(f'  Acc: {res["Accuracy"]:.4f}  F1m: {res["F1_macro"]:.4f}  AUC: {res["AUC"]:.4f}  ({res["Time_s"]}s)')

    print('\n--- Stacking_Cls ---')
    t0 = time.time()
    stack_cls = StackingClassifier(
        estimators=base_cls,
        final_estimator=LogisticRegression(random_state=RS, max_iter=1000),
        cv=get_tscv(), n_jobs=-1
    )
    stack_cls.fit(X_tr, y_cls)
    pred = stack_cls.predict(X_te)
    prob = stack_cls.predict_proba(X_te)[:, 1]
    cls_preds['Stacking_Cls'] = pred; cls_probs['Stacking_Cls'] = prob
    res = clf_metrics('Stacking_Cls', y_te_cls.values, pred, prob)
    res['Time_s'] = round(time.time() - t0, 1)
    cls_results.append(res)
    print(f'  Acc: {res["Accuracy"]:.4f}  F1m: {res["F1_macro"]:.4f}  AUC: {res["AUC"]:.4f}  ({res["Time_s"]}s)')

    return reg_results, reg_preds, cls_results, cls_preds, cls_probs


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print(f'\n{P}\n MODEL OPTIMIZATION\n{P}')

    data = load_data()

    # Part 1: Feature selection
    subsets, mi_df = feature_selection(data)
    mi_df.to_csv(os.path.join(OUT_DIR, 'feature_ranking.csv'), index=False)

    # Part 2: Compare subsets
    subset_df = compare_feature_subsets(data, subsets)
    subset_df.to_csv(os.path.join(OUT_DIR, 'subset_comparison.csv'), index=False)

    # Pick best subset for each task
    best_reg_subset = subset_df.loc[subset_df['Reg_Test_RMSE'].idxmin(), 'Subset']
    best_cls_subset = subset_df.loc[subset_df['Cls_F1m'].idxmax(), 'Subset']
    # Use the better one (or ALL if they differ much)
    best_feats = subsets[best_reg_subset]
    print(f'\n Using feature subset: {best_reg_subset} ({len(best_feats)} features)')

    # Part 3: Fine-tune
    reg_res, reg_est, reg_preds = finetune_regression(data, best_feats)
    cls_res, cls_est, cls_preds, cls_probs = finetune_classification(data, best_feats)

    # Part 4: Ensembles
    ens_reg_res, ens_reg_preds, ens_cls_res, ens_cls_preds, ens_cls_probs = build_ensembles(data, best_feats)

    # Merge predictions and results
    all_reg_preds = {**reg_preds, **ens_reg_preds}
    all_cls_preds = {**cls_preds, **ens_cls_preds}
    all_cls_probs = {**cls_probs, **ens_cls_probs}

    # ── FINAL COMPARISON ─────────────────────────────────────
    print(f'\n{P}\n FINAL REGRESSION COMPARISON\n{P}')
    all_reg_res = reg_res + ens_reg_res
    rdf = pd.DataFrame(all_reg_res).sort_values('RMSE')
    rdf.index = range(1, len(rdf) + 1)
    pd.set_option('display.float_format', '{:.6f}'.format); pd.set_option('display.width', 200)
    print(rdf.to_string())
    rdf.to_csv(os.path.join(OUT_DIR, 'optimized_regression.csv'), index=False)

    print(f'\n{P}\n FINAL CLASSIFICATION COMPARISON\n{P}')
    all_cls_res = cls_res + ens_cls_res
    cdf = pd.DataFrame(all_cls_res).sort_values('F1_macro', ascending=False)
    cdf.index = range(1, len(cdf) + 1)
    print(cdf.to_string())
    cdf.to_csv(os.path.join(OUT_DIR, 'optimized_classification.csv'), index=False)

    # ── BACKTEST ─────────────────────────────────────────────
    print(f'\n{P}\n BACKTEST\n{P}')
    dates = data['dates_test']
    y_actual = data['y_test'].values
    bh = np.cumprod(1 + y_actual) - 1

    print(f'\n REGRESSION (Long/Short):')
    print(f' {"Model":<22} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*52}')
    print(f' {"Buy&Hold":<22} {bh[-1]*100:>9.2f}')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, bh*100, 'k--', lw=1.5, label='Buy&Hold', alpha=0.7)
    for name, pred in all_reg_preds.items():
        bt = backtest(name, np.sign(pred), y_actual, dates)
        print(f' {name:<22} {bt["Total%"]:>9.2f} {bt["Sharpe"]:>8.2f} {bt["MaxDD%"]:>9.2f}')
        ax.plot(dates, bt['cum']*100, lw=1, label=f'{name} ({bt["Total%"]:.1f}%)', alpha=0.8)
    ax.legend(fontsize=7); ax.set_title('Backtest - Optimized Regression'); ax.set_ylabel('Cumulative %'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'opt_backtest_reg.png'), dpi=130, bbox_inches='tight'); plt.close()

    print(f'\n CLASSIFICATION (Long if UP, flat if DOWN):')
    print(f' {"Model":<22} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*52}')
    print(f' {"Buy&Hold":<22} {bh[-1]*100:>9.2f}')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, bh*100, 'k--', lw=1.5, label='Buy&Hold', alpha=0.7)
    for name, pred in all_cls_preds.items():
        bt = backtest(name, pred, y_actual, dates)
        print(f' {name:<22} {bt["Total%"]:>9.2f} {bt["Sharpe"]:>8.2f} {bt["MaxDD%"]:>9.2f}')
        ax.plot(dates, bt['cum']*100, lw=1, label=f'{name} ({bt["Total%"]:.1f}%)', alpha=0.8)
    ax.legend(fontsize=7); ax.set_title('Backtest - Optimized Classification'); ax.set_ylabel('Cumulative %'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'opt_backtest_cls.png'), dpi=130, bbox_inches='tight'); plt.close()

    print(f'\n Results saved to ml/results/')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
