"""
STEP 3: Stacking & Voting Ensembles
Usage: python ml/step3_ensemble.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                               StackingRegressor, StackingClassifier,
                               VotingRegressor, VotingClassifier)
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score)
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from config import load_data, get_tscv, OUT_DIR, RANDOM_STATE as RS

P = '=' * 90


def main():
    print(f'\n{P}\n STEP 3: ENSEMBLES\n{P}')
    data = load_data()

    # Read best subsets from step1
    comp = pd.read_csv(os.path.join(OUT_DIR, 'subset_comparison.csv'))
    best_reg = comp.loc[comp['RMSE'].idxmin(), 'Subset'].split('(')[0].strip()
    best_cls = comp.loc[comp['F1m'].idxmax(), 'Subset'].split('(')[0].strip()
    reg_f = pd.read_csv(os.path.join(OUT_DIR, f'subset_{best_reg}.csv'), header=None)[0].tolist()
    cls_f = pd.read_csv(os.path.join(OUT_DIR, f'subset_{best_cls}.csv'), header=None)[0].tolist()
    print(f'  Reg subset: {best_reg} ({len(reg_f)} feats) | Cls subset: {best_cls} ({len(cls_f)} feats)')

    y_actual = data['y_test'].values
    dates = data['dates_test']
    bh = np.cumprod(1 + y_actual) - 1

    # ═══════════════════════════════════════
    # REGRESSION ENSEMBLES
    # ═══════════════════════════════════════
    print(f'\n{P}\n 3A: REGRESSION ENSEMBLES ({len(reg_f)} features)\n{P}')
    X_tr = data['X_train'][reg_f]
    X_te = data['X_test'][reg_f]
    y_tr, y_te = data['y_train'], data['y_test']

    base_reg = [
        ('lgbm', LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1,
                                n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=15)),
        ('xgb', XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1,
                              n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('gbm', GradientBoostingRegressor(random_state=RS,
                                           n_estimators=100, max_depth=3, learning_rate=0.05)),
    ]

    reg_results = []
    reg_preds = {}

    print('\n--- Voting_Reg ---')
    t0 = time.time()
    m = VotingRegressor(estimators=base_reg, n_jobs=-1)
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    reg_preds['Voting_Reg'] = p
    rmse = np.sqrt(mean_squared_error(y_te, p)); r2 = r2_score(y_te, p)
    da = np.mean(np.sign(y_te) == np.sign(p)) * 100
    reg_results.append({'Model': 'Voting_Reg', 'RMSE': rmse, 'R2': r2, 'DA%': da, 'Time_s': round(time.time()-t0,1)})
    print(f'  RMSE={rmse:.6f} R2={r2:.6f} DA={da:.1f}% ({reg_results[-1]["Time_s"]}s)')

    print('\n--- Stacking_Reg ---')
    t0 = time.time()
    m = StackingRegressor(estimators=base_reg, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    reg_preds['Stacking_Reg'] = p
    rmse = np.sqrt(mean_squared_error(y_te, p)); r2 = r2_score(y_te, p)
    da = np.mean(np.sign(y_te) == np.sign(p)) * 100
    reg_results.append({'Model': 'Stacking_Reg', 'RMSE': rmse, 'R2': r2, 'DA%': da, 'Time_s': round(time.time()-t0,1)})
    print(f'  RMSE={rmse:.6f} R2={r2:.6f} DA={da:.1f}% ({reg_results[-1]["Time_s"]}s)')

    rdf = pd.DataFrame(reg_results)
    print(f'\n{rdf.to_string(index=False)}')
    rdf.to_csv(os.path.join(OUT_DIR, 'ensemble_regression.csv'), index=False)

    # ═══════════════════════════════════════
    # CLASSIFICATION ENSEMBLES
    # ═══════════════════════════════════════
    print(f'\n{P}\n 3B: CLASSIFICATION ENSEMBLES ({len(cls_f)} features)\n{P}')
    X_tr_c = data['X_train'][cls_f]
    X_te_c = data['X_test'][cls_f]
    y_tr_c = (data['y_train'] > 0).astype(int)
    y_te_c = (data['y_test'] > 0).astype(int)

    base_cls = [
        ('lgbm', LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=-1,
                                 n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=15)),
        ('xgb', XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1, eval_metric='logloss',
                               n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('gbm', GradientBoostingClassifier(random_state=RS,
                                            n_estimators=100, max_depth=3, learning_rate=0.05)),
    ]

    cls_results = []
    cls_preds = {}

    print('\n--- Voting_Cls (soft) ---')
    t0 = time.time()
    m = VotingClassifier(estimators=base_cls, voting='soft', n_jobs=-1)
    m.fit(X_tr_c, y_tr_c)
    p = m.predict(X_te_c)
    prob = m.predict_proba(X_te_c)[:, 1]
    cls_preds['Voting_Cls'] = p
    acc = accuracy_score(y_te_c, p); f1m = f1_score(y_te_c, p, average='macro')
    auc = roc_auc_score(y_te_c, prob)
    cls_results.append({'Model': 'Voting_Cls', 'Acc': acc, 'F1m': f1m, 'AUC': auc, 'Time_s': round(time.time()-t0,1)})
    print(f'  Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f} ({cls_results[-1]["Time_s"]}s)')

    print('\n--- Stacking_Cls ---')
    t0 = time.time()
    m = StackingClassifier(estimators=base_cls,
                           final_estimator=LogisticRegression(random_state=RS, max_iter=1000),
                           cv=3, n_jobs=-1)
    m.fit(X_tr_c, y_tr_c)
    p = m.predict(X_te_c)
    prob = m.predict_proba(X_te_c)[:, 1]
    cls_preds['Stacking_Cls'] = p
    acc = accuracy_score(y_te_c, p); f1m = f1_score(y_te_c, p, average='macro')
    auc = roc_auc_score(y_te_c, prob)
    cls_results.append({'Model': 'Stacking_Cls', 'Acc': acc, 'F1m': f1m, 'AUC': auc, 'Time_s': round(time.time()-t0,1)})
    print(f'  Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f} ({cls_results[-1]["Time_s"]}s)')

    cdf = pd.DataFrame(cls_results)
    print(f'\n{cdf.to_string(index=False)}')
    cdf.to_csv(os.path.join(OUT_DIR, 'ensemble_classification.csv'), index=False)

    # ═══════════════════════════════════════
    # BACKTEST
    # ═══════════════════════════════════════
    print(f'\n{P}\n BACKTEST\n{P}')

    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*50}')
    print(f' {"Buy&Hold":<20} {bh[-1]*100:>9.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Reg backtest
    axes[0].plot(dates, bh*100, 'k--', lw=1.5, label='Buy&Hold', alpha=0.7)
    for name, pred in reg_preds.items():
        sr = np.sign(pred) * y_actual
        cum = np.cumprod(1 + sr) - 1
        pk = np.maximum.accumulate(np.cumprod(1 + sr))
        dd = (np.cumprod(1 + sr) - pk) / pk
        total = cum[-1]*100; sharpe = np.mean(sr)/(np.std(sr)+1e-10)*np.sqrt(252)
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {dd.min()*100:>9.2f}')
        axes[0].plot(dates, cum*100, lw=1, label=f'{name} ({total:.1f}%)', alpha=0.8)
    axes[0].set_title('Regression Ensembles'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    # Cls backtest
    axes[1].plot(dates, bh*100, 'k--', lw=1.5, label='Buy&Hold', alpha=0.7)
    for name, pred in cls_preds.items():
        sr = pred * y_actual
        cum = np.cumprod(1 + sr) - 1
        pk = np.maximum.accumulate(np.cumprod(1 + sr))
        dd = (np.cumprod(1 + sr) - pk) / pk
        total = cum[-1]*100; sharpe = np.mean(sr)/(np.std(sr)+1e-10)*np.sqrt(252)
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {dd.min()*100:>9.2f}')
        axes[1].plot(dates, cum*100, lw=1, label=f'{name} ({total:.1f}%)', alpha=0.8)
    axes[1].set_title('Classification Ensembles'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ensemble_backtest.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Saved to ml/results/')
    print(f'{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
