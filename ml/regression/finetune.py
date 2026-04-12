"""
REGRESSION - Fine-tune + Ensemble
Usage: python ml/regression/finetune.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import load_data, get_tscv, RANDOM_STATE as RS

OUT = os.path.join(os.path.dirname(__file__), 'results')
P = '=' * 90

def main():
    print(f'\n{P}\n REGRESSION FINE-TUNE + ENSEMBLE\n{P}')
    data = load_data(); tscv = get_tscv()
    X_tr, X_te = data['X_train'], data['X_test']
    y_tr, y_te = data['y_train'], data['y_test']

    # Fine-tune
    models = {
        'LightGBM_v2': (LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
                        {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                         'num_leaves': [15, 31], 'min_child_samples': [5, 20], 'reg_alpha': [0, 0.1]}),
        'XGBoost_v2': (XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1),
                       {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                        'reg_alpha': [0, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}),
        'GBM_v2': (GradientBoostingRegressor(random_state=RS),
                   {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                    'min_samples_leaf': [5, 10], 'subsample': [0.8, 1.0]}),
    }
    results = []; preds = {}
    for name, (model, grid) in models.items():
        print(f'\n--- {name} ---'); t0 = time.time()
        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv, scoring='neg_root_mean_squared_error',
                                refit=True, n_jobs=-1, random_state=RS)
        gs.fit(X_tr, y_tr); pred = gs.best_estimator_.predict(X_te); preds[name] = pred
        rmse = np.sqrt(mean_squared_error(y_te, pred)); r2 = r2_score(y_te, pred)
        da = np.mean(np.sign(y_te)==np.sign(pred))*100
        results.append({'Model': name, 'RMSE': rmse, 'R2': r2, 'DA%': da, 'Time_s': round(time.time()-t0, 1)})
        print(f'  Best: {gs.best_params_}\n  RMSE={rmse:.6f} R2={r2:.6f} DA={da:.1f}%')

    # Ensemble
    base = [
        ('lgbm', LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('xgb', XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1, n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('gbm', GradientBoostingRegressor(random_state=RS, n_estimators=100, max_depth=3, learning_rate=0.05)),
    ]
    for ename, ens in [('Voting', VotingRegressor(estimators=base, n_jobs=-1)),
                        ('Stacking', StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1))]:
        print(f'\n--- {ename} ---'); t0 = time.time()
        ens.fit(X_tr, y_tr); pred = ens.predict(X_te); preds[ename] = pred
        rmse = np.sqrt(mean_squared_error(y_te, pred)); r2 = r2_score(y_te, pred)
        da = np.mean(np.sign(y_te)==np.sign(pred))*100
        results.append({'Model': ename, 'RMSE': rmse, 'R2': r2, 'DA%': da, 'Time_s': round(time.time()-t0, 1)})
        print(f'  RMSE={rmse:.6f} R2={r2:.6f} DA={da:.1f}%')

    rdf = pd.DataFrame(results).sort_values('RMSE')
    print(f'\n{P}\n RESULTS\n{P}'); print(rdf.to_string(index=False))
    rdf.to_csv(os.path.join(OUT, 'finetune_results.csv'), index=False)
    print(f'\n{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
