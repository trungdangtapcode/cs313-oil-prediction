"""
REGRESSION - Baseline 10 models
Usage: python ml/regression/train.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import load_data, get_tscv, RANDOM_STATE

OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
P = '=' * 90
RS = RANDOM_STATE

def evaluate(name, y_true, y_pred):
    return {'Model': name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'DA%': np.mean(np.sign(y_true) == np.sign(y_pred)) * 100}

def get_models():
    return [
        ('LinearRegression', LinearRegression(), {}, True),
        ('Ridge', Ridge(random_state=RS), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}, True),
        ('Lasso', Lasso(random_state=RS, max_iter=5000), {'alpha': [0.0001, 0.001, 0.01, 0.1]}, True),
        ('ElasticNet', ElasticNet(random_state=RS, max_iter=5000),
         {'alpha': [0.0001, 0.001, 0.01, 0.1], 'l1_ratio': [0.2, 0.5, 0.8]}, True),
        ('RandomForest', RandomForestRegressor(random_state=RS, n_jobs=-1),
         {'n_estimators': [100, 300], 'max_depth': [5, 10, 15], 'min_samples_leaf': [3, 5, 10]}, False),
        ('GradientBoosting', GradientBoostingRegressor(random_state=RS),
         {'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}, False),
        ('XGBoost', XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1),
         {'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'reg_alpha': [0, 0.1]}, False),
        ('LightGBM', LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, importance_type='gain'),
         {'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [15, 31]}, False),
        ('SVR', SVR(), {'C': [0.1, 1.0, 10.0], 'epsilon': [0.001, 0.01, 0.1], 'gamma': ['scale', 'auto']}, True),
        ('MLP', MLPRegressor(random_state=RS, max_iter=500, early_stopping=True),
         {'hidden_layer_sizes': [(64, 32), (128, 64, 32)], 'learning_rate_init': [0.001, 0.01], 'alpha': [0.0001, 0.001]}, True),
    ]

def main():
    print(f'\n{P}\n REGRESSION BASELINE\n{P}')
    data = load_data()
    tscv = get_tscv()
    results = []; predictions = {}; importances = {}

    for name, model, grid, use_sc in get_models():
        print(f'\n--- {name} ---')
        t0 = time.time()
        X_tr = data['X_train_sc'] if use_sc else data['X_train']
        X_te = data['X_test_sc'] if use_sc else data['X_test']
        if grid:
            gs = GridSearchCV(model, grid, cv=tscv, scoring='neg_root_mean_squared_error', refit=True, n_jobs=-1)
            gs.fit(X_tr, data['y_train']); best = gs.best_estimator_
            print(f'  Best: {gs.best_params_}  CV RMSE: {-gs.best_score_:.6f}')
        else:
            best = model; best.fit(X_tr, data['y_train'])
        y_pred = best.predict(X_te); predictions[name] = y_pred
        res = evaluate(name, data['y_test'].values, y_pred); res['Time_s'] = round(time.time()-t0, 1)
        results.append(res)
        print(f'  RMSE={res["RMSE"]:.6f} R2={res["R2"]:.6f} DA={res["DA%"]:.1f}% ({res["Time_s"]}s)')
        if hasattr(best, 'feature_importances_'):
            importances[name] = best.feature_importances_ / best.feature_importances_.sum()

    # Results
    rdf = pd.DataFrame(results).sort_values('RMSE'); rdf.index = range(1, len(rdf)+1)
    print(f'\n{P}\n RESULTS\n{P}'); print(rdf.to_string())
    rdf.to_csv(os.path.join(OUT, 'baseline_results.csv'), index=False)

    # Feature importance
    if importances:
        imp_df = pd.DataFrame(importances, index=data['features'])
        imp_df['Mean'] = imp_df.mean(axis=1); imp_df.sort_values('Mean', ascending=False, inplace=True)
        imp_df.to_csv(os.path.join(OUT, 'feature_importance.csv'))
        print(f'\n{P}\n FEATURE IMPORTANCE\n{P}'); print(imp_df.round(4).head(20).to_string())

    # Backtest
    print(f'\n{P}\n BACKTEST\n{P}')
    y_test = data['y_test'].values; dates = data['dates_test']
    bh = np.cumprod(1 + y_test) - 1
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, bh*100, 'k--', lw=1.5, label='Buy&Hold', alpha=0.7)
    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9}')
    print(f' {"-"*50}'); print(f' {"Buy&Hold":<20} {bh[-1]*100:>9.2f}')
    for name in rdf['Model'].values[:5]:
        sig = np.sign(predictions[name]); sr = sig * y_test
        cum = np.cumprod(1+sr)-1; pk = np.maximum.accumulate(np.cumprod(1+sr))
        dd = (np.cumprod(1+sr)-pk)/pk
        total = cum[-1]*100; sharpe = np.mean(sr)/(np.std(sr)+1e-10)*np.sqrt(252)
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {dd.min()*100:>9.2f}')
        ax.plot(dates, cum*100, lw=1, label=f'{name} ({total:.1f}%)', alpha=0.8)
    ax.legend(fontsize=7); ax.set_title('Backtest - Regression'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'backtest.png'), dpi=130, bbox_inches='tight'); plt.close()

    # Residuals
    bn = rdf.iloc[0]['Model']; res = data['y_test'].values - predictions[bn]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(predictions[bn], res, s=3, alpha=0.3); axes[0].axhline(0, color='red', ls='--')
    axes[0].set_title(f'{bn}: Residuals vs Predicted')
    axes[1].hist(res, bins=60, alpha=0.7, color='steelblue'); axes[1].set_title('Residual Distribution')
    axes[2].plot(dates, res, lw=0.5, alpha=0.6); axes[2].axhline(0, color='red', ls='--')
    axes[2].set_title('Residuals over Time')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'residuals.png'), dpi=130, bbox_inches='tight'); plt.close()

    print(f'\n Saved to ml/regression/results/')
    print(f'{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
