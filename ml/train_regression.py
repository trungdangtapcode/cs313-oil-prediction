"""
REGRESSION - Predict oil_return (continuous)
Models: LinearReg, Ridge, Lasso, ElasticNet, RF, GBM, XGBoost, LightGBM, SVR, MLP
Usage: python ml/train_regression.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

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

from config import load_data, get_tscv, OUT_DIR, RANDOM_STATE

P = '=' * 90

# ── Metrics ──────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Directional Accuracy
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'DA%': da}


# ── Model Definitions ───────────────────────────────────────────
def get_models():
    """Returns list of (name, model, param_grid, use_scaled_data)."""
    return [
        # --- Linear ---
        ('LinearRegression', LinearRegression(), {}, True),
        ('Ridge', Ridge(random_state=RANDOM_STATE), {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }, True),
        ('Lasso', Lasso(random_state=RANDOM_STATE, max_iter=5000), {
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }, True),
        ('ElasticNet', ElasticNet(random_state=RANDOM_STATE, max_iter=5000), {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'l1_ratio': [0.2, 0.5, 0.8]
        }, True),

        # --- Tree-based ---
        ('RandomForest', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
            'n_estimators': [100, 300],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [3, 5, 10]
        }, False),
        ('GradientBoosting', GradientBoostingRegressor(random_state=RANDOM_STATE), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }, False),
        ('XGBoost', XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=-1), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1]
        }, False),
        ('LightGBM', LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1, importance_type='gain'), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31]
        }, False),

        # --- Other ---
        ('SVR', SVR(), {
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.001, 0.01, 0.1],
            'gamma': ['scale', 'auto']
        }, True),
        ('MLP', MLPRegressor(random_state=RANDOM_STATE, max_iter=500, early_stopping=True), {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }, True),
    ]


# ── Main ─────────────────────────────────────────────────────────
def main():
    print(f'\n{P}\n REGRESSION - Oil Return Prediction\n{P}')

    data = load_data()
    tscv = get_tscv()
    models = get_models()

    results = []
    predictions = {}
    best_params = {}
    importances = {}

    for name, model, grid, use_sc in models:
        print(f'\n--- {name} ---')
        t0 = time.time()

        X_tr = data['X_train_sc'] if use_sc else data['X_train']
        X_te = data['X_test_sc'] if use_sc else data['X_test']

        if grid:
            gs = GridSearchCV(model, grid, cv=tscv, scoring='neg_root_mean_squared_error',
                              refit=True, n_jobs=-1)
            gs.fit(X_tr, data['y_train'])
            best = gs.best_estimator_
            best_params[name] = gs.best_params_
            cv_rmse = -gs.best_score_
            print(f'  Best params: {gs.best_params_}')
            print(f'  CV RMSE:     {cv_rmse:.6f}')
        else:
            best = model
            best.fit(X_tr, data['y_train'])
            best_params[name] = {}

        y_pred = best.predict(X_te)
        predictions[name] = y_pred
        res = evaluate(name, data['y_test'].values, y_pred)
        res['Time_s'] = round(time.time() - t0, 1)
        results.append(res)

        print(f'  Test MAE:  {res["MAE"]:.6f}')
        print(f'  Test RMSE: {res["RMSE"]:.6f}')
        print(f'  Test R2:   {res["R2"]:.6f}')
        print(f'  Test DA%:  {res["DA%"]:.1f}%')
        print(f'  Time:      {res["Time_s"]}s')

        # Feature importance (tree-based)
        if hasattr(best, 'feature_importances_'):
            imp = best.feature_importances_
            imp_norm = imp / imp.sum()  # normalize to sum=1
            importances[name] = imp_norm

    # ── Results Table ────────────────────────────────────────────
    print(f'\n{P}\n REGRESSION RESULTS (sorted by RMSE)\n{P}')
    rdf = pd.DataFrame(results).sort_values('RMSE')
    rdf.index = range(1, len(rdf) + 1)
    pd.set_option('display.float_format', '{:.6f}'.format)
    pd.set_option('display.width', 200)
    print(rdf.to_string())
    rdf.to_csv(os.path.join(OUT_DIR, 'regression_results.csv'), index=False)

    # ── Feature Importance (averaged, normalized) ────────────────
    if importances:
        print(f'\n{P}\n FEATURE IMPORTANCE (Tree models, normalized)\n{P}')
        imp_df = pd.DataFrame(importances, index=data['features'])
        imp_df['Mean'] = imp_df.mean(axis=1)
        imp_df.sort_values('Mean', ascending=False, inplace=True)
        print(imp_df.round(4).to_string())
        imp_df.to_csv(os.path.join(OUT_DIR, 'regression_feature_importance.csv'))

        # Plot
        top20 = imp_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.arange(len(top20))
        w = 0.2
        for i, col in enumerate([c for c in imp_df.columns if c != 'Mean']):
            ax.barh(x + i * w, top20[col], w, label=col, alpha=0.8)
        ax.set_yticks(x + w * 1.5)
        ax.set_yticklabels(top20.index, fontsize=7)
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.set_title('Feature Importance - Regression (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'regression_importance.png'), dpi=130, bbox_inches='tight')
        plt.close()

    # ── Backtest ─────────────────────────────────────────────────
    print(f'\n{P}\n BACKTEST (Long/Short strategy)\n{P}')
    y_test = data['y_test'].values
    dates = data['dates_test']
    buy_hold = np.cumprod(1 + y_test) - 1

    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9} {"DA%":>7}')
    print(f' {"-"*56}')
    print(f' {"Buy&Hold":<20} {buy_hold[-1]*100:>9.2f} {"":>8} {"":>9} {"":>7}')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, buy_hold * 100, 'k--', lw=1.5, label='Buy & Hold', alpha=0.7)

    for name in rdf['Model'].values[:5]:  # top 5 by RMSE
        pred = predictions[name]
        signal = np.sign(pred)
        strat_ret = signal * y_test
        cum = np.cumprod(1 + strat_ret) - 1
        peak = np.maximum.accumulate(np.cumprod(1 + strat_ret))
        dd = (np.cumprod(1 + strat_ret) - peak) / peak
        maxdd = dd.min() * 100
        total = cum[-1] * 100
        sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
        da = np.mean(np.sign(y_test) == signal) * 100
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {maxdd:>9.2f} {da:>6.1f}%')
        ax.plot(dates, cum * 100, lw=1, label=f'{name} ({total:.1f}%)', alpha=0.8)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Backtest - Top 5 Regression Models')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'regression_backtest.png'), dpi=130, bbox_inches='tight')
    plt.close()

    # ── Residual Analysis (best model) ───────────────────────────
    best_name = rdf.iloc[0]['Model']
    best_pred = predictions[best_name]
    residuals = data['y_test'].values - best_pred

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].scatter(best_pred, residuals, s=3, alpha=0.3)
    axes[0].axhline(0, color='red', ls='--'); axes[0].set_title(f'{best_name}: Residuals vs Predicted')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')

    axes[1].hist(residuals, bins=60, alpha=0.7, color='steelblue', edgecolor='white')
    axes[1].set_title('Residual Distribution')

    axes[2].plot(dates, residuals, lw=0.5, alpha=0.6)
    axes[2].axhline(0, color='red', ls='--'); axes[2].set_title('Residuals over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'regression_residuals.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Results saved to ml/results/')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
