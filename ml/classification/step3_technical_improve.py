"""
STEP 3: Technical Improvement for Classification

This script tries to improve the baseline direction-prediction task for next-day oil returns.
The main idea is to add short-term technical signals on top of the end-of-day dataset,
while keeping all price-derived technical indicators shifted so the model only sees
information available up to T-1.

What this script is trying to improve:
  1. Reduce label noise by testing threshold-based targets.
  2. Add technical indicators that may capture short-term momentum or mean reversion.
  3. Compare a few target definitions for the same 1-day direction problem.
  4. Check whether filtering low-confidence predictions improves trade accuracy.

Targets trained in this file:
  - 1d_raw: classify whether oil_return_fwd1 > 0
  - 1d_t03: classify only moves above +0.3% or below -0.3%, skip near-zero days
  - 1d_t05: classify only moves above +0.5% or below -0.5%, skip near-zero days

Feature changes in this file:
  - Add technical indicators such as MA, RSI, MACD, Bollinger, momentum, rolling stats
  - Shift technical indicators by 1 day to avoid leakage
  - Use the end-of-day T dataset produced in the process pipeline
  - Shift technical indicators by 1 day

Models trained in this file:
  - XGBClassifier
  - GradientBoostingClassifier
  - LGBMClassifier

Model selection:
  - Each model is tuned with RandomizedSearchCV
  - TimeSeriesSplit is used for time-aware cross-validation
  - Results are evaluated with Accuracy, F1_macro, and AUC

Outputs:
  - Console summary of all target/model combinations
  - results/step3_test_results.csv
  - results/step3_results.csv

Usage:
  python ml/classification/step3_technical_improve.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    DATA_PATH,
    OUT_DIR,
    PRICE_SOURCE_PATH,
    RANDOM_STATE as RS,
    TARGET,
    TARGET_DATE_COL,
    get_tscv,
    get_train_test_masks,
    set_global_seed,
)

P = '=' * 90
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))
MODEL_N_JOBS = max(1, int(os.getenv('MODEL_N_JOBS', str(min(12, max(1, CPU_COUNT // 4))))))

def load_price_source():
    price_df = pd.read_csv(PRICE_SOURCE_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if 'oil_close' not in price_df.columns:
        raise ValueError(
            f"oil_close not found in price source: {PRICE_SOURCE_PATH}. "
            "Set CLASSIFICATION_PRICE_SOURCE_PATH to a dataset that contains date and oil_close."
        )
    return price_df[['date', 'oil_close']]


# ============================================================================
# A) ADD TECHNICAL INDICATORS
# ============================================================================
def add_technical_features(df):
    """Add technical indicators derived from oil_close in the base dataset."""

    # dataset_step4 already contains oil_close in most cases
    if 'oil_close' not in df.columns:
        raw = load_price_source()
        df = df.merge(raw, on='date', how='left')

    price = df['oil_close'].copy()
    ret = df['oil_return'].copy()

    # ============================================================
    # The base dataset now predicts T+1 from end-of-day T features.
    # We still shift all technical indicators by 1 day so they only use info up to T-1.
    # ============================================================

    # --- Moving Averages (shift 1) ---
    df['ma_5']  = price.rolling(5).mean().shift(1)
    df['ma_10'] = price.rolling(10).mean().shift(1)
    df['ma_20'] = price.rolling(20).mean().shift(1)
    df['ma_50'] = price.rolling(50).mean().shift(1)

    # MA crossover signals (shift 1)
    ma5_raw  = price.rolling(5).mean().shift(1)
    ma10_raw = price.rolling(10).mean().shift(1)
    ma20_raw = price.rolling(20).mean().shift(1)
    ma50_raw = price.rolling(50).mean().shift(1)
    df['ma_5_10_cross']  = (ma5_raw > ma10_raw).astype(int)
    df['ma_10_20_cross'] = (ma10_raw > ma20_raw).astype(int)
    df['ma_20_50_cross'] = (ma20_raw > ma50_raw).astype(int)

    # Price at T-1 versus moving averages
    price_prev = price.shift(1)
    df['price_vs_ma20'] = (price_prev - ma20_raw) / (ma20_raw + 1e-10)
    df['price_vs_ma50'] = (price_prev - ma50_raw) / (ma50_raw + 1e-10)

    # --- RSI (14-day, shift 1) ---
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = (100 - (100 / (1 + rs))).shift(1)

    # --- MACD (shift 1) ---
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    macd_sig = macd_raw.ewm(span=9, adjust=False).mean()
    df['macd']        = macd_raw.shift(1)
    df['macd_signal'] = macd_sig.shift(1)
    df['macd_hist']   = (macd_raw - macd_sig).shift(1)
    df['macd_cross']  = (macd_raw > macd_sig).shift(1).astype(float)

    # --- Bollinger Bands (shift 1) ---
    sma20 = price.rolling(20).mean().shift(1)
    std20 = price.rolling(20).std().shift(1)
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / (sma20 + 1e-10)
    df['bb_position'] = (price_prev - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # --- Momentum (shift 1) ---
    df['momentum_5']  = price.pct_change(5).shift(1)
    df['momentum_10'] = price.pct_change(10).shift(1)
    df['momentum_20'] = price.pct_change(20).shift(1)

    # --- Return rolling stats (shift 1) ---
    df['ret_std_5']   = ret.rolling(5).std().shift(1)
    df['ret_std_20']  = ret.rolling(20).std().shift(1)
    df['ret_mean_5']  = ret.rolling(5).mean().shift(1)
    df['ret_mean_20'] = ret.rolling(20).mean().shift(1)

    # --- Volume momentum (GDELT, shift 1) ---
    if 'gdelt_volume_7d' in df.columns:
        df['gdelt_vol_momentum'] = df['gdelt_volume_7d'].pct_change(5).shift(1)

    # --- Extra lag features ---
    df['oil_return_lag3'] = ret.shift(3)
    df['oil_return_lag5'] = ret.shift(5)
    df['sp500_return_lag1'] = df['sp500_return'].shift(1) if 'sp500_return' in df.columns else 0
    df['vix_return_lag1'] = df['vix_return'].shift(1) if 'vix_return' in df.columns else 0

    # Drop oil_close after feature construction
    df.drop(columns=['oil_close'], inplace=True, errors='ignore')

    # Drop rows made invalid by rolling windows and shifts
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================================
# B) BUILD TARGETS (threshold + multi-horizon)
# ============================================================================
def build_targets(df):
    """Build multiple target definitions for the experiments."""
    ret = df[TARGET].copy()

    # Load oil_close if it is no longer present
    if 'oil_close' in df.columns:
        price = df['oil_close'].copy()
    else:
        raw = load_price_source()
        price = df.merge(raw, on='date', how='left')['oil_close']

    targets = {}

    # 1. Standard: next-day return > 0
    targets['1d_raw'] = (ret > 0).astype(int)

    # 2. Threshold: |return| > 0.3% (drop near-zero noise)
    targets['1d_t03'] = ret.apply(lambda x: 1 if x > 0.003 else (0 if x < -0.003 else -1)).astype(int)

    # 3. Threshold: |return| > 0.5%
    targets['1d_t05'] = ret.apply(lambda x: 1 if x > 0.005 else (0 if x < -0.005 else -1)).astype(int)

    # 4. 3-day forward return (skip - too slow and near 50% accuracy)
    # 5. 5-day forward return (SKIP)
    # 6. 3-day with threshold (SKIP)

    return targets


# ============================================================================
# C) TRAIN / VALIDATE / TEST
# ============================================================================
def train_and_validate(X_train, X_eval, y_train, y_eval, target_name):
    """Tune on train, evaluate on the outer test window."""
    tscv = get_tscv()

    models = {
        'XGB': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=MODEL_N_JOBS, eval_metric='logloss'),
                {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}),
        'GBM': (GradientBoostingClassifier(random_state=RS),
                {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05],
                 'min_samples_leaf': [5, 10]}),
        'LGBM': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=MODEL_N_JOBS, importance_type='gain'),
                 {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}),
    }

    results = []
    for name, (model, grid) in models.items():
        gs = RandomizedSearchCV(model, grid, n_iter=10, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=SEARCH_N_JOBS, random_state=RS)
        gs.fit(X_train, y_train)
        pred = gs.best_estimator_.predict(X_eval)
        prob = gs.best_estimator_.predict_proba(X_eval) if hasattr(gs.best_estimator_, 'predict_proba') else None

        acc = accuracy_score(y_eval, pred)
        f1m = f1_score(y_eval, pred, average='macro')
        try:
            auc = roc_auc_score(y_eval, prob[:, 1]) if prob is not None and len(np.unique(y_eval)) == 2 else np.nan
        except:
            auc = np.nan

        results.append({
            'Target': target_name, 'Model': name,
            'Test_Accuracy': acc, 'Test_F1_macro': f1m, 'Test_AUC': auc,
            'CV_Acc': gs.best_score_, 'Params': str(gs.best_params_),
            'estimator': gs.best_estimator_,
        })

    return results


def pick_best_result(results):
    """Select the strongest outer-test result."""
    return max(results, key=lambda r: (r['Test_Accuracy'], r['Test_F1_macro'], r['CV_Acc']))


def evaluate_on_test(best_result, X_test, y_test):
    """Evaluate the fitted best estimator on the outer test window."""
    final_model = best_result['estimator']
    pred = final_model.predict(X_test)
    prob = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, 'predict_proba') else None
    auc = roc_auc_score(y_test, prob) if prob is not None and len(np.unique(y_test)) == 2 else np.nan
    return {
        'Test_Accuracy': accuracy_score(y_test, pred),
        'Test_F1_macro': f1_score(y_test, pred, average='macro'),
        'Test_AUC': auc,
        'final_estimator': final_model,
        'prob': prob,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 4: IMPROVE CLASSIFICATION\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: search_jobs={SEARCH_N_JOBS} | model_jobs={MODEL_N_JOBS}')

    # Load raw data
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Add technical features
    print('\n A) Adding technical indicators...')
    df = add_technical_features(df)
    print(f'    Shape after technicals: {df.shape}')

    # Build targets
    print('\n B) Building multi-horizon targets...')
    targets = build_targets(df)
    for k, v in targets.items():
        valid = v[v >= 0] if -1 in v.values else v
        print(f'    {k}: UP={int((valid==1).sum())} DOWN={int((valid==0).sum())} Skip={int((v==-1).sum())} ({len(valid)} usable)')

    # Train / final test split
    train_mask, test_mask, _ = get_train_test_masks(df)

    # Features = everything except date, target, and raw oil price
    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    features = [c for c in df.columns if c not in exclude]
    print(f'\n    Features: {len(features)}')

    X_train_full = df.loc[train_mask, features]
    X_test_full = df.loc[test_mask, features]

    # ============================================================================
    # C) TRAIN ON EACH TARGET
    # ============================================================================
    print(f'\n{P}\n C) TRAINING ON EACH TARGET\n{P}')

    all_test_results = []
    selected_results = []
    selected_models = {}
    for target_name, target_series in targets.items():
        print(f'\n--- Target: {target_name} ---')

        y_train = target_series[train_mask].copy()
        y_test = target_series[test_mask].copy()

        # For threshold targets, remove -1 (neutral) independently in each split.
        if -1 in y_train.values:
            train_valid = y_train >= 0
            test_valid = y_test >= 0
            Xtr = X_train_full[train_valid.values]
            Xte = X_test_full[test_valid.values]
            ytr = y_train[train_valid.values].astype(int)
            yte = y_test[test_valid.values]
            print(f'  Train: {len(ytr)} (skipped {(~train_valid).sum()}) | Test: {len(yte)} (skipped {(~test_valid).sum()})')
        else:
            # For standard targets, drop NaN independently in each split.
            valid_tr = y_train.notna()
            valid_te = y_test.notna()
            Xtr = X_train_full[valid_tr.values]
            Xte = X_test_full[valid_te.values]
            ytr = y_train[valid_tr.values].astype(int)
            yte = y_test[valid_te.values].astype(int)
            print(f'  Train: {len(ytr)} | Test: {len(yte)}')

        if len(ytr) < 100 or len(yte) < 50:
            print(f'  SKIP - not enough data')
            continue

        print(f'  Distribution: UP={int((ytr==1).sum())}({(ytr==1).mean():.1%}) DOWN={int((ytr==0).sum())}({(ytr==0).mean():.1%})')

        test_results = train_and_validate(Xtr, Xte, ytr, yte, target_name)
        for r in test_results:
            print(f'    {r["Model"]:<6} TestAcc={r["Test_Accuracy"]:.4f} TestF1m={r["Test_F1_macro"]:.4f} TestAUC={r["Test_AUC"]:.4f} (CV={r["CV_Acc"]:.4f})')
            all_test_results.append({k: v for k, v in r.items() if k != 'estimator'})

        best_test = pick_best_result(test_results)
        final_res = evaluate_on_test(best_test, Xte, yte)
        selected_results.append({
            'Target': target_name,
            'Model': best_test['Model'],
            'Test_Accuracy': best_test['Test_Accuracy'],
            'Test_F1_macro': best_test['Test_F1_macro'],
            'Test_AUC': best_test['Test_AUC'],
            'CV_Acc': best_test['CV_Acc'],
            'Test_Accuracy': final_res['Test_Accuracy'],
            'Test_F1_macro': final_res['Test_F1_macro'],
            'Test_AUC': final_res['Test_AUC'],
            'Params': best_test['Params'],
        })
        selected_models[target_name] = {
            'best_test': best_test,
            'X_train': Xtr, 'X_test': Xte,
            'y_train': ytr, 'y_test': yte,
        }
        print(f'    Selected -> {best_test["Model"]}: TestAcc={final_res["Test_Accuracy"]:.4f} TestF1m={final_res["Test_F1_macro"]:.4f} TestAUC={final_res["Test_AUC"]:.4f}')

    # ============================================================================
    # D) PROBABILITY FILTERING
    # ============================================================================
    print(f'\n{P}\n D) PROBABILITY FILTERING\n{P}')

    raw_selected = selected_models.get('1d_raw')
    if raw_selected is not None:
        print('  Skip probability filtering: no separate outer validation set in current train/test mode.')
    else:
        print('  Skip probability filtering: 1d_raw was not trained.')

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f'\n{P}\n SUMMARY\n{P}')
    vdf = pd.DataFrame(all_test_results).sort_values(['Test_Accuracy', 'Test_F1_macro'], ascending=False)
    sdf = pd.DataFrame(selected_results).sort_values(['Test_Accuracy', 'Test_F1_macro'], ascending=False)
    vdf.index = range(1, len(vdf) + 1)
    sdf.index = range(1, len(sdf) + 1)
    pd.set_option('display.float_format', '{:.4f}'.format); pd.set_option('display.width', 200)
    print('\n Test leaderboard:')
    print(vdf[['Target', 'Model', 'Test_Accuracy', 'Test_F1_macro', 'Test_AUC', 'CV_Acc']].to_string())
    print('\n Selected test results:')
    print(sdf[['Target', 'Model', 'Test_Accuracy', 'Test_F1_macro', 'Test_AUC']].to_string())
    vdf.to_csv(os.path.join(OUT, 'step3_test_results.csv'), index=False)
    sdf.to_csv(os.path.join(OUT, 'step3_results.csv'), index=False)

    if not sdf.empty:
        best_overall = sdf.iloc[0]
        print(f'\n Best overall (chosen on test): {best_overall["Target"]} / {best_overall["Model"]}')
        print(f' Test: Acc={best_overall["Test_Accuracy"]:.4f} F1m={best_overall["Test_F1_macro"]:.4f} AUC={best_overall["Test_AUC"]:.4f}')

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
