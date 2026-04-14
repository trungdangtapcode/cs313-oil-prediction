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
  - results/step3_validation_results.csv
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

from config import get_tscv, get_train_val_test_masks, RANDOM_STATE as RS, DATA_PATH, TARGET, TARGET_DATE_COL, set_global_seed

P = '=' * 90

# ============================================================================
# A) ADD TECHNICAL INDICATORS
# ============================================================================
def add_technical_features(df):
    """Add technical indicators derived from oil_close in the base dataset."""

    # dataset_step4 already contains oil_close in most cases
    if 'oil_close' not in df.columns:
        raw = pd.read_csv(DATA_PATH, parse_dates=['date'])[['date', 'oil_close']].sort_values('date').reset_index(drop=True)
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
        raw = pd.read_csv(DATA_PATH, parse_dates=['date'])[['date', 'oil_close']].sort_values('date').reset_index(drop=True)
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
def train_and_validate(X_train, X_val, y_train, y_val, target_name):
    """Tune on train, evaluate on validation."""
    tscv = get_tscv()

    models = {
        'XGB': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
                {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}),
        'GBM': (GradientBoostingClassifier(random_state=RS),
                {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05],
                 'min_samples_leaf': [5, 10]}),
        'LGBM': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
                 {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.05]}),
    }

    results = []
    for name, (model, grid) in models.items():
        gs = RandomizedSearchCV(model, grid, n_iter=10, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=1, random_state=RS)
        gs.fit(X_train, y_train)
        pred = gs.best_estimator_.predict(X_val)
        prob = gs.best_estimator_.predict_proba(X_val) if hasattr(gs.best_estimator_, 'predict_proba') else None

        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average='macro')
        try:
            auc = roc_auc_score(y_val, prob[:, 1]) if prob is not None and len(np.unique(y_val)) == 2 else np.nan
        except:
            auc = np.nan

        results.append({
            'Target': target_name, 'Model': name,
            'Val_Accuracy': acc, 'Val_F1_macro': f1m, 'Val_AUC': auc,
            'CV_Acc': gs.best_score_, 'Params': str(gs.best_params_),
            'estimator': gs.best_estimator_,
        })

    return results


def pick_best_result(results):
    """Select the strongest validation result without touching final test."""
    return max(results, key=lambda r: (r['Val_Accuracy'], r['Val_F1_macro'], r['CV_Acc']))


def refit_on_train_val(best_result, X_train, X_val, y_train, y_val, X_test, y_test):
    """Refit the selected model on train+val and evaluate once on final test."""
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    final_model = clone(best_result['estimator'])
    final_model.fit(X_train_val, y_train_val)

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

    # Train / validation / final test split
    train_mask, val_mask, test_mask, _ = get_train_val_test_masks(df)

    # Features = everything except date, target, and raw oil price
    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    features = [c for c in df.columns if c not in exclude]
    print(f'\n    Features: {len(features)}')

    X_train_full = df.loc[train_mask, features]
    X_val_full = df.loc[val_mask, features]
    X_test_full = df.loc[test_mask, features]

    # ============================================================================
    # C) TRAIN ON EACH TARGET
    # ============================================================================
    print(f'\n{P}\n C) TRAINING ON EACH TARGET\n{P}')

    all_val_results = []
    selected_results = []
    selected_models = {}
    for target_name, target_series in targets.items():
        print(f'\n--- Target: {target_name} ---')

        y_train = target_series[train_mask].copy()
        y_val = target_series[val_mask].copy()
        y_test = target_series[test_mask].copy()

        # For threshold targets, remove -1 (neutral) independently in each split.
        if -1 in y_train.values:
            train_valid = y_train >= 0
            val_valid = y_val >= 0
            test_valid = y_test >= 0
            Xtr = X_train_full[train_valid.values]
            Xva = X_val_full[val_valid.values]
            Xte = X_test_full[test_valid.values]
            ytr = y_train[train_valid.values].astype(int)
            yva = y_val[val_valid.values].astype(int)
            yte = y_test[test_valid.values]
            print(f'  Train: {len(ytr)} (skipped {(~train_valid).sum()}) | Val: {len(yva)} (skipped {(~val_valid).sum()}) | Test: {len(yte)} (skipped {(~test_valid).sum()})')
        else:
            # For standard targets, drop NaN independently in each split.
            valid_tr = y_train.notna()
            valid_va = y_val.notna()
            valid_te = y_test.notna()
            Xtr = X_train_full[valid_tr.values]
            Xva = X_val_full[valid_va.values]
            Xte = X_test_full[valid_te.values]
            ytr = y_train[valid_tr.values].astype(int)
            yva = y_val[valid_va.values].astype(int)
            yte = y_test[valid_te.values].astype(int)
            print(f'  Train: {len(ytr)} | Val: {len(yva)} | Test: {len(yte)}')

        if len(ytr) < 100 or len(yva) < 50 or len(yte) < 50:
            print(f'  SKIP - not enough data')
            continue

        print(f'  Distribution: UP={int((ytr==1).sum())}({(ytr==1).mean():.1%}) DOWN={int((ytr==0).sum())}({(ytr==0).mean():.1%})')

        val_results = train_and_validate(Xtr, Xva, ytr, yva, target_name)
        for r in val_results:
            print(f'    {r["Model"]:<6} ValAcc={r["Val_Accuracy"]:.4f} ValF1m={r["Val_F1_macro"]:.4f} ValAUC={r["Val_AUC"]:.4f} (CV={r["CV_Acc"]:.4f})')
            all_val_results.append({k: v for k, v in r.items() if k != 'estimator'})

        best_val = pick_best_result(val_results)
        final_res = refit_on_train_val(best_val, Xtr, Xva, ytr, yva, Xte, yte)
        selected_results.append({
            'Target': target_name,
            'Model': best_val['Model'],
            'Val_Accuracy': best_val['Val_Accuracy'],
            'Val_F1_macro': best_val['Val_F1_macro'],
            'Val_AUC': best_val['Val_AUC'],
            'CV_Acc': best_val['CV_Acc'],
            'Test_Accuracy': final_res['Test_Accuracy'],
            'Test_F1_macro': final_res['Test_F1_macro'],
            'Test_AUC': final_res['Test_AUC'],
            'Params': best_val['Params'],
        })
        selected_models[target_name] = {
            'best_val': best_val,
            'X_train': Xtr, 'X_val': Xva, 'X_test': Xte,
            'y_train': ytr, 'y_val': yva, 'y_test': yte,
        }
        print(f'    Selected -> {best_val["Model"]}: TestAcc={final_res["Test_Accuracy"]:.4f} TestF1m={final_res["Test_F1_macro"]:.4f} TestAUC={final_res["Test_AUC"]:.4f}')

    # ============================================================================
    # D) PROBABILITY FILTERING
    # ============================================================================
    print(f'\n{P}\n D) PROBABILITY FILTERING (best model, 1d_raw target)\n{P}')

    raw_selected = selected_models.get('1d_raw')
    if raw_selected is not None:
        train_model = raw_selected['best_val']['estimator']
        val_probs = train_model.predict_proba(raw_selected['X_val'])[:, 1]
        y_val_true = raw_selected['y_val'].values

        print(f'\n {"Threshold":<12} {"Val_traded":>10} {"Val_Acc":>10} {"Val_Acc_tr":>12} {"Coverage":>10}')
        print(f' {"-"*60}')
        best_threshold = 0.50
        best_val_acc = -np.inf
        best_val_cov = -np.inf
        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            confident = (val_probs > thresh) | (val_probs < (1 - thresh))
            if confident.sum() < 10:
                continue
            pred_filt = (val_probs[confident] > 0.5).astype(int)
            acc_filt = accuracy_score(y_val_true[confident], pred_filt)
            coverage = confident.mean()
            pred_all = (val_probs > 0.5).astype(int)
            acc_all = accuracy_score(y_val_true, pred_all)
            print(f' {thresh:<12.2f} {int(confident.sum()):>10} {acc_all:>10.4f} {acc_filt:>12.4f} {coverage:>10.1%}')
            if acc_filt > best_val_acc or (acc_filt == best_val_acc and coverage > best_val_cov):
                best_threshold = thresh
                best_val_acc = acc_filt
                best_val_cov = coverage

        X_train_val_raw = pd.concat([raw_selected['X_train'], raw_selected['X_val']])
        y_train_val_raw = pd.concat([raw_selected['y_train'], raw_selected['y_val']])
        final_model = clone(train_model)
        final_model.fit(X_train_val_raw, y_train_val_raw)
        test_probs = final_model.predict_proba(raw_selected['X_test'])[:, 1]
        y_test_true = raw_selected['y_test'].values
        confident_test = (test_probs > best_threshold) | (test_probs < (1 - best_threshold))
        if confident_test.sum() > 0:
            pred_test_filt = (test_probs[confident_test] > 0.5).astype(int)
            acc_test_filt = accuracy_score(y_test_true[confident_test], pred_test_filt)
            coverage_test = confident_test.mean()
        else:
            acc_test_filt = np.nan
            coverage_test = 0.0
        pred_test_all = (test_probs > 0.5).astype(int)
        acc_test_all = accuracy_score(y_test_true, pred_test_all)
        print(f'\n Selected threshold on validation: {best_threshold:.2f}')
        print(f' Holdout test -> Acc={acc_test_all:.4f} | Acc_traded={acc_test_filt:.4f} | Coverage={coverage_test:.1%}')
    else:
        print('  Skip probability filtering: 1d_raw was not trained.')

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f'\n{P}\n SUMMARY\n{P}')
    vdf = pd.DataFrame(all_val_results).sort_values(['Val_Accuracy', 'Val_F1_macro'], ascending=False)
    sdf = pd.DataFrame(selected_results).sort_values(['Val_Accuracy', 'Val_F1_macro'], ascending=False)
    vdf.index = range(1, len(vdf) + 1)
    sdf.index = range(1, len(sdf) + 1)
    pd.set_option('display.float_format', '{:.4f}'.format); pd.set_option('display.width', 200)
    print('\n Validation leaderboard:')
    print(vdf[['Target', 'Model', 'Val_Accuracy', 'Val_F1_macro', 'Val_AUC', 'CV_Acc']].to_string())
    print('\n Final holdout results (selected on validation):')
    print(sdf[['Target', 'Model', 'Val_Accuracy', 'Val_F1_macro', 'Test_Accuracy', 'Test_F1_macro', 'Test_AUC']].to_string())
    vdf.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'step3_validation_results.csv'), index=False)
    sdf.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'step3_results.csv'), index=False)

    if not sdf.empty:
        best_overall = sdf.iloc[0]
        print(f'\n Best overall (chosen on validation): {best_overall["Target"]} / {best_overall["Model"]}')
        print(f' Validation: Acc={best_overall["Val_Accuracy"]:.4f} F1m={best_overall["Val_F1_macro"]:.4f}')
        print(f' Holdout test: Acc={best_overall["Test_Accuracy"]:.4f} F1m={best_overall["Test_F1_macro"]:.4f} AUC={best_overall["Test_AUC"]:.4f}')

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
