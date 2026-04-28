"""
STEP 4: Permutation Importance Feature Selection

This step uses sklearn's permutation_importance to rank features
by how much accuracy drops when each feature is shuffled.

Pipeline:
  1. Train a LGBM proxy model on inner train split
  2. Compute permutation importance on inner eval split
  3. Compare subsets: ALL, TOP_20, TOP_15, TOP_10, TOP_5
  4. Final train/test with the best subset

Target:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Models:
  - XGBClassifier, GradientBoostingClassifier, LGBMClassifier

Outputs:
  - results/step4_perm_importance.csv
  - results/step4_selected_features.csv
  - results/step4_results.csv
  - results/step4_set_comparison.csv

Usage:
  python ml/classification/final/step4_group_selection_2.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')

from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DATA_PATH, DROP_COLS, OUT_DIR, RANDOM_STATE as RS, TARGET, TARGET_DATE_COL, get_tscv, get_train_test_masks, set_global_seed

from metrics import evaluate, get_scores, METRIC_COLS, SORT_COLS

P = '=' * 90
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))
MODEL_N_JOBS = max(1, int(os.getenv('MODEL_N_JOBS', str(min(12, max(1, CPU_COUNT // 4))))))
PERM_REPEATS = max(1, int(os.getenv('STEP4_PERM_REPEATS', '10')))
N_ITER = max(1, int(os.getenv('STEP4_N_ITER', '15')))
TOPN_LIST = [
    int(x.strip()) for x in os.getenv('STEP4_TOPN_LIST', '5,10,15,20,25').split(',') if x.strip()
]


def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 4: PERMUTATION IMPORTANCE FEATURE SELECTION\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: search_jobs={SEARCH_N_JOBS} | model_jobs={MODEL_N_JOBS}')
    print(f'  Permutation repeats: {PERM_REPEATS}')
    print(f'  Top-N sets: {TOPN_LIST}')

    # Load + split
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    features = [c for c in df.columns if c not in DROP_COLS and c != TARGET]

    train_mask, test_mask, _ = get_train_test_masks(df)

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    print(f'  Features: {len(features)} | Train: {len(X_train)} | Test: {len(X_test)}')

    # ============================================================================
    # A) PERMUTATION IMPORTANCE
    # ============================================================================
    print(f'\n{P}\n A) PERMUTATION IMPORTANCE\n{P}')

    # Use last TimeSeriesSplit fold as inner eval
    tscv = get_tscv()
    perm_train_idx, perm_eval_idx = list(tscv.split(X_train, y_train))[-1]
    X_perm_train = X_train.iloc[perm_train_idx]
    y_perm_train = y_train.iloc[perm_train_idx]
    X_perm_eval = X_train.iloc[perm_eval_idx]
    y_perm_eval = y_train.iloc[perm_eval_idx]

    print(f'  Inner split: train={len(X_perm_train)} | eval={len(X_perm_eval)}')

    # Train LGBM proxy
    proxy = LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=MODEL_N_JOBS,
                           n_estimators=300, max_depth=5, learning_rate=0.05)
    proxy.fit(X_perm_train, y_perm_train)

    proxy_acc = (proxy.predict(X_perm_eval) == y_perm_eval).mean()
    print(f'  Proxy model accuracy on inner eval: {proxy_acc:.4f}')

    # sklearn permutation_importance
    print(f'  Computing permutation importance ({PERM_REPEATS} repeats)...')
    t0 = time.time()
    perm = permutation_importance(
        proxy, X_perm_eval, y_perm_eval,
        n_repeats=PERM_REPEATS, random_state=RS,
        n_jobs=SEARCH_N_JOBS, scoring='accuracy',
    )
    print(f'  Done ({time.time()-t0:.1f}s)')

    perm_df = pd.DataFrame({
        'feature': features,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

    # Print all features ranked
    print(f'\n  Permutation Importance Ranking:')
    print(f'  {"#":<4} {"Feature":<40} {"Mean":>10} {"Std":>10}')
    print(f'  {"-"*68}')
    for i, r in perm_df.iterrows():
        marker = '  *' if r.importance_mean > 0 else '   '
        print(f'  {i+1:<4} {r.feature:<40} {r.importance_mean:>10.5f} {r.importance_std:>10.5f}{marker}')

    n_positive = (perm_df['importance_mean'] > 0).sum()
    print(f'\n  Features with positive importance: {n_positive} / {len(features)}')

    # ============================================================================
    # B) COMPARE FEATURE SUBSETS
    # ============================================================================
    print(f'\n{P}\n B) COMPARE FEATURE SUBSETS\n{P}')

    sets = {f'ALL_{len(features)}': features}

    for n in TOPN_LIST:
        if n <= len(features):
            topn = perm_df.head(n)['feature'].tolist()
            sets[f'PERM_TOP_{n}'] = topn

    # Also add "positive only" set
    pos_features = perm_df[perm_df['importance_mean'] > 0]['feature'].tolist()
    if 0 < len(pos_features) < len(features):
        sets[f'PERM_POS_{len(pos_features)}'] = pos_features

    print(f'\n  {"Set":<25} {"N":>4} {"LGBM_CV":>10} {"XGB_CV":>10} {"GBM_CV":>10}')
    print(f'  {"-"*64}')

    all_results = []
    for set_name, feats in sets.items():
        row = {'Set': set_name, 'N': len(feats)}
        for mname, model in [
            ('LGBM', LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=MODEL_N_JOBS,
                                     n_estimators=300, max_depth=5, learning_rate=0.05)),
            ('XGB', XGBClassifier(random_state=RS, verbosity=0, n_jobs=MODEL_N_JOBS, eval_metric='logloss',
                                   n_estimators=300, max_depth=5, learning_rate=0.05)),
            ('GBM', GradientBoostingClassifier(random_state=RS,
                                                n_estimators=300, max_depth=5, learning_rate=0.05)),
        ]:
            cv_scores = cross_val_score(model, X_train[feats], y_train, cv=tscv, scoring='accuracy')
            row[f'{mname}_CV'] = cv_scores.mean()
        all_results.append(row)
        print(f'  {set_name:<25} {len(feats):>4} {row["LGBM_CV"]:>10.4f} {row["XGB_CV"]:>10.4f} {row["GBM_CV"]:>10.4f}')

    # ============================================================================
    # C) FINAL TRAIN ON BEST SET
    # ============================================================================
    res_df = pd.DataFrame(all_results)
    res_df['best_acc'] = res_df[['LGBM_CV', 'XGB_CV', 'GBM_CV']].max(axis=1)
    best_set_name = res_df.loc[res_df['best_acc'].idxmax(), 'Set']
    best_feats = sets[best_set_name]

    print(f'\n{P}\n C) FINAL TRAIN ON {best_set_name} ({len(best_feats)} features)\n{P}')

    final_models = {
        'XGB': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
                {'n_estimators': [300, 500], 'max_depth': [3, 5, 7],
                 'learning_rate': [0.01, 0.03, 0.05], 'reg_alpha': [0, 0.1]}),
        'GBM': (GradientBoostingClassifier(random_state=RS),
                {'n_estimators': [300, 500], 'max_depth': [3, 5, 7],
                 'learning_rate': [0.01, 0.03, 0.05], 'min_samples_leaf': [5, 10]}),
        'LGBM': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
                 {'n_estimators': [300, 500], 'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31]}),
    }

    final_results = []
    for name, (model, grid) in final_models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        gs = RandomizedSearchCV(model, grid, n_iter=N_ITER, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=SEARCH_N_JOBS, random_state=RS)
        gs.fit(X_train[best_feats], y_train)
        pred = gs.best_estimator_.predict(X_test[best_feats])
        score = get_scores(gs.best_estimator_, X_test[best_feats])
        metrics = evaluate(name, y_test.values, pred, score)
        elapsed = round(time.time() - t0, 1)
        final_results.append({
            **metrics,
            'CV_Acc': gs.best_score_,
            'Time_s': elapsed,
        })
        print(f'  Best: {gs.best_params_}')
        print(f'  Acc={metrics["Accuracy"]:.4f} F1m={metrics["F1_macro"]:.4f} AUC={metrics["AUC"]:.4f} (CV={gs.best_score_:.4f}, {elapsed}s)')

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f'\n{P}\n SUMMARY\n{P}')

    fdf = pd.DataFrame(final_results).sort_values(SORT_COLS, ascending=False)
    best = fdf.iloc[0]
    print(fdf[METRIC_COLS + ['CV_Acc', 'Time_s']].to_string(index=False))

    print(f'\n  Best set: {best_set_name} ({len(best_feats)} features)')
    print(f'  Best model: {best["Model"]}')
    print(f'  Test: Acc={best["Accuracy"]:.4f} F1m={best["F1_macro"]:.4f} AUC={best["AUC"]:.4f}')

    # Save
    perm_df.to_csv(os.path.join(OUT, 'step4_perm_importance.csv'), index=False)
    pd.DataFrame({'feature': best_feats}).to_csv(os.path.join(OUT, 'step4_selected_features.csv'), index=False)
    fdf.to_csv(os.path.join(OUT, 'step4_results.csv'), index=False)
    res_df.to_csv(os.path.join(OUT, 'step4_set_comparison.csv'), index=False)

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
