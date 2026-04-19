"""
STEP 5: Smart Feature Selection

This step tries a more structured alternative to naive ranking.
Instead of only keeping the top-scored features, it groups correlated
features into clusters and keeps the strongest feature from each group.

Goal of this step:
  - Reduce multicollinearity inside the 81-feature technical dataset
  - Use permutation importance to estimate feature usefulness
  - Compare clustered feature sets against naive top-N selection

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base features plus no-leakage technical features from step 3
  - Candidate set is the same 81-feature pool used in step 4

Methods and models used in this file:
  - Spearman correlation clustering
  - Permutation importance
  - LGBMClassifier for permutation-importance estimation and some set comparisons
  - XGBClassifier, GradientBoostingClassifier, and LGBMClassifier for final comparison

Model selection:
  - Cluster construction from pairwise feature correlation
  - One best feature kept per cluster
  - Final model tuning with RandomizedSearchCV and TimeSeriesSplit

Outputs:
  - Console cluster summary and model comparison
  - results/step5_selected_features.csv
  - results/step5_perm_importance.csv
  - results/step5_results.csv
  - results/step5_set_comparison.csv

Usage:
  python ml/classification/step5_smart_selection.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DATA_PATH, OUT_DIR, RANDOM_STATE as RS, TARGET, TARGET_DATE_COL, get_tscv, get_train_test_masks, set_global_seed
from step3_technical_improve import add_technical_features

P = '=' * 90
os.makedirs(OUT_DIR, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))
MODEL_N_JOBS = max(1, int(os.getenv('MODEL_N_JOBS', str(min(12, max(1, CPU_COUNT // 4))))))
STEP5_PERM_REPEATS = max(1, int(os.getenv('STEP5_PERM_REPEATS', '5')))
STEP5_N_ITER = max(1, int(os.getenv('STEP5_N_ITER', '15')))
STEP5_TOPN_LIST = [
    int(x.strip()) for x in os.getenv('STEP5_TOPN_LIST', '10,15,20,25').split(',') if x.strip()
]


def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 5: SMART FEATURE SELECTION\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: search_jobs={SEARCH_N_JOBS} | model_jobs={MODEL_N_JOBS}')
    print(f'  Permutation repeats: {STEP5_PERM_REPEATS}')
    print(f'  Final randomized-search iterations: {STEP5_N_ITER}')
    print(f'  Extra top-N sets: {STEP5_TOPN_LIST}')

    # Load + technicals (shifted)
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    features = [c for c in df.columns if c not in exclude]

    train_mask, test_mask, _ = get_train_test_masks(df)

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    print(f'  Features: {len(features)} | Train: {len(X_train)} | Test: {len(X_test)}')

    # ============================================================================
    # STEP A: Correlation Clustering
    # ============================================================================
    print(f'\n{P}\n A) CORRELATION CLUSTERING\n{P}')

    # Spearman correlation -> distance matrix
    corr = X_train.corr(method='spearman').abs().fillna(0.0)
    dist_arr = (1 - corr).to_numpy(copy=True).astype(float)
    np.fill_diagonal(dist_arr, 0)
    distance = pd.DataFrame(dist_arr, index=corr.index, columns=corr.columns)

    # Hierarchical clustering
    dist_condensed = squareform(dist_arr, checks=False)
    Z = linkage(dist_condensed, method='average')

    # Try multiple thresholds
    print(f'\n {"Threshold":<12} {"N_clusters":>12} {"Max_size":>10} {"Singletons":>12}')
    print(f' {"-"*50}')
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        labels = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(set(labels))
        sizes = pd.Series(labels).value_counts()
        print(f' {t:<12.1f} {n_clusters:>12} {sizes.max():>10} {(sizes==1).sum():>12}')

    # Choose threshold = 0.3 (features with |rho| > 0.7 tend to cluster)
    THRESH = 0.3
    labels = fcluster(Z, t=THRESH, criterion='distance')
    cluster_df = pd.DataFrame({'feature': features, 'cluster': labels})
    n_clusters = len(set(labels))
    print(f'\n Choose threshold = {THRESH} -> {n_clusters} clusters')

    # In clusters
    print(f'\n Clusters with more than 1 feature (multicollinearity):')
    for cid in sorted(set(labels)):
        members = cluster_df[cluster_df.cluster == cid]['feature'].tolist()
        if len(members) > 1:
            # Compute max |rho| within the cluster
            sub_arr = corr.loc[members, members].to_numpy(copy=True).astype(float)
            np.fill_diagonal(sub_arr, 0)
            max_rho = sub_arr.max()
            print(f'   Cluster {cid} ({len(members)} features, max|rho|={max_rho:.3f}):')
            for m in members:
                print(f'     - {m}')

    # ============================================================================
    # STEP B: Permutation Importance
    # ============================================================================
    print(f'\n{P}\n B) PERMUTATION IMPORTANCE\n{P}')

    # Train a quick model to compute permutation importance on an internal
    # train-only split (last TimeSeriesSplit fold).
    tscv = get_tscv()
    perm_train_idx, perm_eval_idx = list(tscv.split(X_train, y_train))[-1]
    X_perm_train = X_train.iloc[perm_train_idx]
    y_perm_train = y_train.iloc[perm_train_idx]
    X_perm_eval = X_train.iloc[perm_eval_idx]
    y_perm_eval = y_train.iloc[perm_eval_idx]

    base_model = LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=MODEL_N_JOBS,
                                 n_estimators=300, max_depth=5, learning_rate=0.05)
    base_model.fit(X_perm_train, y_perm_train)

    print(f'  Computing permutation importance ({STEP5_PERM_REPEATS} repeats)...')
    t0 = time.time()
    perm = permutation_importance(base_model, X_perm_eval, y_perm_eval,
                                  n_repeats=STEP5_PERM_REPEATS, random_state=RS, n_jobs=SEARCH_N_JOBS,
                                  scoring='accuracy')
    print(f'  Done ({time.time()-t0:.1f}s)')

    perm_df = pd.DataFrame({
        'feature': features,
        'perm_mean': perm.importances_mean,
        'perm_std': perm.importances_std,
    }).sort_values('perm_mean', ascending=False)

    print(f'\n Top 20 Permutation Importance:')
    print(f' {"#":<4} {"Feature":<30} {"Perm_mean":>10} {"Perm_std":>10}')
    print(f' {"-"*58}')
    for i, (_, r) in enumerate(perm_df.head(20).iterrows()):
        print(f' {i+1:<4} {r.feature:<30} {r.perm_mean:>10.5f} {r.perm_std:>10.5f}')

    # ============================================================================
    # STEP C: Combine - keep the best feature in each cluster
    # ============================================================================
    print(f'\n{P}\n C) COMBINE: BEST FEATURE PER CLUSTER\n{P}')

    cluster_df['perm_mean'] = cluster_df['feature'].map(perm_df.set_index('feature')['perm_mean'])
    selected = []
    print(f'\n {"Cluster":>8} {"Selected":<30} {"Perm":>8} {"Dropped":<50}')
    print(f' {"-"*100}')

    for cid in sorted(set(labels)):
        members = cluster_df[cluster_df.cluster == cid].sort_values('perm_mean', ascending=False)
        best = members.iloc[0]
        dropped = members.iloc[1:]['feature'].tolist() if len(members) > 1 else []
        selected.append(best['feature'])
        dropped_str = ', '.join(dropped) if dropped else '-'
        print(f' {cid:>8} {best["feature"]:<30} {best["perm_mean"]:>8.5f} {dropped_str:<50}')

    print(f'\n Selected: {len(selected)} features (from {len(features)} -> minus {len(features)-len(selected)})')

    # Also remove features with permutation importance <= 0
    perm_map = perm_df.set_index('feature')['perm_mean']
    selected_positive = [f for f in selected if perm_map.get(f, 0) > 0]
    print(f' After dropping perm <= 0: {len(selected_positive)} features')

    # ============================================================================
    # STEP D: Compare feature sets
    # ============================================================================
    print(f'\n{P}\n D) COMPARE FEATURE SETS\n{P}')

    sets = {
        f'ALL_{len(features)}': features,
        f'CLUSTER_{len(selected)}': selected,
        f'CLUSTER_POS_{len(selected_positive)}': selected_positive,
    }

    # Also add top-N sets from raw permutation importance
    for n in STEP5_TOPN_LIST:
        topn = perm_df.head(n)['feature'].tolist()
        sets[f'PERM_TOP_{n}'] = topn

    print(f'\n {"Set":<25} {"N":>4} {"LGBM_CV":>10} {"XGB_CV":>10} {"GBM_CV":>10}')
    print(f' {"-"*64}')

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
        print(f' {set_name:<25} {len(feats):>4} {row["LGBM_CV"]:>10.4f} {row["XGB_CV"]:>10.4f} {row["GBM_CV"]:>10.4f}')

    # ============================================================================
    # STEP E: Final training with grid search
    # ============================================================================
    # Find the best feature set on train-only CV
    res_df = pd.DataFrame(all_results)
    res_df['best_acc'] = res_df[['LGBM_CV', 'XGB_CV', 'GBM_CV']].max(axis=1)
    best_set_name = res_df.loc[res_df['best_acc'].idxmax(), 'Set']
    best_feats = sets[best_set_name]

    print(f'\n{P}\n E) FINAL TRAIN ON {best_set_name} ({len(best_feats)} features)\n{P}')

    from sklearn.model_selection import RandomizedSearchCV
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
        gs = RandomizedSearchCV(model, grid, n_iter=STEP5_N_ITER, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=SEARCH_N_JOBS, random_state=RS)
        gs.fit(X_train[best_feats], y_train)
        pred = gs.best_estimator_.predict(X_test[best_feats])
        prob = gs.best_estimator_.predict_proba(X_test[best_feats])[:, 1]
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average='macro')
        auc = roc_auc_score(y_test, prob)
        elapsed = round(time.time() - t0, 1)
        final_results.append({'Model': name, 'Test_Accuracy': acc, 'Test_F1_macro': f1m, 'Test_AUC': auc,
                              'CV_Acc': gs.best_score_, 'Time_s': elapsed, 'Estimator': gs.best_estimator_})
        print(f'  Best: {gs.best_params_}')
        print(f'  TestAcc={acc:.4f} TestF1m={f1m:.4f} TestAUC={auc:.4f} (CV={gs.best_score_:.4f}, {elapsed}s)')

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f'\n{P}\n SUMMARY\n{P}')

    fdf = pd.DataFrame(final_results).sort_values(['Test_Accuracy', 'Test_F1_macro'], ascending=False)
    best = fdf.iloc[0]
    print(fdf.drop(columns=['Estimator']).to_string(index=False))
    print(f'\n Pipeline:')
    print(f'   81 features')
    print(f'   -> Correlation clustering (threshold={THRESH}, |rho|>0.7 grouped)')
    print(f'   -> {n_clusters} clusters')
    print(f'   -> Permutation importance keeps 1 best feature per cluster')
    print(f'   -> {best_set_name}: {len(best_feats)} features')
    print(f'   -> {best["Model"]}: TestAcc={best["Test_Accuracy"]:.4f}')
    print(f'   -> Test: Acc={best["Test_Accuracy"]:.4f} F1m={best["Test_F1_macro"]:.4f} AUC={best["Test_AUC"]:.4f}')
    print(f'\n Comparison:')
    print(f'   Baseline (42 features, no technicals):   Acc=0.5274')
    print(f'   Step4 (81 features, no selection):        Acc=0.5530')
    print(f'   Step5 (naive MI+Spearman selection):      Acc=0.5619')
    print(f'   Step5 (cluster+perm selection, test):      Acc={best["Test_Accuracy"]:.4f}')

    # Save
    selected_df = pd.DataFrame({'feature': best_feats})
    selected_df.to_csv(os.path.join(OUT_DIR, 'step5_selected_features.csv'), index=False)
    fdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT_DIR, 'step5_results.csv'), index=False)
    perm_df.to_csv(os.path.join(OUT_DIR, 'step5_perm_importance.csv'), index=False)
    res_df.to_csv(os.path.join(OUT_DIR, 'step5_set_comparison.csv'), index=False)

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
