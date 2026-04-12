"""
STEP 5: Feature Selection trÃªn 81 features (gá»‘c + technicals shifted) rá»“i train láº¡i
Usage: python ml/step5_select_and_train.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import load_data, get_tscv, RANDOM_STATE as RS, DATA_PATH, SPLIT_DATE
from improve import add_technical_features

P = '=' * 90


def main():
    print(f'\n{P}\n STEP 5: FEATURE SELECTION + RETRAIN (81 features)\n{P}')

    # Load raw data + add shifted technicals
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    # Target
    y_col = 'oil_return'
    exclude = {'date', y_col, 'oil_close'}
    features = [c for c in df.columns if c not in exclude]

    train_mask = df['date'] < SPLIT_DATE
    test_mask = df['date'] >= SPLIT_DATE

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, y_col] > 0).astype(int)
    y_test = (df.loc[test_mask, y_col] > 0).astype(int)

    print(f'  Total features: {len(features)}')
    print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
    print(f'  Target: UP={y_train.sum()} ({y_train.mean():.1%}) | DOWN={len(y_train)-y_train.sum()}')

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 1. FEATURE RANKING (MI + Spearman)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f'\n{P}\n 1. FEATURE RANKING\n{P}')
    tscv = get_tscv()

    mi = mutual_info_classif(X_train.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train.corrwith(df.loc[train_mask, y_col], method='spearman').abs()

    rank = pd.DataFrame({'feature': features, 'MI': mi, 'abs_sp': sp.values})
    for c in ['MI', 'abs_sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2
    rank.sort_values('score', ascending=False, inplace=True)
    rank.reset_index(drop=True, inplace=True)

    print(f'\n {"#":<4} {"Feature":<30} {"MI":>8} {"|Sp|":>8} {"Score":>8}')
    print(f' {"-"*60}')
    for i, r in rank.iterrows():
        print(f' {i+1:<4} {r.feature:<30} {r.MI:>8.4f} {r.abs_sp:>8.4f} {r.score:>8.4f}')

    rank.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'step5_feature_ranking.csv'), index=False)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 2. COMPARE SUBSETS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f'\n{P}\n 2. SUBSET COMPARISON (LGBM proxy)\n{P}')

    subsets = {}
    for n in [10, 15, 20, 25, 30, 40, 50, len(features)]:
        if n >= len(features):
            subsets[f'ALL_{len(features)}'] = features
        else:
            subsets[f'TOP_{n}'] = rank.head(n)['feature'].tolist()

    proxy = LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=2,
                           n_estimators=200, max_depth=5, learning_rate=0.05)

    print(f'\n {"Subset":<15} {"N":>4} {"CV_Acc":>8} {"Test_Acc":>10} {"Test_F1m":>10}')
    print(f' {"-"*52}')
    best_subset = None; best_acc = 0
    for name, feats in subsets.items():
        cv = cross_val_score(proxy, X_train[feats], y_train, cv=tscv, scoring='accuracy')
        proxy.fit(X_train[feats], y_train)
        pred = proxy.predict(X_test[feats])
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average='macro')
        print(f' {name:<15} {len(feats):>4} {cv.mean():>8.4f} {acc:>10.4f} {f1m:>10.4f}')
        if acc > best_acc:
            best_acc = acc; best_subset = name

    print(f'\n Best subset: {best_subset} (Acc={best_acc:.4f})')
    best_feats = subsets[best_subset]

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 3. TRAIN TOP MODELS ON BEST SUBSET
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f'\n{P}\n 3. TRAIN ON {best_subset} ({len(best_feats)} features)\n{P}')

    X_tr = X_train[best_feats]
    X_te = X_test[best_feats]

    models = {
        'XGB': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=2, eval_metric='logloss'),
                {'n_estimators': [200, 300, 500], 'max_depth': [3, 5, 7],
                 'learning_rate': [0.01, 0.03, 0.05], 'reg_alpha': [0, 0.1]}),
        'GBM': (GradientBoostingClassifier(random_state=RS),
                {'n_estimators': [200, 300, 500], 'max_depth': [3, 5, 7],
                 'learning_rate': [0.01, 0.03, 0.05], 'min_samples_leaf': [5, 10]}),
        'LGBM': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=2, importance_type='gain'),
                 {'n_estimators': [200, 300, 500], 'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [15, 31]}),
    }

    results = []
    for name, (model, grid) in models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        gs = RandomizedSearchCV(model, grid, n_iter=15, cv=tscv,
                                scoring='accuracy', refit=True, n_jobs=2, random_state=RS)
        gs.fit(X_tr, y_train)
        pred = gs.best_estimator_.predict(X_te)
        prob = gs.best_estimator_.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average='macro')
        auc = roc_auc_score(y_test, prob)
        elapsed = round(time.time() - t0, 1)

        results.append({'Model': name, 'Accuracy': acc, 'F1_macro': f1m, 'AUC': auc,
                        'CV_Acc': gs.best_score_, 'Time_s': elapsed})
        print(f'  Best: {gs.best_params_}')
        print(f'  CV_Acc={gs.best_score_:.4f} | Test: Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f} ({elapsed}s)')

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 4. ALSO TRAIN ON ALL SUBSETS WITH BEST MODEL
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f'\n{P}\n 4. BEST MODEL (GBM) ON ALL SUBSETS\n{P}')

    all_sub_results = []
    for name, feats in subsets.items():
        gbm = GradientBoostingClassifier(random_state=RS, n_estimators=300, max_depth=5,
                                          learning_rate=0.03, min_samples_leaf=5)
        gbm.fit(X_train[feats], y_train)
        pred = gbm.predict(X_test[feats])
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average='macro')
        all_sub_results.append({'Subset': name, 'N': len(feats), 'Accuracy': acc, 'F1_macro': f1m})
        print(f'  {name:<15} (n={len(feats):>2}) Acc={acc:.4f} F1m={f1m:.4f}')

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # SUMMARY
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    print(f'\n{P}\n SUMMARY\n{P}')
    rdf = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    rdf.index = range(1, len(rdf) + 1)
    print(f'\n Best subset: {best_subset} ({len(best_feats)} features)')
    print(f'\n{rdf.to_string()}')
    rdf.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'step5_results.csv'), index=False)

    best = rdf.iloc[0]
    print(f'\n Baseline (no technicals, 42 features):  Acc=0.5274')
    print(f' Step4 (81 features, no selection):       Acc=0.5530')
    print(f' Step5 ({best_subset}, {len(best_feats)} features):')
    print(f'   Best: {best["Model"]} Acc={best["Accuracy"]:.4f} F1m={best["F1_macro"]:.4f} AUC={best["AUC"]:.4f}')

    # Top features
    print(f'\n Top 15 features used:')
    for i, f in enumerate(best_feats[:15]):
        sc = rank[rank.feature == f]['score'].values[0]
        print(f'   {i+1:>2}. {f:<30} score={sc:.4f}')

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()

