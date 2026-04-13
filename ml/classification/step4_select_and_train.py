"""
STEP 4: Feature Selection and Retraining

This step starts from the technical-feature dataset created in step 3
and asks a practical question: if we rank the 81 features and keep only
the strongest ones, does validation accuracy improve?

Goal of this step:
  - Rank the 81-feature set using three signals: MI, Spearman, and MI+Spearman
  - Compare 28 subset cases with a validation-safe proxy model
  - Retrain several models on the best-performing subset case

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base features from dataset_step4_transformed.csv
  - Technical and lag features added by step3_technical_improve.add_technical_features()
  - Final candidate set is the no-leakage 81-feature dataset

Models trained in this file:
  - XGBClassifier
  - GradientBoostingClassifier
  - LGBMClassifier
  - LGBMClassifier is also used as a proxy model for subset comparison

Model selection:
  - Feature ranking with MI-only, Spearman-only, and MI+Spearman
  - Validation subset comparison with TimeSeriesSplit CV on train
  - RandomizedSearchCV for the final retraining stage

Outputs:
  - Console ranking and subset comparison
  - results/step4_feature_ranking.csv
  - results/step4_subset_comparison.csv
  - results/step4_results.csv

Usage:
  python ml/classification/step4_select_and_train.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import get_tscv, get_train_val_test_masks, RANDOM_STATE as RS, DATA_PATH, TARGET, TARGET_DATE_COL, set_global_seed
from step3_technical_improve import add_technical_features

P = '=' * 90
OUT = os.path.join(os.path.dirname(__file__), 'results')
SUBSET_SIZES = [10, 15, 20, 25, 30, 40, 50, 60, 70]


def build_rankings(X_train, y_train, features):
    mi = mutual_info_classif(X_train.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train.corrwith(y_train, method='spearman').abs()

    rank = pd.DataFrame({'feature': features, 'MI': mi, 'abs_sp': sp.values})
    for col in ['MI', 'abs_sp']:
        mx = rank[col].max()
        rank[f'{col}_n'] = rank[col] / mx if mx > 0 else 0
    rank['mix_score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2

    scheme_orders = {
        'spearman': rank.sort_values(['abs_sp', 'MI'], ascending=False).reset_index(drop=True),
        'mi': rank.sort_values(['MI', 'abs_sp'], ascending=False).reset_index(drop=True),
        'mi_spearman': rank.sort_values(['mix_score', 'MI', 'abs_sp'], ascending=False).reset_index(drop=True),
    }

    rank['rank_spearman'] = rank['feature'].map(
        {f: i + 1 for i, f in enumerate(scheme_orders['spearman']['feature'])}
    )
    rank['rank_mi'] = rank['feature'].map(
        {f: i + 1 for i, f in enumerate(scheme_orders['mi']['feature'])}
    )
    rank['rank_mi_spearman'] = rank['feature'].map(
        {f: i + 1 for i, f in enumerate(scheme_orders['mi_spearman']['feature'])}
    )
    rank.sort_values('rank_mi_spearman', inplace=True)
    rank.reset_index(drop=True, inplace=True)
    return rank, scheme_orders


def build_subset_cases(features, scheme_orders):
    subset_sizes = [n for n in SUBSET_SIZES if n < len(features)]
    cases = []
    for scheme_name, ordered in scheme_orders.items():
        for n in subset_sizes:
            cases.append({
                'Case': f'{scheme_name.upper()}_TOP_{n}',
                'Ranking': scheme_name,
                'Subset': f'TOP_{n}',
                'N': n,
                'Features': ordered.head(n)['feature'].tolist(),
            })
    cases.append({
        'Case': f'ALL_{len(features)}',
        'Ranking': 'all',
        'Subset': f'ALL_{len(features)}',
        'N': len(features),
        'Features': list(features),
    })
    return cases


def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 4: FEATURE SELECTION + RETRAIN (81 features)\n{P}')
    print(f'  Seed: {seed}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    y_col = TARGET
    exclude = {'date', y_col, TARGET_DATE_COL, 'oil_close'}
    features = [c for c in df.columns if c not in exclude]

    train_mask, val_mask, test_mask, _ = get_train_val_test_masks(df)

    X_train = df.loc[train_mask, features]
    X_val = df.loc[val_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, y_col] > 0).astype(int)
    y_val = (df.loc[val_mask, y_col] > 0).astype(int)
    y_test = (df.loc[test_mask, y_col] > 0).astype(int)

    print(f'  Total features: {len(features)}')
    print(f'  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')
    print(f'  Target: UP={y_train.sum()} ({y_train.mean():.1%}) | DOWN={len(y_train)-y_train.sum()}')

    print(f'\n{P}\n 1. FEATURE RANKING\n{P}')
    tscv = get_tscv()
    rank, scheme_orders = build_rankings(X_train, y_train, features)

    print(f'\n {"#":<4} {"Feature":<30} {"MI":>8} {"|Sp|":>8} {"Mix":>8} {"MI_rk":>7} {"Sp_rk":>7} {"Mix_rk":>8}')
    print(f' {"-"*92}')
    for i, row in rank.iterrows():
        print(
            f' {i+1:<4} {row.feature:<30} {row.MI:>8.4f} {row.abs_sp:>8.4f} {row.mix_score:>8.4f} '
            f'{int(row.rank_mi):>7} {int(row.rank_spearman):>7} {int(row.rank_mi_spearman):>8}'
        )

    rank.to_csv(os.path.join(OUT, 'step4_feature_ranking.csv'), index=False)

    print(f'\n Top 10 by ranking scheme:')
    for scheme_name, ordered in scheme_orders.items():
        print(f'  {scheme_name}: {", ".join(ordered.head(10)["feature"].tolist())}')

    print(f'\n{P}\n 2. SUBSET COMPARISON (28 cases, LGBM proxy)\n{P}')
    subset_cases = build_subset_cases(features, scheme_orders)
    print(f'  Cases: {len(subset_cases)}')
    if len(subset_cases) != 28:
        print(f'  Warning: expected 28 cases, got {len(subset_cases)} because total features={len(features)}')

    proxy = LGBMClassifier(
        random_state=RS,
        verbosity=-1,
        n_jobs=1,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
    )

    print(f'\n {"Case":<24} {"Rank":<13} {"N":>4} {"CV_Acc":>8} {"Val_Acc":>10} {"Val_F1m":>10}')
    print(f' {"-"*76}')
    subset_rows = []
    best_case = None
    best_acc = -np.inf
    best_f1 = -np.inf
    for case in subset_cases:
        feats = case['Features']
        cv = cross_val_score(proxy, X_train[feats], y_train, cv=tscv, scoring='accuracy')
        proxy.fit(X_train[feats], y_train)
        pred = proxy.predict(X_val[feats])
        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average='macro')
        row = {
            'Case': case['Case'],
            'Ranking': case['Ranking'],
            'Subset': case['Subset'],
            'N': len(feats),
            'CV_Acc': cv.mean(),
            'Val_Acc': acc,
            'Val_F1m': f1m,
        }
        subset_rows.append(row)
        print(f' {case["Case"]:<24} {case["Ranking"]:<13} {len(feats):>4} {cv.mean():>8.4f} {acc:>10.4f} {f1m:>10.4f}')
        if acc > best_acc or (acc == best_acc and f1m > best_f1):
            best_acc = acc
            best_f1 = f1m
            best_case = case

    best_feats = best_case['Features']
    print(f'\n Best case on validation: {best_case["Case"]} ({len(best_feats)} features, Acc={best_acc:.4f})')

    print(f'\n Best case by ranking family:')
    best_family_rows = []
    subset_df = pd.DataFrame(subset_rows)
    for ranking_name in ['spearman', 'mi', 'mi_spearman', 'all']:
        fam = subset_df[subset_df['Ranking'] == ranking_name]
        if fam.empty:
            continue
        best_fam = fam.sort_values(['Val_Acc', 'Val_F1m', 'CV_Acc'], ascending=False).iloc[0]
        best_family_rows.append(best_fam.to_dict())
        print(
            f'  {ranking_name:<13} {best_fam["Case"]:<24} '
            f'ValAcc={best_fam["Val_Acc"]:.4f} ValF1m={best_fam["Val_F1m"]:.4f}'
        )

    print(f'\n{P}\n 3. TRAIN ON {best_case["Case"]} ({len(best_feats)} features)\n{P}')
    X_tr = X_train[best_feats]
    X_va = X_val[best_feats]
    X_te = X_test[best_feats]

    models = {
        'XGB': (
            XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'reg_alpha': [0, 0.1],
            },
        ),
        'GBM': (
            GradientBoostingClassifier(random_state=RS),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'min_samples_leaf': [5, 10],
            },
        ),
        'LGBM': (
            LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.03, 0.05],
                'num_leaves': [15, 31],
            },
        ),
    }

    results = []
    for name, (model, grid) in models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        gs = RandomizedSearchCV(
            model,
            grid,
            n_iter=15,
            cv=tscv,
            scoring='accuracy',
            refit=True,
            n_jobs=1,
            random_state=RS,
        )
        gs.fit(X_tr, y_train)
        pred = gs.best_estimator_.predict(X_va)
        prob = gs.best_estimator_.predict_proba(X_va)[:, 1]

        acc = accuracy_score(y_val, pred)
        f1m = f1_score(y_val, pred, average='macro')
        auc = roc_auc_score(y_val, prob)
        elapsed = round(time.time() - t0, 1)

        results.append({
            'Model': name,
            'Val_Accuracy': acc,
            'Val_F1_macro': f1m,
            'Val_AUC': auc,
            'CV_Acc': gs.best_score_,
            'Time_s': elapsed,
            'Estimator': gs.best_estimator_,
        })
        print(f'  Best: {gs.best_params_}')
        print(f'  CV_Acc={gs.best_score_:.4f} | Val: Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f} ({elapsed}s)')

    print(f'\n{P}\n SUMMARY\n{P}')
    rdf = pd.DataFrame(results).sort_values(['Val_Accuracy', 'Val_F1_macro'], ascending=False)
    rdf.index = range(1, len(rdf) + 1)
    best_row = rdf.iloc[0]
    final_model = clone(best_row['Estimator'])
    X_train_val = pd.concat([X_train[best_feats], X_val[best_feats]])
    y_train_val = pd.concat([y_train, y_val])
    final_model.fit(X_train_val, y_train_val)
    test_pred = final_model.predict(X_te)
    test_prob = final_model.predict_proba(X_te)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_f1m = f1_score(y_test, test_pred, average='macro')
    test_auc = roc_auc_score(y_test, test_prob)

    rdf['Selected_Case'] = best_case['Case']
    rdf['Ranking'] = best_case['Ranking']
    rdf['N_Features'] = len(best_feats)
    rdf['Final_Test_Accuracy'] = np.nan
    rdf['Final_Test_F1_macro'] = np.nan
    rdf['Final_Test_AUC'] = np.nan
    rdf.loc[rdf.index[0], 'Final_Test_Accuracy'] = test_acc
    rdf.loc[rdf.index[0], 'Final_Test_F1_macro'] = test_f1m
    rdf.loc[rdf.index[0], 'Final_Test_AUC'] = test_auc

    print(f'\n Best case on validation: {best_case["Case"]} ({len(best_feats)} features)')
    print(f' Validation model comparison:')
    print(rdf.drop(columns=['Estimator']).to_string())
    print(f'\n Final holdout test ({best_row["Model"]} selected on validation): Acc={test_acc:.4f} F1m={test_f1m:.4f} AUC={test_auc:.4f}')

    pd.DataFrame(subset_rows).sort_values(['Val_Acc', 'Val_F1m', 'CV_Acc'], ascending=False).to_csv(
        os.path.join(OUT, 'step4_subset_comparison.csv'),
        index=False,
    )
    pd.DataFrame(best_family_rows).to_csv(os.path.join(OUT, 'step4_best_by_ranking.csv'), index=False)
    rdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT, 'step4_results.csv'), index=False)

    print(f'\n Top 15 features used in {best_case["Case"]}:')
    ordered_lookup = scheme_orders.get(best_case['Ranking'])
    if ordered_lookup is not None:
        best_rank_df = ordered_lookup.set_index('feature')
    else:
        best_rank_df = rank.set_index('feature')
    for i, feat in enumerate(best_feats[:15], start=1):
        mi_val = best_rank_df.loc[feat, 'MI']
        sp_val = best_rank_df.loc[feat, 'abs_sp']
        mix_val = best_rank_df.loc[feat, 'mix_score']
        print(f'   {i:>2}. {feat:<30} MI={mi_val:.4f} |Sp|={sp_val:.4f} Mix={mix_val:.4f}')

    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
