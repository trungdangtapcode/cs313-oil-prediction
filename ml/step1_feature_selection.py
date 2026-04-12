"""
STEP 1: Feature Selection & Subset Comparison
Usage: python ml/step1_feature_selection.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

from config import load_data, get_tscv, OUT_DIR, RANDOM_STATE as RS

P = '=' * 90

def main():
    print(f'\n{P}\n STEP 1: FEATURE SELECTION\n{P}')
    data = load_data()
    X_train, y_train = data['X_train'], data['y_train']
    features = data['features']
    tscv = get_tscv()

    # MI regression + classification + Spearman
    mi_reg = mutual_info_regression(X_train.fillna(0), y_train, random_state=RS, n_neighbors=5)
    y_cls = (y_train > 0).astype(int)
    mi_cls = mutual_info_classif(X_train.fillna(0), y_cls, random_state=RS, n_neighbors=5)
    sp = X_train.corrwith(y_train, method='spearman')

    mi_df = pd.DataFrame({'feature': features, 'MI_reg': mi_reg, 'MI_cls': mi_cls, 'abs_sp': sp.abs().values})
    for c in ['MI_reg', 'MI_cls', 'abs_sp']:
        mx = mi_df[c].max()
        mi_df[f'{c}_n'] = mi_df[c] / mx if mx > 0 else 0
    mi_df['score'] = (mi_df['MI_reg_n'] + mi_df['MI_cls_n'] + mi_df['abs_sp_n']) / 3
    mi_df.sort_values('score', ascending=False, inplace=True)
    mi_df.reset_index(drop=True, inplace=True)

    print(f'\n {"#":<4} {"Feature":<30} {"MI_reg":>8} {"MI_cls":>8} {"|Sp|":>8} {"Score":>8}')
    print(f' {"-"*70}')
    for i, r in mi_df.iterrows():
        print(f' {i+1:<4} {r.feature:<30} {r.MI_reg:>8.4f} {r.MI_cls:>8.4f} {r.abs_sp:>8.4f} {r.score:>8.4f}')

    # Subsets
    subsets = {
        'ALL_42': features,
        'TOP_10': mi_df.head(10)['feature'].tolist(),
        'TOP_15': mi_df.head(15)['feature'].tolist(),
        'TOP_20': mi_df.head(20)['feature'].tolist(),
        'TOP_25': mi_df.head(25)['feature'].tolist(),
    }

    # Compare
    print(f'\n{P}\n SUBSET COMPARISON\n{P}')
    reg_m = LGBMRegressor(random_state=RS, verbosity=-1, n_jobs=-1, n_estimators=200, max_depth=5, learning_rate=0.05)
    cls_m = GradientBoostingClassifier(random_state=RS, n_estimators=200, max_depth=5, learning_rate=0.05)

    rows = []
    for name, feats in subsets.items():
        X_tr, X_te = data['X_train'][feats], data['X_test'][feats]
        y_te_reg, y_te_cls = data['y_test'], (data['y_test'] > 0).astype(int)

        reg_m.fit(X_tr, y_train)
        p_reg = reg_m.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te_reg, p_reg))
        r2 = r2_score(y_te_reg, p_reg)

        cls_m.fit(X_tr, y_cls)
        p_cls = cls_m.predict(X_te)
        acc = accuracy_score(y_te_cls, p_cls)
        f1m = f1_score(y_te_cls, p_cls, average='macro')

        rows.append({'Subset': name, 'N': len(feats), 'RMSE': rmse, 'R2': r2, 'Acc': acc, 'F1m': f1m})
        print(f'  {name:<15} (n={len(feats):>2}) | RMSE={rmse:.5f} R2={r2:>7.4f} | Acc={acc:.4f} F1m={f1m:.4f}')

    rdf = pd.DataFrame(rows)
    print(f'\n Best regression:      {rdf.loc[rdf.RMSE.idxmin(), "Subset"]}')
    print(f' Best classification:  {rdf.loc[rdf.F1m.idxmax(), "Subset"]}')

    mi_df.to_csv(os.path.join(OUT_DIR, 'feature_ranking.csv'), index=False)
    rdf.to_csv(os.path.join(OUT_DIR, 'subset_comparison.csv'), index=False)

    # Save best subsets for next steps
    for name, feats in subsets.items():
        pd.Series(feats).to_csv(os.path.join(OUT_DIR, f'subset_{name}.csv'), index=False, header=False)

    print(f'\n Saved to ml/results/')
    print(f'{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
