"""
STEP 2: Fine-tune and Ensemble

This step improves on the baseline by tuning stronger classifiers and
combining them with ensemble methods.

Goal of this step:
  - Fine-tune top classification models from the baseline stage
  - Test whether soft voting or stacking improves generalization
  - Compare tuned single models against ensemble models

Target used in this file:
  - Binary classification: oil_return > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Features loaded from config.load_data()
  - A TOP_25 subset is built inside this script using mutual information
    and Spearman ranking on the training split

Models trained in this file:
  - Fine-tuned XGBClassifier
  - Fine-tuned GradientBoostingClassifier
  - Fine-tuned LGBMClassifier
  - VotingClassifier
  - StackingClassifier

Model selection:
  - RandomizedSearchCV is used for tuned single-model variants
  - TimeSeriesSplit is used for time-aware cross-validation
  - Results are evaluated with Accuracy, F1_macro, and AUC

Outputs:
  - Console comparison of tuned models and ensembles
  - results/finetune_classification.csv
  - results/ensemble_classification.csv

Usage:
  python ml/classification/step2_finetune_ensemble.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import load_data, get_tscv, RANDOM_STATE as RS

OUT = os.path.join(os.path.dirname(__file__), 'results')
P = '=' * 90

def main():
    print(f'\n{P}\n CLASSIFICATION FINE-TUNE + ENSEMBLE\n{P}')
    data = load_data(); tscv = get_tscv()
    y_tr = (data['y_train'] > 0).astype(int)
    y_te = (data['y_test'] > 0).astype(int)

    # Use TOP_25 features (best from feature selection)
    from sklearn.feature_selection import mutual_info_classif
    feats = data['features']
    mi = mutual_info_classif(data['X_train'].fillna(0), y_tr, random_state=RS, n_neighbors=5)
    sp = data['X_train'].corrwith(data['y_train'], method='spearman').abs()
    rank = pd.DataFrame({'f': feats, 'MI': mi, 'sp': sp.values})
    for c in ['MI', 'sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['sp_n']) / 2
    top25 = rank.sort_values('score', ascending=False).head(25)['f'].tolist()

    X_tr = data['X_train'][top25]; X_te = data['X_test'][top25]
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=top25, index=X_tr.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=top25, index=X_te.index)
    print(f'  Features: {len(top25)} (TOP_25)')

    # Fine-tune
    models = {
        'GBM_v2': (GradientBoostingClassifier(random_state=RS),
                   {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                    'min_samples_leaf': [5, 10], 'subsample': [0.8, 1.0]}, False),
        'XGB_v2': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'),
                   {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                    'reg_alpha': [0, 0.1], 'subsample': [0.8, 1.0]}, False),
        'LGBM_v2': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'),
                    {'n_estimators': [200, 500], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.03, 0.05],
                     'num_leaves': [15, 31], 'min_child_samples': [5, 20]}, False),
        'SVM_RBF_v2': (SVC(random_state=RS, probability=True, kernel='rbf'),
                       {'C': [0.1, 1.0, 5.0, 10.0], 'gamma': ['scale', 'auto', 0.01]}, True),
    }
    results = []; preds = {}
    for name, (model, grid, use_sc) in models.items():
        print(f'\n--- {name} ---'); t0 = time.time()
        Xtr = X_tr_sc if use_sc else X_tr; Xte = X_te_sc if use_sc else X_te
        gs = RandomizedSearchCV(model, grid, n_iter=30, cv=tscv, scoring='accuracy', refit=True, n_jobs=1, random_state=RS)
        gs.fit(Xtr, y_tr); pred = gs.best_estimator_.predict(Xte); preds[name] = pred
        prob = gs.best_estimator_.predict_proba(Xte)[:, 1] if hasattr(gs.best_estimator_, 'predict_proba') else None
        acc = accuracy_score(y_te, pred); f1m = f1_score(y_te, pred, average='macro')
        auc = roc_auc_score(y_te, prob) if prob is not None else np.nan
        results.append({'Model': name, 'Acc': acc, 'F1m': f1m, 'AUC': auc, 'Time_s': round(time.time()-t0, 1)})
        print(f'  Best: {gs.best_params_}\n  Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f}')

    # Ensemble
    base = [
        ('lgbm', LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('xgb', XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss', n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('gbm', GradientBoostingClassifier(random_state=RS, n_estimators=100, max_depth=3, learning_rate=0.05)),
    ]
    for ename, ens in [('Voting', VotingClassifier(estimators=base, voting='soft', n_jobs=1)),
                        ('Stacking', StackingClassifier(estimators=base, final_estimator=LogisticRegression(random_state=RS, max_iter=1000), cv=3, n_jobs=1))]:
        print(f'\n--- {ename} ---'); t0 = time.time()
        ens.fit(X_tr, y_tr); pred = ens.predict(X_te); preds[ename] = pred
        prob = ens.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_te, pred); f1m = f1_score(y_te, pred, average='macro')
        auc = roc_auc_score(y_te, prob)
        results.append({'Model': ename, 'Acc': acc, 'F1m': f1m, 'AUC': auc, 'Time_s': round(time.time()-t0, 1)})
        print(f'  Acc={acc:.4f} F1m={f1m:.4f} AUC={auc:.4f}')

    rdf = pd.DataFrame(results).sort_values('F1m', ascending=False)
    print(f'\n{P}\n RESULTS\n{P}'); print(rdf.to_string(index=False))
    rdf.to_csv(os.path.join(OUT, 'finetune_results.csv'), index=False)
    print(f'\n{P}\n DONE\n{P}')

if __name__ == '__main__':
    main()
