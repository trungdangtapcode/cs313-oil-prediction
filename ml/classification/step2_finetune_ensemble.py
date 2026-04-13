"""
STEP 2: Fine-tune and Ensemble

This step improves on the baseline by tuning stronger classifiers and
combining them with ensemble methods.

Goal of this step:
  - Fine-tune top classification models from the baseline stage
  - Test whether soft voting or stacking improves generalization
  - Compare tuned single models against ensemble models on validation

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Features from dataset_step4_transformed.csv
  - A TOP_25 subset is built inside this script using mutual information
    and Spearman ranking on the training split only

Models trained in this file:
  - Fine-tuned XGBClassifier
  - Fine-tuned GradientBoostingClassifier
  - Fine-tuned LGBMClassifier
  - Fine-tuned SVC
  - VotingClassifier
  - StackingClassifier

Model selection:
  - RandomizedSearchCV is used for tuned single-model variants
  - TimeSeriesSplit is used for time-aware cross-validation
  - Model selection happens on validation only
  - The chosen candidate is refit on train+validation and evaluated once on holdout test

Outputs:
  - Console validation comparison
  - Console holdout report for the selected candidate
  - results/step2_validation_results.csv
  - results/step2_holdout_result.csv

Usage:
  python ml/classification/step2_finetune_ensemble.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    DATA_PATH,
    DROP_COLS,
    TARGET,
    RANDOM_STATE as RS,
    get_tscv,
    get_train_val_test_masks,
    set_global_seed,
)

OUT = os.path.join(os.path.dirname(__file__), 'results')
P = '=' * 90


def _iloc_frame(X, idx):
    if hasattr(X, 'iloc'):
        return X.iloc[idx]
    return X[idx]


def _iloc_series(y, idx):
    if hasattr(y, 'iloc'):
        return y.iloc[idx]
    return y[idx]


def get_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    return None


class TimeSeriesStackingClassifier(BaseEstimator, ClassifierMixin):
    """Manual stacking compatible with TimeSeriesSplit."""

    def __init__(self, estimators, final_estimator=None, cv=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        if self.cv is None:
            raise ValueError('cv must be provided for TimeSeriesStackingClassifier')

        final_estimator = self.final_estimator
        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000)

        n_samples = len(X)
        n_estimators = len(self.estimators)
        oof_features = np.full((n_samples, n_estimators), np.nan, dtype=float)
        covered = np.zeros(n_samples, dtype=bool)

        for train_idx, val_idx in self.cv.split(X, y):
            X_tr = _iloc_frame(X, train_idx)
            y_tr = _iloc_series(y, train_idx)
            X_va = _iloc_frame(X, val_idx)
            for col_idx, (_, estimator) in enumerate(self.estimators):
                fold_model = clone(estimator)
                fold_model.fit(X_tr, y_tr)
                fold_scores = get_scores(fold_model, X_va)
                if fold_scores is None:
                    raise ValueError('Base estimator must support predict_proba or decision_function')
                oof_features[val_idx, col_idx] = fold_scores
            covered[val_idx] = True

        valid_rows = covered & np.isfinite(oof_features).all(axis=1)
        if valid_rows.sum() == 0:
            raise ValueError('No valid OOF rows available for stacking meta-model')

        self.meta_estimator_ = clone(final_estimator)
        self.meta_estimator_.fit(oof_features[valid_rows], _iloc_series(y, np.flatnonzero(valid_rows)))

        self.base_estimators_ = []
        for name, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.base_estimators_.append((name, fitted))

        self.classes_ = np.unique(y)
        return self

    def _stack_features(self, X):
        cols = []
        for _, estimator in self.base_estimators_:
            scores = get_scores(estimator, X)
            if scores is None:
                raise ValueError('Base estimator must support predict_proba or decision_function')
            cols.append(scores)
        return np.column_stack(cols)

    def predict_proba(self, X):
        stack_X = self._stack_features(X)
        if hasattr(self.meta_estimator_, 'predict_proba'):
            return self.meta_estimator_.predict_proba(stack_X)
        scores = self.meta_estimator_.decision_function(stack_X)
        probs_pos = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs_pos, probs_pos])

    def predict(self, X):
        stack_X = self._stack_features(X)
        return self.meta_estimator_.predict(stack_X)


def score_predictions(y_true, pred, score):
    return {
        'Accuracy': accuracy_score(y_true, pred),
        'F1m': f1_score(y_true, pred, average='macro'),
        'AUC': roc_auc_score(y_true, score) if score is not None else np.nan,
    }


def main():
    seed = set_global_seed()
    print(f'\n{P}\n CLASSIFICATION FINE-TUNE + ENSEMBLE\n{P}')
    print(f'  Seed: {seed}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    train_mask, val_mask, test_mask, split_col = get_train_val_test_masks(df)
    features = [c for c in df.columns if c not in DROP_COLS and c != TARGET]

    X_train_full = df.loc[train_mask, features]
    X_val_full = df.loc[val_mask, features]
    X_test_full = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_val = (df.loc[val_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    tscv = get_tscv()
    print(f'  Train: {len(X_train_full)} | Val: {len(X_val_full)} | Test: {len(X_test_full)}')
    if split_col != 'date':
        target_dates = pd.to_datetime(df[split_col])
        print(f'  Train targets: {target_dates[train_mask].iloc[0].date()} -> {target_dates[train_mask].iloc[-1].date()}')
        print(f'  Val targets:   {target_dates[val_mask].iloc[0].date()} -> {target_dates[val_mask].iloc[-1].date()}')
        print(f'  Test targets:  {target_dates[test_mask].iloc[0].date()} -> {target_dates[test_mask].iloc[-1].date()}')

    mi = mutual_info_classif(X_train_full.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train_full.corrwith(df.loc[train_mask, TARGET], method='spearman').abs()
    rank = pd.DataFrame({'f': features, 'MI': mi, 'sp': sp.values})
    for c in ['MI', 'sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['sp_n']) / 2
    top25 = rank.sort_values('score', ascending=False).head(25)['f'].tolist()

    X_train = X_train_full[top25]
    X_val = X_val_full[top25]
    X_test = X_test_full[top25]
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=top25, index=X_train.index)
    X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=top25, index=X_val.index)
    print(f'  Features: {len(top25)} (TOP_25)')

    models = {
        'GBM_v2': (GradientBoostingClassifier(random_state=RS), {
            'n_estimators': [200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.05],
            'min_samples_leaf': [5, 10],
            'subsample': [0.8, 1.0],
        }, False),
        'XGB_v2': (XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss'), {
            'n_estimators': [200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.05],
            'reg_alpha': [0, 0.1],
            'subsample': [0.8, 1.0],
        }, False),
        'LGBM_v2': (LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, importance_type='gain'), {
            'n_estimators': [200, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.03, 0.05],
            'num_leaves': [15, 31],
            'min_child_samples': [5, 20],
        }, False),
        'SVM_RBF_v2': (SVC(random_state=RS, probability=False, kernel='rbf'), {
            'C': [0.1, 1.0, 5.0, 10.0],
            'gamma': ['scale', 'auto', 0.01],
        }, True),
    }

    val_results = []
    for name, (model, grid, use_sc) in models.items():
        print(f'\n--- {name} ---')
        t0 = time.time()
        Xtr = X_train_sc if use_sc else X_train
        Xva = X_val_sc if use_sc else X_val
        gs = RandomizedSearchCV(
            model,
            grid,
            n_iter=30,
            cv=tscv,
            scoring='accuracy',
            refit=True,
            n_jobs=1,
            random_state=RS,
        )
        gs.fit(Xtr, y_train)
        pred = gs.best_estimator_.predict(Xva)
        score = get_scores(gs.best_estimator_, Xva)
        metrics = score_predictions(y_val, pred, score)
        elapsed = round(time.time() - t0, 1)
        val_results.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'F1m': metrics['F1m'],
            'AUC': metrics['AUC'],
            'CV_Acc': gs.best_score_,
            'Time_s': elapsed,
            'Estimator': clone(gs.best_estimator_),
            'Use_Scaled': use_sc,
            'Params': str(gs.best_params_),
        })
        print(f'  Best: {gs.best_params_}')
        print(f'  CV={gs.best_score_:.4f} | Val Acc={metrics["Accuracy"]:.4f} F1m={metrics["F1m"]:.4f} AUC={metrics["AUC"]:.4f}')

    base = [
        ('lgbm', LGBMClassifier(random_state=RS, verbosity=-1, n_jobs=1, n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('xgb', XGBClassifier(random_state=RS, verbosity=0, n_jobs=1, eval_metric='logloss', n_estimators=100, max_depth=3, learning_rate=0.05)),
        ('gbm', GradientBoostingClassifier(random_state=RS, n_estimators=100, max_depth=3, learning_rate=0.05)),
    ]
    ensembles = [
        ('Voting', VotingClassifier(estimators=base, voting='soft', n_jobs=1), False),
        ('Stacking', TimeSeriesStackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(random_state=RS, max_iter=1000),
            cv=tscv,
        ), False),
    ]
    for name, estimator, use_sc in ensembles:
        print(f'\n--- {name} ---')
        t0 = time.time()
        Xtr = X_train_sc if use_sc else X_train
        Xva = X_val_sc if use_sc else X_val
        estimator.fit(Xtr, y_train)
        pred = estimator.predict(Xva)
        score = get_scores(estimator, Xva)
        metrics = score_predictions(y_val, pred, score)
        elapsed = round(time.time() - t0, 1)
        val_results.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'F1m': metrics['F1m'],
            'AUC': metrics['AUC'],
            'CV_Acc': np.nan,
            'Time_s': elapsed,
            'Estimator': clone(estimator),
            'Use_Scaled': use_sc,
            'Params': 'predefined',
        })
        print(f'  Val Acc={metrics["Accuracy"]:.4f} F1m={metrics["F1m"]:.4f} AUC={metrics["AUC"]:.4f}')

    rdf = pd.DataFrame(val_results).sort_values(['F1m', 'Accuracy', 'CV_Acc'], ascending=False)
    print(f'\n{P}\n VALIDATION RESULTS\n{P}')
    print(rdf[['Model', 'Accuracy', 'F1m', 'AUC', 'CV_Acc', 'Time_s']].to_string(index=False))
    rdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT, 'step2_validation_results.csv'), index=False)

    best = rdf.iloc[0]
    final_model = clone(best['Estimator'])
    X_train_val_raw = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    if best['Use_Scaled']:
        scaler_final = StandardScaler()
        X_train_val = pd.DataFrame(
            scaler_final.fit_transform(X_train_val_raw),
            columns=top25,
            index=X_train_val_raw.index,
        )
        X_holdout = pd.DataFrame(
            scaler_final.transform(X_test),
            columns=top25,
            index=X_test.index,
        )
    else:
        X_train_val = X_train_val_raw
        X_holdout = X_test

    final_model.fit(X_train_val, y_train_val)
    holdout_pred = final_model.predict(X_holdout)
    holdout_score = get_scores(final_model, X_holdout)
    holdout = score_predictions(y_test, holdout_pred, holdout_score)
    holdout_row = {
        'Model': best['Model'],
        'Selected_On': 'validation',
        'Validation_Accuracy': best['Accuracy'],
        'Validation_F1m': best['F1m'],
        'Validation_AUC': best['AUC'],
        'Holdout_Accuracy': holdout['Accuracy'],
        'Holdout_F1m': holdout['F1m'],
        'Holdout_AUC': holdout['AUC'],
        'Params': best['Params'],
    }
    pd.DataFrame([holdout_row]).to_csv(os.path.join(OUT, 'step2_holdout_result.csv'), index=False)

    print(f'\n{P}\n SELECTED CANDIDATE\n{P}')
    print(f'  {best["Model"]}: Val Acc={best["Accuracy"]:.4f} F1m={best["F1m"]:.4f} AUC={best["AUC"]:.4f}')
    print(f'  Holdout: Acc={holdout["Accuracy"]:.4f} F1m={holdout["F1m"]:.4f} AUC={holdout["AUC"]:.4f}')
    print(f'\n{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
