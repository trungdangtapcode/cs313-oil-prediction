"""
STEP 1: Baseline Classification

This is the baseline classification benchmark for the project.
It trains a standard set of classifiers on the original feature set
without adding technical indicators or extra feature-selection logic.

Goal of this step:
  - Establish a clean baseline for predicting daily oil direction
  - Compare common model families with train-only CV tuning
  - Evaluate the tuned models directly on the final test window

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base feature set from dataset_step4_transformed.csv
  - Raw price columns and leakage-prone columns are already dropped in config
  - This step uses the baseline feature list from config, not the 81-feature technical set

Models trained in this file:
  - LogisticRegression
  - SVC (linear)
  - SVC (RBF)
  - RandomForestClassifier
  - GradientBoostingClassifier
  - XGBClassifier
  - LGBMClassifier
  - MLPClassifier

Model selection:
  - Each model is tuned with GridSearchCV on the training window
  - TimeSeriesSplit is used for time-aware cross-validation
  - Hyperparameter tuning happens only inside the training window
  - Tuned candidates are evaluated directly on the final test window

Outputs:
  - Console test leaderboard
  - Console test report for the selected baseline
  - results/step1_test_results.csv
  - results/step1_selected_result.csv
  - results/step1_feature_importance.csv
  - plots such as ROC, confusion matrix, and backtest for the selected baseline

Usage:
  python ml/classification/step1_train_baseline.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from model_preprocessing import (
    build_model_time_preprocessor,
    get_model_time_groups,
    get_preprocessor_feature_names,
    save_model_bundle,
)
from config import (
    DATA_PATH,
    DROP_COLS,
    OUT_DIR,
    TARGET,
    RANDOM_STATE,
    get_tscv,
    get_train_test_masks,
    set_global_seed,
)

from metrics import evaluate, get_scores, METRIC_COLS, SORT_COLS

P = '=' * 90
OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
SEARCH_N_JOBS = max(1, int(os.getenv('SEARCH_N_JOBS', str(min(8, max(1, CPU_COUNT // 6))))))





def get_models():
    return [
        ('LogisticRegression', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000), {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
        }, True),
        ('SVM_RBF', SVC(random_state=RANDOM_STATE, probability=False), {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf'],
        }, True),
        ('SVM_Linear', SVC(random_state=RANDOM_STATE, probability=False, kernel='linear'), {
            'C': [0.01, 0.1, 1.0, 10.0],
        }, True),
        ('RandomForest', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), {
            'n_estimators': [100, 300],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [3, 5, 10],
        }, False),
        ('GradientBoosting', GradientBoostingClassifier(random_state=RANDOM_STATE), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
        }, False),
        ('XGBoost', XGBClassifier(random_state=RANDOM_STATE, verbosity=0, n_jobs=1, eval_metric='logloss'), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1],
        }, False),
        ('LightGBM', LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1, importance_type='gain'), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31],
        }, False),
        ('MLP', MLPClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=False), {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001],
        }, True),
    ]


def main():
    seed = set_global_seed()
    print(f'\n{P}\n CLASSIFICATION - Oil Price Direction (UP/DOWN)\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: search_jobs={SEARCH_N_JOBS}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    train_mask, test_mask, split_col = get_train_test_masks(df)
    features = [c for c in df.columns if c not in DROP_COLS and c != TARGET]

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)
    y_test_ret = df.loc[test_mask, TARGET].values
    dates_test = df.loc[test_mask, 'date']

    scale_groups = get_model_time_groups(features, data_path=DATA_PATH)
    use_baked_step5c = scale_groups["schema"] == "step5c_baked_scaled_passthrough"
    scale_preprocessor = None if use_baked_step5c else build_model_time_preprocessor(features, data_path=DATA_PATH)
    scaled_feature_names = None
    X_train_sc = None
    X_test_sc = None
    if any(use_sc for _, _, _, use_sc in get_models()):
        if use_baked_step5c:
            X_train_sc = X_train
            X_test_sc = X_test
            scaled_feature_names = list(features)
        else:
            X_train_sc_arr = scale_preprocessor.fit_transform(X_train)
            scaled_feature_names = get_preprocessor_feature_names(scale_preprocessor)
            X_train_sc = pd.DataFrame(X_train_sc_arr, columns=scaled_feature_names, index=X_train.index)
            X_test_sc = pd.DataFrame(
                scale_preprocessor.transform(X_test),
                columns=scaled_feature_names,
                index=X_test.index,
            )

    tscv = get_tscv()
    models = get_models()

    print(f'  Features: {len(features)}')
    print(f'  Train: {len(X_train)} | Test: {len(X_test)}')
    print(f'  Model-time preprocessing schema: {scale_groups["schema"]}')
    if use_baked_step5c:
        print('  Input is baked-scaled step5c dataset -> skip model-time scaler')
    if scale_groups['schema'].endswith('_curated'):
        print(
            f'  Preprocessor groups: standard={len(scale_groups["standard"])} '
            f'robust={len(scale_groups["robust"])} power={len(scale_groups["power"])} '
            f'passthrough={len(scale_groups["passthrough"]) + len(scale_groups["other"])}'
        )
    if split_col != 'date':
        target_dates = pd.to_datetime(df[split_col])
        print(f'  Train targets: {target_dates[train_mask].iloc[0].date()} -> {target_dates[train_mask].iloc[-1].date()}')
        print(f'  Test targets:  {target_dates[test_mask].iloc[0].date()} -> {target_dates[test_mask].iloc[-1].date()}')
    print(f'\n  Target distribution (Train): UP={y_train.sum()} ({y_train.mean():.1%}) | DOWN={len(y_train)-y_train.sum()} ({1-y_train.mean():.1%})')
    print(f'  Target distribution (Test):  UP={y_test.sum()} ({y_test.mean():.1%}) | DOWN={len(y_test)-y_test.sum()} ({1-y_test.mean():.1%})')

    test_results = []

    for name, model, grid, use_sc in models:
        print(f'\n--- {name} ---')
        t0 = time.time()

        X_tr = X_train_sc if use_sc else X_train
        X_te = X_test_sc if use_sc else X_test

        gs = GridSearchCV(model, grid, cv=tscv, scoring='accuracy', refit=True, n_jobs=SEARCH_N_JOBS)
        gs.fit(X_tr, y_train)
        best = gs.best_estimator_

        y_pred = best.predict(X_te)
        y_score = get_scores(best, X_te)
        res = evaluate(name, y_test.values, y_pred, y_score)
        elapsed = round(time.time() - t0, 1)
        test_results.append({
            **res,
            'CV_Acc': gs.best_score_,
            'Time_s': elapsed,
            'Estimator': best,
            'Use_Scaled': use_sc,
            'Params': str(gs.best_params_),
        })

        print(f'  Best params: {gs.best_params_}')
        print(f'  CV Accuracy: {gs.best_score_:.4f}')
        print(f'  Test Accuracy:   {res["Accuracy"]:.4f}')
        print(f'  Test F1 (binary): {res["F1_binary"]:.4f}')
        print(f'  Test F1 (macro):  {res["F1_macro"]:.4f}')
        print(f'  Test AUC:         {res["AUC"]:.4f}')
        print(f'  Test Confusion:   TP={res["TP"]} FP={res["FP"]} TN={res["TN"]} FN={res["FN"]}')
        print(f'  Time:            {elapsed}s')

    print(f'\n{P}\n TEST LEADERBOARD (sorted by F1_macro)\n{P}')
    rdf = pd.DataFrame(test_results).sort_values(SORT_COLS, ascending=False)
    rdf.index = range(1, len(rdf) + 1)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 250)
    print(rdf[METRIC_COLS + ['CV_Acc', 'Time_s']].to_string())
    rdf.drop(columns=['Estimator']).to_csv(os.path.join(OUT, 'step1_test_results.csv'), index=False)

    best_row = rdf.iloc[0]
    best_model = best_row['Estimator']
    artifact_preprocessor = None
    artifact_features = features

    if best_row['Use_Scaled']:
        artifact_preprocessor = scale_preprocessor
        if artifact_preprocessor is None:
            artifact_features = features
            X_eval = X_test
        else:
            artifact_features = get_preprocessor_feature_names(artifact_preprocessor)
            X_eval = pd.DataFrame(
                artifact_preprocessor.transform(X_test),
                columns=artifact_features,
                index=X_test.index,
            )
    else:
        X_eval = X_test

    holdout_pred = best_model.predict(X_eval)
    holdout_scores = get_scores(best_model, X_eval)
    holdout_res = evaluate(best_row['Model'], y_test.values, holdout_pred, holdout_scores)
    holdout_res['Selected_On'] = 'test'
    holdout_res['CV_Acc'] = best_row['CV_Acc']
    holdout_res['Params'] = best_row['Params']
    pd.DataFrame([holdout_res]).to_csv(os.path.join(OUT, 'step1_selected_result.csv'), index=False)
    artifact_path = save_model_bundle(
        os.path.join(OUT, 'step1_selected_bundle.joblib'),
        best_model,
        artifact_features,
        artifact_preprocessor,
        data_path=DATA_PATH,
    )

    print(f'\n{P}\n CLASSIFICATION REPORT - {best_row["Model"]} (test)\n{P}')
    print(classification_report(y_test, holdout_pred, target_names=['DOWN', 'UP']))
    print(f' Test: Acc={holdout_res["Accuracy"]:.4f} F1m={holdout_res["F1_macro"]:.4f} AUC={holdout_res["AUC"]:.4f}')

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, holdout_pred)
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'{best_row["Model"]}\nTest Acc={holdout_res["Accuracy"]:.3f} F1={holdout_res["F1_macro"]:.3f}', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['DOWN', 'UP'])
    ax.set_yticklabels(['DOWN', 'UP'])
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center',
                    fontsize=12, color='white' if cm[ii, jj] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'step1_confusion.png'), dpi=130, bbox_inches='tight')
    plt.close()

    if holdout_scores is not None:
        fig, ax = plt.subplots(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_test, holdout_scores)
        auc_val = roc_auc_score(y_test, holdout_scores)
        ax.plot(fpr, tpr, lw=1.5, label=f'{best_row["Model"]} (AUC={auc_val:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curve - Selected Baseline (Test)')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'step1_roc.png'), dpi=130, bbox_inches='tight')
        plt.close()

    if hasattr(best_model, 'feature_importances_'):
        imp = best_model.feature_importances_
        imp_norm = imp / imp.sum() if imp.sum() != 0 else imp
        imp_df = pd.DataFrame({
            'feature': features,
            'importance': imp_norm,
        }).sort_values('importance', ascending=False)
        print(f'\n{P}\n FEATURE IMPORTANCE - {best_row["Model"]}\n{P}')
        print(imp_df.head(25).round(4).to_string(index=False))
        imp_df.to_csv(os.path.join(OUT, 'step1_feature_importance.csv'), index=False)

        top20 = imp_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top20['feature'], top20['importance'], color='#4C72B0')
        ax.invert_yaxis()
        ax.set_title(f'Feature Importance - {best_row["Model"]} (Top 20)')
        ax.set_xlabel('Normalized importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, 'step1_importance.png'), dpi=130, bbox_inches='tight')
        plt.close()

    print(f'\n{P}\n BACKTEST (Long if UP, flat if DOWN)\n{P}')
    buy_hold = np.cumprod(1 + y_test_ret) - 1
    strat_ret = holdout_pred * y_test_ret
    cum = np.cumprod(1 + strat_ret) - 1
    peak = np.maximum.accumulate(np.cumprod(1 + strat_ret))
    dd = (np.cumprod(1 + strat_ret) - peak) / peak
    maxdd = dd.min() * 100
    total = cum[-1] * 100
    sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
    acc = accuracy_score(y_test, holdout_pred) * 100
    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9} {"Acc%":>7}')
    print(f' {"-"*56}')
    print(f' {"Buy&Hold":<20} {buy_hold[-1]*100:>9.2f} {"":>8} {"":>9} {"":>7}')
    print(f' {best_row["Model"]:<20} {total:>9.2f} {sharpe:>8.2f} {maxdd:>9.2f} {acc:>6.1f}%')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_test, buy_hold * 100, 'k--', lw=1.5, label='Buy & Hold', alpha=0.7)
    ax.plot(dates_test, cum * 100, lw=1.3, label=f'{best_row["Model"]} ({total:.1f}%)', alpha=0.9)
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Backtest - Selected Baseline (Long if UP, flat if DOWN)')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'step1_backtest.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Saved model bundle: {artifact_path}')
    print(f' Results saved to {OUT}')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
