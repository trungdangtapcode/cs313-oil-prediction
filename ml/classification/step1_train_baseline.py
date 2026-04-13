"""
STEP 1: Baseline Classification

This is the baseline classification benchmark for the project.
It trains a standard set of classifiers on the original feature set
without adding technical indicators or extra feature-selection logic.

Goal of this step:
  - Establish a clean baseline for predicting daily oil direction
  - Compare common model families on the same train/test split
  - Produce reference metrics before later improvement steps

Target used in this file:
  - Binary classification: oil_return > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base feature set loaded from config.load_data()
  - Raw same-day price columns and some leakage-prone columns are already dropped in config
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
  - Each model is tuned with GridSearchCV
  - TimeSeriesSplit is used for time-aware cross-validation
  - Results are evaluated with Accuracy, F1, F1_macro, AUC, and confusion-matrix stats

Outputs:
  - Console summary table
  - results/classification_results.csv
  - results/classification_feature_importance.csv
  - plots such as ROC, confusion matrix, backtest, and feature importance

Usage:
  python ml/classification/step1_train_baseline.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import load_data, get_tscv, RANDOM_STATE

P = '=' * 90

def evaluate(name, y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='binary')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec_up = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_up  = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {
        'Model': name, 'Accuracy': acc, 'F1_binary': f1, 'F1_macro': f1_macro,
        'AUC': auc, 'Precision_UP': prec_up, 'Recall_UP': rec_up,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def get_models():
    return [
        ('LogisticRegression', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000), {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga']
        }, True),
        ('SVM_RBF', SVC(random_state=RANDOM_STATE, probability=True), {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }, True),
        ('SVM_Linear', SVC(random_state=RANDOM_STATE, probability=True, kernel='linear'), {
            'C': [0.01, 0.1, 1.0, 10.0],
        }, True),
        ('RandomForest', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), {
            'n_estimators': [100, 300],
            'max_depth': [5, 10, 15],
            'min_samples_leaf': [3, 5, 10]
        }, False),
        ('GradientBoosting', GradientBoostingClassifier(random_state=RANDOM_STATE), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }, False),
        ('XGBoost', XGBClassifier(random_state=RANDOM_STATE, verbosity=0, n_jobs=1, eval_metric='logloss'), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'reg_alpha': [0, 0.1]
        }, False),
        ('LightGBM', LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1, importance_type='gain'), {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31]
        }, False),
        ('MLP', MLPClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True), {
            'hidden_layer_sizes': [(64, 32), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }, True),
    ]


def main():
    print(f'\n{P}\n CLASSIFICATION - Oil Price Direction (UP/DOWN)\n{P}')

    data = load_data()
    tscv = get_tscv()
    models = get_models()

    # Create binary target: 1 = UP (return > 0), 0 = DOWN (return <= 0)
    y_train = (data['y_train'] > 0).astype(int)
    y_test  = (data['y_test'] > 0).astype(int)

    print(f'\n  Target distribution (Train): UP={y_train.sum()} ({y_train.mean():.1%}) | DOWN={len(y_train)-y_train.sum()} ({1-y_train.mean():.1%})')
    print(f'  Target distribution (Test):  UP={y_test.sum()} ({y_test.mean():.1%}) | DOWN={len(y_test)-y_test.sum()} ({1-y_test.mean():.1%})')

    results = []
    predictions = {}
    probabilities = {}
    best_params = {}
    importances = {}

    for name, model, grid, use_sc in models:
        print(f'\n--- {name} ---')
        t0 = time.time()

        X_tr = data['X_train_sc'] if use_sc else data['X_train']
        X_te = data['X_test_sc'] if use_sc else data['X_test']

        if grid:
            gs = GridSearchCV(model, grid, cv=tscv, scoring='accuracy',
                              refit=True, n_jobs=1)
            gs.fit(X_tr, y_train)
            best = gs.best_estimator_
            best_params[name] = gs.best_params_
            cv_acc = gs.best_score_
            print(f'  Best params: {gs.best_params_}')
            print(f'  CV Accuracy: {cv_acc:.4f}')
        else:
            best = model
            best.fit(X_tr, y_train)
            best_params[name] = {}

        y_pred = best.predict(X_te)
        predictions[name] = y_pred

        # Probabilities
        y_prob = None
        if hasattr(best, 'predict_proba'):
            y_prob = best.predict_proba(X_te)[:, 1]
        elif hasattr(best, 'decision_function'):
            y_prob = best.decision_function(X_te)
        probabilities[name] = y_prob

        res = evaluate(name, y_test.values, y_pred, y_prob)
        res['Time_s'] = round(time.time() - t0, 1)
        results.append(res)

        print(f'  Accuracy:    {res["Accuracy"]:.4f}')
        print(f'  F1 (binary): {res["F1_binary"]:.4f}')
        print(f'  F1 (macro):  {res["F1_macro"]:.4f}')
        print(f'  AUC:         {res["AUC"]:.4f}')
        print(f'  Confusion:   TP={res["TP"]} FP={res["FP"]} TN={res["TN"]} FN={res["FN"]}')
        print(f'  Time:        {res["Time_s"]}s')

        # Feature importance
        if hasattr(best, 'feature_importances_'):
            imp = best.feature_importances_
            imp_norm = imp / imp.sum()
            importances[name] = imp_norm

    # â”€â”€ Results Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\n{P}\n CLASSIFICATION RESULTS (sorted by F1_macro)\n{P}')
    rdf = pd.DataFrame(results).sort_values('F1_macro', ascending=False)
    rdf.index = range(1, len(rdf) + 1)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 250)
    print(rdf[['Model', 'Accuracy', 'F1_binary', 'F1_macro', 'AUC',
               'Precision_UP', 'Recall_UP', 'TP', 'FP', 'TN', 'FN', 'Time_s']].to_string())
    rdf.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_results.csv'), index=False)

    # â”€â”€ Classification Report (best model) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_name = rdf.iloc[0]['Model']
    print(f'\n{P}\n CLASSIFICATION REPORT - {best_name}\n{P}')
    print(classification_report(y_test, predictions[best_name], target_names=['DOWN', 'UP']))

    # â”€â”€ Confusion Matrix Plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    af = axes.flatten()
    for i, res in enumerate(rdf.head(8).itertuples()):
        ax = af[i]
        cm = confusion_matrix(y_test, predictions[res.Model])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{res.Model}\nAcc={res.Accuracy:.3f} F1={res.F1_macro:.3f}', fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['DOWN', 'UP'], fontsize=7)
        ax.set_yticklabels(['DOWN', 'UP'], fontsize=7)
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center',
                        fontsize=12, color='white' if cm[ii, jj] > cm.max()/2 else 'black')
    for j in range(len(rdf.head(8)), 8):
        af[j].set_visible(False)
    fig.suptitle('Confusion Matrices - Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_confusion.png'), dpi=130, bbox_inches='tight')
    plt.close()

    # â”€â”€ ROC Curves â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, ax = plt.subplots(figsize=(8, 8))
    from sklearn.metrics import roc_curve
    for name in rdf['Model'].values:
        prob = probabilities[name]
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc_val = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, lw=1.2, label=f'{name} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves - Classification')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_roc.png'), dpi=130, bbox_inches='tight')
    plt.close()

    # â”€â”€ Feature Importance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if importances:
        print(f'\n{P}\n FEATURE IMPORTANCE (Tree models, normalized)\n{P}')
        imp_df = pd.DataFrame(importances, index=data['features'])
        imp_df['Mean'] = imp_df.mean(axis=1)
        imp_df.sort_values('Mean', ascending=False, inplace=True)
        print(imp_df.round(4).to_string())
        imp_df.to_csv(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_feature_importance.csv'))

        top20 = imp_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        x = np.arange(len(top20))
        w = 0.2
        for i, col in enumerate([c for c in imp_df.columns if c != 'Mean']):
            ax.barh(x + i * w, top20[col], w, label=col, alpha=0.8)
        ax.set_yticks(x + w * 1.5)
        ax.set_yticklabels(top20.index, fontsize=7)
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.set_title('Feature Importance - Classification (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_importance.png'), dpi=130, bbox_inches='tight')
        plt.close()

    # â”€â”€ Backtest (signal = predicted direction) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\n{P}\n BACKTEST (Long if UP, flat if DOWN)\n{P}')
    y_test_ret = data['y_test'].values  # actual returns
    dates = data['dates_test']
    buy_hold = np.cumprod(1 + y_test_ret) - 1

    print(f' {"Model":<20} {"Total%":>9} {"Sharpe":>8} {"MaxDD%":>9} {"Acc%":>7}')
    print(f' {"-"*56}')
    print(f' {"Buy&Hold":<20} {buy_hold[-1]*100:>9.2f} {"":>8} {"":>9} {"":>7}')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, buy_hold * 100, 'k--', lw=1.5, label='Buy & Hold', alpha=0.7)

    for name in rdf['Model'].values[:5]:
        pred = predictions[name]
        # Long when predict UP (1), flat when predict DOWN (0)
        strat_ret = pred * y_test_ret
        cum = np.cumprod(1 + strat_ret) - 1
        peak = np.maximum.accumulate(np.cumprod(1 + strat_ret))
        dd = (np.cumprod(1 + strat_ret) - peak) / peak
        maxdd = dd.min() * 100
        total = cum[-1] * 100
        sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)
        acc = accuracy_score(y_test, pred) * 100
        print(f' {name:<20} {total:>9.2f} {sharpe:>8.2f} {maxdd:>9.2f} {acc:>6.1f}%')
        ax.plot(dates, cum * 100, lw=1, label=f'{name} ({total:.1f}%)', alpha=0.8)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Backtest - Top 5 Classification Models (Long if UP, flat if DOWN)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), 'results'), 'classification_backtest.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Results saved to ml/results/')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
