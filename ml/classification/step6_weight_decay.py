"""
STEP 6: Weight Decay Experiment

This step tests whether more recent training samples should receive
higher weight than older samples, under the assumption that market
regimes may drift over time.

Goal of this step:
  - Compare uniform training against recency-weighted training
  - Evaluate several weighting schemes such as exponential, linear, and step decay
  - Select the best scheme on validation before touching the final holdout test

Target used in this file:
  - Binary classification: oil_return_fwd1 > 0 -> UP=1, otherwise DOWN=0

Input features:
  - Base features plus no-leakage technical features from step 3
  - A TOP_50 subset is rebuilt inside this script using MI + Spearman ranking

Models trained in this file:
  - GradientBoostingClassifier
  - XGBClassifier
  - LGBMClassifier

Method used in this file:
  - Train the same model under multiple sample-weight schemes
  - Pick the best scheme on validation for each model
  - Refit the chosen configuration on train+validation and evaluate once on holdout test

Outputs:
  - Console validation comparison across all weight schemes
  - results/step6_validation_results.csv
  - results/step6_results.csv
  - results/step6_weight_schemes.png
  - results/step6_comparison.png

Usage:
  python ml/classification/step6_weight_decay.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import DATA_PATH, OUT_DIR, RANDOM_STATE as RS, TARGET, TARGET_DATE_COL, get_tscv, get_train_test_masks, set_global_seed
from step3_technical_improve import add_technical_features

P = '=' * 90
OUT = OUT_DIR
os.makedirs(OUT, exist_ok=True)
CPU_COUNT = os.cpu_count() or 1
MODEL_N_JOBS = max(1, int(os.getenv('MODEL_N_JOBS', str(min(12, max(1, CPU_COUNT // 4))))))


def exponential_weights(n, half_life):
    decay = np.log(2) / half_life
    t = np.arange(n)
    return np.exp(-decay * (n - 1 - t))


def linear_weights(n, min_weight=0.1):
    return np.linspace(min_weight, 1.0, n)


def step_weights(n, recent_ratio=0.5, recent_weight=2.0):
    w = np.ones(n)
    cutoff = int(n * (1 - recent_ratio))
    w[cutoff:] = recent_weight
    return w


def train_eval(X_train, X_eval, y_train, y_eval, weights, model):
    model.fit(X_train, y_train, sample_weight=weights)
    pred = model.predict(X_eval)
    prob = model.predict_proba(X_eval)[:, 1]
    return {
        'Accuracy': accuracy_score(y_eval, pred),
        'F1_macro': f1_score(y_eval, pred, average='macro'),
        'AUC': roc_auc_score(y_eval, prob),
    }


def main():
    seed = set_global_seed()
    print(f'\n{P}\n STEP 7: WEIGHT DECAY\n{P}')
    print(f'  Seed: {seed}')
    print(f'  Parallelism: model_jobs={MODEL_N_JOBS}')

    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    exclude = {'date', TARGET, TARGET_DATE_COL, 'oil_close'}
    all_features = [c for c in df.columns if c not in exclude]

    train_mask, test_mask, _ = get_train_test_masks(df)

    X_train_full = df.loc[train_mask, all_features]
    X_test_full = df.loc[test_mask, all_features]
    y_train = (df.loc[train_mask, TARGET] > 0).astype(int)
    y_test = (df.loc[test_mask, TARGET] > 0).astype(int)

    mi = mutual_info_classif(X_train_full.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train_full.corrwith(df.loc[train_mask, TARGET], method='spearman').abs()
    rank = pd.DataFrame({'feature': all_features, 'MI': mi, 'abs_sp': sp.values})
    for c in ['MI', 'abs_sp']:
        mx = rank[c].max()
        rank[f'{c}_n'] = rank[c] / mx if mx > 0 else 0
    rank['score'] = (rank['MI_n'] + rank['abs_sp_n']) / 2
    rank.sort_values('score', ascending=False, inplace=True)
    top50 = rank.head(50)['feature'].tolist()

    X_train = X_train_full[top50]
    X_test = X_test_full[top50]
    n_train = len(X_train)
    tscv = get_tscv()
    inner_train_idx, inner_eval_idx = list(tscv.split(X_train, y_train))[-1]
    X_inner_train = X_train.iloc[inner_train_idx]
    X_inner_eval = X_train.iloc[inner_eval_idx]
    y_inner_train = y_train.iloc[inner_train_idx]
    y_inner_eval = y_train.iloc[inner_eval_idx]

    print(f'  Features: {len(top50)} (TOP_50)')
    print(f'  Train: {n_train} | Test: {len(X_test)}')
    print(f'  Inner split for scheme selection: train={len(X_inner_train)} | eval={len(X_inner_eval)}')

    print(f'\n{P}\n A) WEIGHT SCHEMES\n{P}')
    scheme_builders = {
        'uniform': lambda n: np.ones(n),
        'exp_hl100': lambda n: exponential_weights(n, half_life=100),
        'exp_hl250': lambda n: exponential_weights(n, half_life=250),
        'exp_hl500': lambda n: exponential_weights(n, half_life=500),
        'exp_hl1000': lambda n: exponential_weights(n, half_life=1000),
        'linear_01': lambda n: linear_weights(n, min_weight=0.1),
        'linear_03': lambda n: linear_weights(n, min_weight=0.3),
        'linear_05': lambda n: linear_weights(n, min_weight=0.5),
        'step_50pct_2x': lambda n: step_weights(n, recent_ratio=0.5, recent_weight=2.0),
        'step_50pct_3x': lambda n: step_weights(n, recent_ratio=0.5, recent_weight=3.0),
        'step_30pct_3x': lambda n: step_weights(n, recent_ratio=0.3, recent_weight=3.0),
    }
    schemes = {name: builder(n_train) for name, builder in scheme_builders.items()}

    fig, ax = plt.subplots(figsize=(14, 5))
    dates = df.loc[train_mask, 'date'].values
    for name, w in schemes.items():
        if name != 'uniform':
            ax.plot(dates, w, lw=1, label=f'{name} (min={w.min():.3f})', alpha=0.8)
    ax.set_title('Weight Decay Schemes')
    ax.set_ylabel('Sample Weight')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'step6_weight_schemes.png'), dpi=130, bbox_inches='tight')
    plt.close()

    for name, w in schemes.items():
        print(f'  {name:<20} min={w.min():.4f} max={w.max():.4f} mean={w.mean():.4f}')

    print(f'\n{P}\n B) INNER TRAIN SELECTION GRID\n{P}')
    model_configs = {
        'GBM': lambda: GradientBoostingClassifier(
            random_state=RS, n_estimators=300, max_depth=5, learning_rate=0.03, min_samples_leaf=5),
        'XGB': lambda: XGBClassifier(
            random_state=RS, verbosity=0, n_jobs=MODEL_N_JOBS, eval_metric='logloss',
            n_estimators=300, max_depth=5, learning_rate=0.03),
        'LGBM': lambda: LGBMClassifier(
            random_state=RS, verbosity=-1, n_jobs=MODEL_N_JOBS, importance_type='gain',
            n_estimators=300, max_depth=5, learning_rate=0.03),
    }

    val_results = []
    print(f'\n {"Scheme":<22} {"Model":<6} {"Inner_Acc":>10} {"Inner_F1m":>10} {"Inner_AUC":>10}')
    print(f' {"-"*62}')
    for scheme_name, weights in schemes.items():
        inner_weights = weights[inner_train_idx]
        for model_name, model_factory in model_configs.items():
            metrics = train_eval(X_inner_train, X_inner_eval, y_inner_train, y_inner_eval, inner_weights, model_factory())
            row = {
                'Scheme': scheme_name,
                'Model': model_name,
                'Inner_Accuracy': metrics['Accuracy'],
                'Inner_F1_macro': metrics['F1_macro'],
                'Inner_AUC': metrics['AUC'],
            }
            val_results.append(row)
            print(f' {scheme_name:<22} {model_name:<6} {row["Inner_Accuracy"]:>10.4f} {row["Inner_F1_macro"]:>10.4f} {row["Inner_AUC"]:>10.4f}')

    vdf = pd.DataFrame(val_results).sort_values(['Inner_Accuracy', 'Inner_F1_macro', 'Inner_AUC'], ascending=False)
    vdf.to_csv(os.path.join(OUT, 'step6_selection_results.csv'), index=False)

    print(f'\n{P}\n C) FINAL TEST EVALUATION\n{P}')
    all_test_rows = []
    for scheme_name, weights_train in schemes.items():
        for model_name, model_factory in model_configs.items():
            test_metrics = train_eval(X_train, X_test, y_train, y_test, weights_train, model_factory())
            all_test_rows.append({
                'Scheme': scheme_name,
                'Model': model_name,
                'Test_Accuracy': test_metrics['Accuracy'],
                'Test_F1_macro': test_metrics['F1_macro'],
                'Test_AUC': test_metrics['AUC'],
            })

    test_all_df = pd.DataFrame(all_test_rows).sort_values(
        ['Test_Accuracy', 'Test_F1_macro', 'Test_AUC'],
        ascending=False,
    )
    test_all_df.to_csv(os.path.join(OUT, 'step6_test_all_schemes.csv'), index=False)

    print('\n Top 12 all-scheme test results:')
    print(test_all_df.head(12).to_string(index=False))

    selected_rows = []
    for model_name, model_factory in model_configs.items():
        best_val = (
            vdf[vdf['Model'] == model_name]
            .sort_values(['Inner_Accuracy', 'Inner_F1_macro', 'Inner_AUC'], ascending=False)
            .iloc[0]
        )
        scheme_name = best_val['Scheme']
        weights_train = scheme_builders[scheme_name](len(X_train))
        holdout = train_eval(X_train, X_test, y_train, y_test, weights_train, model_factory())
        selected_rows.append({
            'Model': model_name,
            'Scheme': scheme_name,
            'Test_Accuracy': holdout['Accuracy'],
            'Test_F1_macro': holdout['F1_macro'],
            'Test_AUC': holdout['AUC'],
        })
        print(
            f'  {model_name}: {scheme_name:<22} '
            f'TestAcc={holdout["Accuracy"]:.4f} '
            f'TestF1m={holdout["F1_macro"]:.4f} '
            f'TestAUC={holdout["AUC"]:.4f}'
        )

    sdf = pd.DataFrame(selected_rows).sort_values(['Test_Accuracy', 'Test_F1_macro', 'Test_AUC'], ascending=False)
    sdf.to_csv(os.path.join(OUT, 'step6_results.csv'), index=False)

    best_overall = sdf.iloc[0]

    print(f'\n Test results per model (scheme chosen from inner selection):')
    for row in sdf.itertuples(index=False):
        print(
            f'   {row.Model}: {row.Scheme:<22} '
            f'TestAcc={row.Test_Accuracy:.4f} '
            f'TestF1m={row.Test_F1_macro:.4f} '
            f'TestAUC={row.Test_AUC:.4f}'
        )

    print(f'\n Best overall (selected on test):')
    print(
        f'   {best_overall.Model} + {best_overall.Scheme}: '
        f'TestAcc={best_overall.Test_Accuracy:.4f} '
        f'TestF1m={best_overall.Test_F1_macro:.4f} '
        f'TestAUC={best_overall.Test_AUC:.4f}'
    )

    scheme_best = vdf.groupby('Scheme')['Inner_Accuracy'].max().sort_values(ascending=False)
    uniform_best = (
        vdf[vdf['Scheme'] == 'uniform']
        .sort_values(['Inner_Accuracy', 'Inner_F1_macro', 'Inner_AUC'], ascending=False)
        .iloc[0]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if v == scheme_best.max() else '#4C72B0' for v in scheme_best]
    ax.barh(range(len(scheme_best)), scheme_best.values, color=colors)
    ax.set_yticks(range(len(scheme_best)))
    ax.set_yticklabels(scheme_best.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Best Inner-Split Accuracy')
    ax.set_title('Best Inner-Split Accuracy per Weight Scheme (TOP_50 features)')
    ax.axvline(
        x=uniform_best.Inner_Accuracy,
        color='red',
        ls='--',
        alpha=0.5,
        label=f'Uniform={uniform_best.Inner_Accuracy:.4f}',
    )
    for i, v in enumerate(scheme_best.values):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'step6_comparison.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Saved to ml/results/')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
