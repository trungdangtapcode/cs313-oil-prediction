"""
STEP 7: Sample Weight Decay - data gần hơn quan trọng hơn
Usage: python ml/step7_weight_decay.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE as RS, DATA_PATH, SPLIT_DATE, OUT_DIR
from step4_improve import add_technical_features

P = '=' * 90


def exponential_weights(n, half_life):
    """
    Tạo weights exponential decay.
    half_life: số sample mà weight giảm còn 50%.
    Sample cuối cùng (gần nhất) có weight = 1.0.
    """
    decay = np.log(2) / half_life
    t = np.arange(n)
    w = np.exp(-decay * (n - 1 - t))  # t=0 xa nhất, t=n-1 gần nhất
    return w


def linear_weights(n, min_weight=0.1):
    """Weight tăng tuyến tính từ min_weight đến 1.0"""
    return np.linspace(min_weight, 1.0, n)


def step_weights(n, recent_ratio=0.5, recent_weight=2.0):
    """Chia 2 giai đoạn: cũ weight=1, gần weight=recent_weight"""
    w = np.ones(n)
    cutoff = int(n * (1 - recent_ratio))
    w[cutoff:] = recent_weight
    return w


def train_eval(X_train, X_test, y_train, y_test, weights, model, name):
    model.fit(X_train, y_train, sample_weight=weights)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average='macro')
    auc = roc_auc_score(y_test, prob)
    return {'Model': name, 'Accuracy': acc, 'F1_macro': f1m, 'AUC': auc}


def main():
    print(f'\n{P}\n STEP 7: WEIGHT DECAY\n{P}')

    # Load + technicals
    df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    df = add_technical_features(df)

    # Dùng TOP_50 features (best từ step5)
    from sklearn.feature_selection import mutual_info_classif

    exclude = {'date', 'oil_return', 'oil_close'}
    all_features = [c for c in df.columns if c not in exclude]

    train_mask = df['date'] < SPLIT_DATE
    test_mask = df['date'] >= SPLIT_DATE

    X_train_full = df.loc[train_mask, all_features]
    X_test_full = df.loc[test_mask, all_features]
    y_train = (df.loc[train_mask, 'oil_return'] > 0).astype(int)
    y_test = (df.loc[test_mask, 'oil_return'] > 0).astype(int)

    # Feature ranking (MI + Spearman) → TOP_50
    mi = mutual_info_classif(X_train_full.fillna(0), y_train, random_state=RS, n_neighbors=5)
    sp = X_train_full.corrwith(df.loc[train_mask, 'oil_return'], method='spearman').abs()
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

    print(f'  Features: {len(top50)} (TOP_50)')
    print(f'  Train: {n_train} | Test: {len(X_test)}')

    # ═══════════════════════════════════════
    # A) VISUALIZE WEIGHT SCHEMES
    # ═══════════════════════════════════════
    print(f'\n{P}\n A) WEIGHT SCHEMES\n{P}')

    schemes = {
        'uniform': np.ones(n_train),
        'exp_hl100': exponential_weights(n_train, half_life=100),
        'exp_hl250': exponential_weights(n_train, half_life=250),
        'exp_hl500': exponential_weights(n_train, half_life=500),
        'exp_hl1000': exponential_weights(n_train, half_life=1000),
        'linear_01': linear_weights(n_train, min_weight=0.1),
        'linear_03': linear_weights(n_train, min_weight=0.3),
        'linear_05': linear_weights(n_train, min_weight=0.5),
        'step_50pct_2x': step_weights(n_train, recent_ratio=0.5, recent_weight=2.0),
        'step_50pct_3x': step_weights(n_train, recent_ratio=0.5, recent_weight=3.0),
        'step_30pct_3x': step_weights(n_train, recent_ratio=0.3, recent_weight=3.0),
    }

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
    plt.savefig(os.path.join(OUT_DIR, 'step7_weight_schemes.png'), dpi=130, bbox_inches='tight')
    plt.close()

    for name, w in schemes.items():
        print(f'  {name:<20} min={w.min():.4f} max={w.max():.4f} mean={w.mean():.4f}')

    # ═══════════════════════════════════════
    # B) TRAIN WITH EACH WEIGHT SCHEME
    # ═══════════════════════════════════════
    print(f'\n{P}\n B) TRAIN ALL COMBINATIONS\n{P}')

    model_configs = {
        'GBM': lambda: GradientBoostingClassifier(
            random_state=RS, n_estimators=300, max_depth=5, learning_rate=0.03, min_samples_leaf=5),
        'XGB': lambda: XGBClassifier(
            random_state=RS, verbosity=0, n_jobs=2, eval_metric='logloss',
            n_estimators=300, max_depth=5, learning_rate=0.03),
        'LGBM': lambda: LGBMClassifier(
            random_state=RS, verbosity=-1, n_jobs=2, importance_type='gain',
            n_estimators=300, max_depth=5, learning_rate=0.03),
    }

    results = []
    print(f'\n {"Scheme":<22} {"Model":<6} {"Accuracy":>10} {"F1_macro":>10} {"AUC":>8}')
    print(f' {"-"*60}')

    for scheme_name, weights in schemes.items():
        for mname, mfunc in model_configs.items():
            model = mfunc()
            res = train_eval(X_train, X_test, y_train, y_test, weights, model, mname)
            res['Scheme'] = scheme_name
            results.append(res)
            print(f' {scheme_name:<22} {mname:<6} {res["Accuracy"]:>10.4f} {res["F1_macro"]:>10.4f} {res["AUC"]:>8.4f}')

    # ═══════════════════════════════════════
    # C) SUMMARY
    # ═══════════════════════════════════════
    print(f'\n{P}\n C) SUMMARY\n{P}')

    rdf = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    rdf.to_csv(os.path.join(OUT_DIR, 'step7_results.csv'), index=False)

    # Best per model
    print(f'\n Best per model:')
    for m in ['GBM', 'XGB', 'LGBM']:
        sub = rdf[rdf.Model == m].iloc[0]
        print(f'   {m}: {sub.Scheme:<22} Acc={sub.Accuracy:.4f} F1m={sub.F1_macro:.4f} AUC={sub.AUC:.4f}')

    # Best overall
    best = rdf.iloc[0]
    print(f'\n Best overall:')
    print(f'   {best.Model} + {best.Scheme}: Acc={best.Accuracy:.4f} F1m={best.F1_macro:.4f} AUC={best.AUC:.4f}')

    # Uniform comparison
    uniform_best = rdf[rdf.Scheme == 'uniform'].iloc[0]
    print(f'\n Uniform (no decay):')
    print(f'   {uniform_best.Model}: Acc={uniform_best.Accuracy:.4f}')
    print(f'\n Improvement: {(best.Accuracy - uniform_best.Accuracy)*100:+.2f}%')

    # Bar chart: best accuracy per scheme
    scheme_best = rdf.groupby('Scheme')['Accuracy'].max().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if v == scheme_best.max() else '#4C72B0' for v in scheme_best]
    ax.barh(range(len(scheme_best)), scheme_best.values, color=colors)
    ax.set_yticks(range(len(scheme_best)))
    ax.set_yticklabels(scheme_best.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Best Accuracy')
    ax.set_title('Best Accuracy per Weight Scheme (TOP_50 features)')
    ax.axvline(x=uniform_best.Accuracy, color='red', ls='--', alpha=0.5, label=f'Uniform={uniform_best.Accuracy:.4f}')
    for i, v in enumerate(scheme_best.values):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'step7_comparison.png'), dpi=130, bbox_inches='tight')
    plt.close()

    print(f'\n Saved to ml/results/')
    print(f'{P}\n DONE\n{P}')


if __name__ == '__main__':
    main()
