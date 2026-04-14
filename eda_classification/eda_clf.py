"""
EDA Classification - Oil Price Direction (UP/DOWN)
Target: oil_return_fwd1 > 0 -> UP(1), else DOWN(0)
Plots saved to eda_classification/
"""
import warnings, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))

import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'figure.facecolor': 'white'})
sns.set_style('whitegrid')

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.getenv('EDA_DATA_PATH', os.path.join(ROOT, 'data', 'processed', 'dataset_step4_transformed.csv'))
OUT = os.getenv('EDA_OUT_DIR', os.path.dirname(__file__)).strip()
PREFIX = os.getenv('EDA_PREFIX', '').strip()
RUN_LABEL = os.getenv('EDA_RUN_LABEL', PREFIX or os.path.splitext(os.path.basename(DATA))[0]).strip()
TARGET_RET = 'oil_return_fwd1'
TARGET_DATE = 'oil_return_fwd1_date'

os.makedirs(OUT, exist_ok=True)

def save(name):
    out_name = f'{PREFIX}_{name}' if PREFIX else name
    plt.savefig(os.path.join(OUT, out_name), bbox_inches='tight', dpi=130); plt.close('all')
    print(f'    -> {out_name}')

P = '=' * 90
SPLIT = '2023-01-01'

# ─── LOAD ────────────────────────────────────────────────────
print(f'\n{P}\n LOAD\n{P}')
print(f'  Run label: {RUN_LABEL}')
print(f'  Dataset: {DATA}')
parse_dates = ['date']
if os.path.exists(DATA):
    probe = pd.read_csv(DATA, nrows=0)
    if TARGET_DATE in probe.columns:
        parse_dates.append(TARGET_DATE)
df = pd.read_csv(DATA, parse_dates=parse_dates).sort_values('date').reset_index(drop=True)

# Target
df['direction'] = (df[TARGET_RET] > 0).astype(int)  # 1=UP, 0=DOWN
split_col = TARGET_DATE if TARGET_DATE in df.columns else 'date'
split_values = pd.to_datetime(df[split_col])

train = df[split_values < SPLIT].copy()
test  = df[split_values >= SPLIT].copy()

NUM = [c for c in df.columns if c not in ['date', TARGET_RET, TARGET_DATE, 'direction']]
TARGET = 'direction'

print(f'  Shape: {df.shape} | Train: {len(train)} | Test: {len(test)}')
print(f'  Features: {len(NUM)}')
if split_col != 'date':
    print(f'  Train targets: {train[split_col].iloc[0].date()} -> {train[split_col].iloc[-1].date()}')
    print(f'  Test targets:  {test[split_col].iloc[0].date()} -> {test[split_col].iloc[-1].date()}')

# ═══════════════════════════════════════════════════════════════
# 1. CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 1. CLASS DISTRIBUTION\n{P}')

for name, subset in [('Full', df), ('Train', train), ('Test', test)]:
    vc = subset[TARGET].value_counts()
    print(f'  {name:>6}: UP={vc.get(1,0)} ({vc.get(1,0)/len(subset):.1%})  DOWN={vc.get(0,0)} ({vc.get(0,0)/len(subset):.1%})')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, subset) in enumerate([('Full', df), ('Train', train), ('Test', test)]):
    vc = subset[TARGET].value_counts().sort_index()
    axes[i].bar(['DOWN (0)', 'UP (1)'], vc.values, color=['#C44E52', '#55A868'], edgecolor='white')
    axes[i].set_title(f'{name} (n={len(subset)})')
    for j, v in enumerate(vc.values):
        axes[i].text(j, v + 5, f'{v}\n({v/len(subset):.1%})', ha='center', fontsize=9)
fig.suptitle('Class Distribution: UP vs DOWN', fontsize=14)
plt.tight_layout(); save('01_class_distribution.png')

# By year
yearly = train.groupby(train.date.dt.year)[TARGET].value_counts().unstack(fill_value=0)
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
yearly.plot(kind='bar', stacked=True, ax=axes[0], color=['#C44E52', '#55A868'])
axes[0].set_title('Class Count by Year (Train)'); axes[0].legend(['DOWN', 'UP'])
yearly_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#C44E52', '#55A868'])
axes[1].set_title('Class Proportion by Year (Train)'); axes[1].axhline(0.5, color='black', ls='--', alpha=0.5)
plt.tight_layout(); save('02_class_by_year.png')

# ═══════════════════════════════════════════════════════════════
# 2. FEATURE DISTRIBUTIONS BY CLASS (UP vs DOWN)
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 2. FEATURE DISTRIBUTIONS BY CLASS\n{P}')

nc = 6; nr = (len(NUM) + nc - 1) // nc
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.5))
af = axes.flatten()
for i, c in enumerate(NUM):
    ax = af[i]
    up = train.loc[train[TARGET] == 1, c].dropna()
    down = train.loc[train[TARGET] == 0, c].dropna()
    ax.hist(down, bins=40, alpha=0.5, density=True, color='#C44E52', label='DOWN')
    ax.hist(up, bins=40, alpha=0.5, density=True, color='#55A868', label='UP')
    # KS test
    ks_s, ks_p = stats.ks_2samp(up, down)
    ax.set_title(f'{c}\nKS p={ks_p:.2e}', fontsize=6, fontweight='bold')
    ax.tick_params(labelsize=5)
    if i == 0: ax.legend(fontsize=5)
for j in range(len(NUM), len(af)): af[j].set_visible(False)
fig.suptitle('Feature Distributions: UP vs DOWN (Train)', fontsize=13, y=1.01)
plt.tight_layout(); save('03_dist_by_class_all.png')
print(f'  Plotted: {len(NUM)} features')

# KS test summary
print(f'\n  KS Test (UP vs DOWN distributions):')
print(f'  {"Feature":<35} {"KS_stat":>9} {"KS_p":>12} {"Significant":>12}')
print(f'  {"-"*70}')
ks_results = []
for c in NUM:
    up = train.loc[train[TARGET] == 1, c].dropna()
    down = train.loc[train[TARGET] == 0, c].dropna()
    ks_s, ks_p = stats.ks_2samp(up, down)
    sig = ks_p < 0.05
    ks_results.append({'feature': c, 'ks_stat': ks_s, 'ks_p': ks_p, 'sig': sig})
ks_df = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)
for _, r in ks_df.iterrows():
    print(f'  {r.feature:<35} {r.ks_stat:>9.4f} {r.ks_p:>12.2e} {"YES" if r.sig else "":>12}')
n_sig = ks_df.sig.sum()
print(f'\n  Features significantly different between UP/DOWN: {n_sig}/{len(NUM)}')

# ═══════════════════════════════════════════════════════════════
# 3. CLASS-WISE MEANS (EFFECT SIZE)
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 3. CLASS-WISE MEANS + COHEN\'S D\n{P}')

effect = []
for c in NUM:
    up = train.loc[train[TARGET] == 1, c]
    down = train.loc[train[TARGET] == 0, c]
    mu_up, mu_down = up.mean(), down.mean()
    pooled_std = np.sqrt((up.std()**2 + down.std()**2) / 2)
    d = (mu_up - mu_down) / (pooled_std + 1e-10)  # Cohen's d
    effect.append({'feature': c, 'mean_UP': mu_up, 'mean_DOWN': mu_down,
                   'diff': mu_up - mu_down, 'cohens_d': d, 'abs_d': abs(d)})
edf = pd.DataFrame(effect).sort_values('abs_d', ascending=False)

print(f'  {"Feature":<35} {"Mean_UP":>10} {"Mean_DOWN":>10} {"Cohen_d":>10}')
print(f'  {"-"*68}')
for _, r in edf.iterrows():
    print(f'  {r.feature:<35} {r.mean_UP:>10.4f} {r.mean_DOWN:>10.4f} {r.cohens_d:>10.4f}')

print(f'\n  |d| > 0.2 (small effect): {(edf.abs_d > 0.2).sum()}')
print(f'  |d| > 0.5 (medium effect): {(edf.abs_d > 0.5).sum()}')
print(f'  |d| > 0.8 (large effect): {(edf.abs_d > 0.8).sum()}')

# Bar chart
fig, ax = plt.subplots(figsize=(10, 16))
colors = ['#C44E52' if x < 0 else '#55A868' for x in edf['cohens_d']]
ax.barh(range(len(edf)), edf['cohens_d'], color=colors, height=0.8)
ax.set_yticks(range(len(edf))); ax.set_yticklabels(edf['feature'], fontsize=6)
ax.invert_yaxis(); ax.axvline(0, color='black', lw=0.5)
ax.axvline(0.2, color='green', ls=':', alpha=0.4, label='|d|=0.2 (small)')
ax.axvline(-0.2, color='green', ls=':', alpha=0.4)
ax.set_title("Cohen's d: UP vs DOWN (all features)")
ax.legend(fontsize=7)
plt.tight_layout(); save('04_cohens_d.png')

# ═══════════════════════════════════════════════════════════════
# 4. POINT-BISERIAL CORRELATION (feature vs binary target)
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 4. POINT-BISERIAL CORRELATION\n{P}')

pb = []
for c in NUM:
    r, p = stats.pointbiserialr(train[TARGET], train[c])
    pb.append({'feature': c, 'rpb': r, 'p': p, 'abs_rpb': abs(r)})
pb_df = pd.DataFrame(pb).sort_values('abs_rpb', ascending=False)

print(f'  {"Feature":<35} {"r_pb":>10} {"p-value":>12}')
print(f'  {"-"*60}')
for _, r in pb_df.iterrows():
    print(f'  {r.feature:<35} {r.rpb:>10.4f} {r.p:>12.2e}')

fig, ax = plt.subplots(figsize=(10, 16))
colors = ['#C44E52' if x < 0 else '#55A868' for x in pb_df['rpb']]
ax.barh(range(len(pb_df)), pb_df['rpb'], color=colors, height=0.8)
ax.set_yticks(range(len(pb_df))); ax.set_yticklabels(pb_df['feature'], fontsize=6)
ax.invert_yaxis(); ax.axvline(0, color='black', lw=0.5)
ax.set_title('Point-Biserial Correlation with Direction (all features)')
for i, v in enumerate(pb_df['rpb']): ax.text(v+(0.001 if v>=0 else -0.001), i, f'{v:.3f}', va='center', fontsize=5, ha='left' if v>=0 else 'right')
plt.tight_layout(); save('05_pointbiserial.png')

# ═══════════════════════════════════════════════════════════════
# 5. MUTUAL INFORMATION (feature vs binary target)
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 5. MUTUAL INFORMATION\n{P}')

from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(train[NUM].fillna(0), train[TARGET], random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': NUM, 'MI': mi}).sort_values('MI', ascending=False)

print(f'  {"#":<4} {"Feature":<35} {"MI":>8}')
print(f'  {"-"*50}')
for i, (_, r) in enumerate(mi_df.iterrows()):
    print(f'  {i+1:<4} {r.feature:<35} {r.MI:>8.4f}')

fig, ax = plt.subplots(figsize=(10, 16))
ax.barh(range(len(mi_df)), mi_df['MI'], color='#4C72B0', height=0.8)
ax.set_yticks(range(len(mi_df))); ax.set_yticklabels(mi_df['feature'], fontsize=6)
ax.invert_yaxis(); ax.set_xlabel('MI Score')
ax.set_title('Mutual Information with Direction (all features)')
plt.tight_layout(); save('06_mutual_info.png')

# ═══════════════════════════════════════════════════════════════
# 6. COMBINED RANKING (rpb + MI + KS)
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 6. COMBINED FEATURE RANKING\n{P}')

combined = pd.DataFrame({'feature': NUM})
combined = combined.merge(pb_df[['feature', 'abs_rpb']], on='feature')
combined = combined.merge(mi_df[['feature', 'MI']], on='feature')
combined = combined.merge(ks_df[['feature', 'ks_stat']], on='feature')

for c in ['abs_rpb', 'MI', 'ks_stat']:
    mx = combined[c].max()
    combined[f'{c}_n'] = combined[c] / mx if mx > 0 else 0
combined['score'] = (combined['abs_rpb_n'] + combined['MI_n'] + combined['ks_stat_n']) / 3
combined.sort_values('score', ascending=False, inplace=True)
combined.reset_index(drop=True, inplace=True)

print(f'  {"#":<4} {"Feature":<35} {"|rpb|":>8} {"MI":>8} {"KS":>8} {"Score":>8}')
print(f'  {"-"*72}')
for i, r in combined.iterrows():
    print(f'  {i+1:<4} {r.feature:<35} {r.abs_rpb:>8.4f} {r.MI:>8.4f} {r.ks_stat:>8.4f} {r.score:>8.4f}')

# ═══════════════════════════════════════════════════════════════
# 7. BOXPLOT BY CLASS — TOP 20 FEATURES
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 7. BOXPLOT BY CLASS — TOP 20\n{P}')

top20 = combined.head(20)['feature'].tolist()
fig, axes = plt.subplots(4, 5, figsize=(22, 16))
af = axes.flatten()
for i, c in enumerate(top20):
    ax = af[i]
    data = [train.loc[train[TARGET]==0, c].dropna(), train.loc[train[TARGET]==1, c].dropna()]
    bp = ax.boxplot(data, labels=['DOWN', 'UP'], patch_artist=True, widths=0.6,
                    boxprops=dict(alpha=0.7), medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    bp['boxes'][0].set_facecolor('#C44E52')
    bp['boxes'][1].set_facecolor('#55A868')
    rpb_val = combined[combined.feature == c]['abs_rpb'].values[0]
    ax.set_title(f'{c}\n|rpb|={rpb_val:.3f}', fontsize=8)
fig.suptitle('Boxplot by Class — Top 20 Features (Train)', fontsize=14)
plt.tight_layout(); save('07_boxplot_top20.png')
print(f'  Plotted: {len(top20)} features')

# ═══════════════════════════════════════════════════════════════
# 8. CLASS PROPORTION OVER TIME
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 8. CLASS PROPORTION OVER TIME\n{P}')

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Rolling 90d UP proportion
roll_up = train[TARGET].rolling(90).mean()
axes[0].plot(train['date'], roll_up, color='#55A868', lw=1)
axes[0].axhline(0.5, color='red', ls='--', alpha=0.5)
axes[0].fill_between(train['date'], roll_up, 0.5, where=roll_up > 0.5, alpha=0.2, color='#55A868')
axes[0].fill_between(train['date'], roll_up, 0.5, where=roll_up < 0.5, alpha=0.2, color='#C44E52')
axes[0].set_ylabel('P(UP) 90d rolling')
axes[0].set_title('Rolling 90-day UP Proportion (Train)')
for d, l in [('2020-03', 'COVID'), ('2022-02', 'Ukraine')]:
    axes[0].axvline(pd.Timestamp(d), color='orange', ls=':', alpha=0.5)
    axes[0].text(pd.Timestamp(d), 0.7, l, fontsize=8, rotation=90, va='top')

# Monthly class counts
monthly = train.set_index('date').resample('M')[TARGET].value_counts().unstack(fill_value=0)
axes[1].bar(monthly.index, monthly.get(0, 0), width=20, color='#C44E52', label='DOWN', alpha=0.7)
axes[1].bar(monthly.index, monthly.get(1, 0), width=20, bottom=monthly.get(0, 0), color='#55A868', label='UP', alpha=0.7)
axes[1].legend(fontsize=8); axes[1].set_ylabel('Count')
axes[1].set_title('Monthly Class Counts (Train)')
plt.tight_layout(); save('08_class_over_time.png')

# ═══════════════════════════════════════════════════════════════
# 9. VIOLIN PLOT — TOP 12 FEATURES
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 9. VIOLIN PLOT — TOP 12\n{P}')

top12 = combined.head(12)['feature'].tolist()
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
af = axes.flatten()
for i, c in enumerate(top12):
    ax = af[i]
    parts = ax.violinplot([train.loc[train[TARGET]==0, c].dropna(),
                           train.loc[train[TARGET]==1, c].dropna()],
                          positions=[0, 1], showmeans=True, showmedians=True)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#C44E52' if j == 0 else '#55A868')
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['DOWN', 'UP'])
    ax.set_title(c, fontsize=9)
fig.suptitle('Violin Plot by Class — Top 12 Features (Train)', fontsize=14)
plt.tight_layout(); save('09_violin_top12.png')

# ═══════════════════════════════════════════════════════════════
# 10. CORRELATION HEATMAP — BY CLASS
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 10. CORRELATION HEATMAP BY CLASS\n{P}')

top15 = combined.head(15)['feature'].tolist()
corr_up = train.loc[train[TARGET]==1, top15].corr(method='spearman')
corr_down = train.loc[train[TARGET]==0, top15].corr(method='spearman')
corr_diff = (corr_up - corr_down).abs()

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
for ax, mat, title in [(axes[0], corr_up, 'UP class'), (axes[1], corr_down, 'DOWN class'),
                        (axes[2], corr_diff, '|UP - DOWN| (difference)')]:
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    cmap = 'Reds' if title.startswith('|') else 'RdBu_r'
    center = None if title.startswith('|') else 0
    sns.heatmap(mat, mask=mask, annot=True, fmt='.2f', cmap=cmap, center=center,
                square=True, linewidths=0.5, annot_kws={'fontsize': 6}, ax=ax)
    ax.set_title(title, fontsize=11); ax.tick_params(labelsize=6)
fig.suptitle('Spearman Correlation — Top 15 Features by Class', fontsize=14)
plt.tight_layout(); save('10_corr_by_class.png')

# ═══════════════════════════════════════════════════════════════
# 11. CLASS SEPARABILITY — SCATTER TOP PAIRS
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 11. SCATTER — TOP FEATURE PAIRS\n{P}')

top6 = combined.head(6)['feature'].tolist()
pairs = [(top6[i], top6[j]) for i in range(len(top6)) for j in range(i+1, len(top6))][:9]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
af = axes.flatten()
for i, (f1, f2) in enumerate(pairs):
    ax = af[i]
    down_mask = train[TARGET] == 0
    up_mask = train[TARGET] == 1
    ax.scatter(train.loc[down_mask, f1], train.loc[down_mask, f2], s=3, alpha=0.15, color='#C44E52', label='DOWN')
    ax.scatter(train.loc[up_mask, f1], train.loc[up_mask, f2], s=3, alpha=0.15, color='#55A868', label='UP')
    ax.set_xlabel(f1, fontsize=7); ax.set_ylabel(f2, fontsize=7)
    ax.set_title(f'{f1} vs {f2}', fontsize=8)
    if i == 0: ax.legend(fontsize=6, markerscale=3)
fig.suptitle('Feature Pair Scatter — UP vs DOWN (Train)', fontsize=14)
plt.tight_layout(); save('11_scatter_pairs.png')

# ═══════════════════════════════════════════════════════════════
# 12. TRAIN vs TEST SHIFT — PER CLASS
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n 12. CLASS SHIFT: TRAIN vs TEST\n{P}')

print(f'\n  Train: UP={train[TARGET].mean():.3f} | Test: UP={test[TARGET].mean():.3f}')
ks_target, p_target = stats.ks_2samp(train[TARGET], test[TARGET])
print(f'  Target shift KS: stat={ks_target:.4f} p={p_target:.4f}')

print(f'\n  Feature shift per class:')
print(f'  {"Feature":<35} {"UP_KS":>8} {"UP_p":>10} {"DOWN_KS":>8} {"DOWN_p":>10}')
print(f'  {"-"*74}')
for c in combined.head(15)['feature']:
    ks_up, p_up = stats.ks_2samp(train.loc[train[TARGET]==1, c], test.loc[test[TARGET]==1, c])
    ks_dn, p_dn = stats.ks_2samp(train.loc[train[TARGET]==0, c], test.loc[test[TARGET]==0, c])
    print(f'  {c:<35} {ks_up:>8.4f} {p_up:>10.2e} {ks_dn:>8.4f} {p_dn:>10.2e}')

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f'\n{P}\n SUMMARY\n{P}')

sig_ks = ks_df[ks_df.sig].shape[0]
sig_rpb = pb_df[pb_df.p < 0.05].shape[0]
top5 = combined.head(5)

print(f"""
  RUN LABEL: {RUN_LABEL}
  TARGET: direction = ({TARGET_RET} > 0)
  CLASS BALANCE:
    Train: UP={train[TARGET].mean():.1%} DOWN={1-train[TARGET].mean():.1%} (near balanced)
    Test:  UP={test[TARGET].mean():.1%} DOWN={1-test[TARGET].mean():.1%}

  FEATURE SEPARABILITY:
    KS test (UP vs DOWN differ): {sig_ks}/{len(NUM)} features significant (p<0.05)
    Point-biserial significant:  {sig_rpb}/{len(NUM)} features
    Cohen's d > 0.2:             {(edf.abs_d > 0.2).sum()}/{len(NUM)} features

  TOP 5 FEATURES (combined rpb + MI + KS):
    1. {top5.iloc[0].feature:<30} score={top5.iloc[0].score:.4f}
    2. {top5.iloc[1].feature:<30} score={top5.iloc[1].score:.4f}
    3. {top5.iloc[2].feature:<30} score={top5.iloc[2].score:.4f}
    4. {top5.iloc[3].feature:<30} score={top5.iloc[3].score:.4f}
    5. {top5.iloc[4].feature:<30} score={top5.iloc[4].score:.4f}

  PLOTS: 12 files in eda_classification/
""")

ranking_name = f'{PREFIX}_feature_ranking_clf.csv' if PREFIX else 'feature_ranking_clf.csv'
combined.to_csv(os.path.join(OUT, ranking_name), index=False)
print(f'{P}\n DONE\n{P}')
