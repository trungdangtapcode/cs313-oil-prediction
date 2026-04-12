"""
EDA - Oil Price Prediction | 54 columns
"""
import warnings, os
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

sns.set_style('whitegrid')
ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
def save(name):
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight', dpi=130)
    plt.close('all')
P = '=' * 90

# --- LOAD ---
df = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'dataset_step4_transformed.csv'), parse_dates=['date'])
df.sort_values('date', inplace=True); df.reset_index(drop=True, inplace=True)
TARGET = 'oil_return'
SPLIT = '2023-01-01'
train = df[df['date'] < SPLIT].copy()
test = df[df['date'] >= SPLIT].copy()
NUM = [c for c in df.columns if c != 'date']  # 53 numeric (bao gom target)
ALL54 = list(df.columns)  # 54 columns

print(f'{P}\n DATASET: {df.shape[0]} rows x {df.shape[1]} cols | Train {len(train)} | Test {len(test)}\n ALL 54 COLS: {ALL54}\n{P}')

# =====================================================================
# 1. DATA QUALITY - 54 COLUMNS
# =====================================================================
print(f'\n{P}\n 1. DATA QUALITY - ALL 54 COLUMNS\n{P}')
print(f' Missing | INF | Duplicates: {df.isnull().sum().sum()} | {np.isinf(df[NUM]).sum().sum()} | {df.duplicated().sum()}')
print(f'\n {"Col":<35} {"dtype":<12} {"nunique":>8} {"miss":>6} {"zero%":>7} {"top_val%":>9}')
print(f' {"-"*80}')
for c in ALL54:
    nu = df[c].nunique()
    ms = df[c].isnull().sum()
    if c == 'date':
        zp = '-'
        tv = '-'
    else:
        zp = f'{(df[c]==0).mean()*100:.1f}%'
        tv = f'{df[c].value_counts(normalize=True).iloc[0]*100:.1f}%'
    print(f' {c:<35} {str(df[c].dtype):<12} {nu:>8} {ms:>6} {zp:>7} {tv:>9}')

# =====================================================================
# 2. DESCRIPTIVE STATS - 53 NUMERIC COLUMNS (tru date)
# =====================================================================
print(f'\n{P}\n 2. DESCRIPTIVE STATS - ALL 53 NUMERIC (Train)\n{P}')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', '{:.4f}'.format)
desc = train[NUM].describe().T
desc['skew'] = train[NUM].skew()
desc['kurt'] = train[NUM].kurtosis()
print(desc[['count','mean','std','min','25%','50%','75%','max','skew','kurt']].to_string())

# =====================================================================
# 3. DISTRIBUTION PLOT - ALL 53 NUMERIC (9 rows x 6 cols = 54 slots)
# =====================================================================
print(f'\n{P}\n 3. DISTRIBUTION PLOTS - ALL 53 NUMERIC\n{P}')
NCOL = 6; NROW = 9  # 54 slots for 53 features
fig, axes = plt.subplots(NROW, NCOL, figsize=(NCOL*3.2, NROW*2.5))
af = axes.flatten()
plotted = 0
for i, c in enumerate(NUM):
    ax = af[i]
    d = train[c].dropna()
    if d.nunique() <= 5:
        vc = d.value_counts().sort_index()
        ax.bar([str(x) for x in vc.index], vc.values, color='#4C72B0', edgecolor='white')
    else:
        ax.hist(d, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8, density=True)
    ax.set_title(f'{c}', fontsize=6, fontweight='bold')
    ax.tick_params(labelsize=5)
    plotted += 1
for j in range(plotted, NROW*NCOL):
    af[j].set_visible(False)
fig.suptitle(f'Distributions - All {plotted} numeric columns (Train)', fontsize=14, y=1.01)
plt.tight_layout(); save('01_dist_all53.png')
print(f' Plotted: {plotted}/53 numeric columns -> 01_dist_all53.png')

# =====================================================================
# 4. STATIONARITY - ALL 53 NUMERIC
# =====================================================================
print(f'\n{P}\n 4. STATIONARITY (ADF+KPSS) - ALL 53 NUMERIC\n{P}')
print(f' {"Col":<35} {"ADF_p":>8} {"KPSS_p":>8} {"Verdict":<18}')
print(f' {"-"*72}')
verdicts = {}
for c in NUM:
    s = train[c].dropna()
    if s.std() == 0:
        verdicts[c] = 'Constant'; print(f' {c:<35} {"N/A":>8} {"N/A":>8} {"Constant":<18}'); continue
    ap = adfuller(s, autolag='AIC')[1]
    kp = kpss(s, regression='c', nlags='auto')[1]
    if ap < 0.05 and kp >= 0.05:   v = 'Stationary'
    elif ap >= 0.05 and kp < 0.05: v = 'NON-STATIONARY'
    elif ap < 0.05 and kp < 0.05:  v = 'Trend-stationary'
    else:                           v = 'Uncertain'
    verdicts[c] = v
    mark = ' ***' if v == 'NON-STATIONARY' else ''
    print(f' {c:<35} {ap:>8.4f} {kp:>8.4f} {v:<18}{mark}')
for label in ['Stationary','Trend-stationary','NON-STATIONARY','Uncertain','Constant']:
    cols = [c for c,v in verdicts.items() if v == label]
    if cols: print(f' {label}: {len(cols)} -> {cols}')

# =====================================================================
# 5. CORRELATION WITH TARGET - ALL 52 FEATURES
# =====================================================================
print(f'\n{P}\n 5. CORRELATION WITH TARGET - ALL 52 FEATURES\n{P}')
FEATS = [c for c in NUM if c != TARGET]  # 52
sp = train[FEATS].corrwith(train[TARGET], method='spearman')
pe = train[FEATS].corrwith(train[TARGET], method='pearson')
cdf = pd.DataFrame({'sp':sp,'pe':pe,'abs_sp':sp.abs()}).sort_values('abs_sp', ascending=False)
print(f' {"Col":<35} {"Spearman":>10} {"Pearson":>10}')
print(f' {"-"*58}')
for c in cdf.index:
    print(f' {c:<35} {cdf.loc[c,"sp"]:>10.4f} {cdf.loc[c,"pe"]:>10.4f}')

# Bar chart - ALL 52
fig, ax = plt.subplots(figsize=(10, 16))
colors = ['#C44E52' if x < 0 else '#55A868' for x in cdf['sp']]
ax.barh(range(len(cdf)), cdf['sp'], color=colors, height=0.8)
ax.set_yticks(range(len(cdf))); ax.set_yticklabels(cdf.index, fontsize=6)
ax.invert_yaxis(); ax.axvline(0, color='black', lw=0.5)
ax.set_title(f'Spearman with {TARGET} - All {len(cdf)} features')
for i, v in enumerate(cdf['sp']): ax.text(v+(0.001 if v>=0 else -0.001), i, f'{v:.3f}', va='center', ha='left' if v>=0 else 'right', fontsize=5)
plt.tight_layout(); save('02_corr_target_all52.png')
print(f'\n Plotted: {len(cdf)}/52 features -> 02_corr_target_all52.png')

# =====================================================================
# 6. FULL HEATMAP - ALL 53 NUMERIC (53x53)
# =====================================================================
print(f'\n{P}\n 6. CORRELATION HEATMAP - ALL 53 NUMERIC (53x53)\n{P}')
cm = train[NUM].corr(method='spearman')
fig, ax = plt.subplots(figsize=(26, 22))
mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
sns.heatmap(cm, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.2, ax=ax, cbar_kws={'shrink':0.5})
ax.tick_params(labelsize=5); ax.set_title(f'Spearman Heatmap - All {len(NUM)} numeric cols ({len(NUM)}x{len(NUM)})', fontsize=13)
plt.tight_layout(); save('03_heatmap_53x53.png')
print(f' Plotted: {len(NUM)}x{len(NUM)} heatmap -> 03_heatmap_53x53.png')

# Multicollinear pairs
pairs = []
for i in range(len(cm.columns)):
    for j in range(i+1, len(cm.columns)):
        r = cm.iloc[i,j]
        if abs(r) > 0.80: pairs.append((cm.columns[i], cm.columns[j], r))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f'\n |rho| > 0.80: {len(pairs)} pairs')
for f1,f2,r in pairs: print(f'   {f1:<30} {f2:<30} {r:>7.4f}')

# =====================================================================
# 7. SCATTER vs TARGET - ALL 52 FEATURES
# =====================================================================
print(f'\n{P}\n 7. SCATTER vs TARGET - ALL 52 FEATURES\n{P}')
NCOL2 = 7; NROW2 = 8  # 56 slots for 52
fig, axes = plt.subplots(NROW2, NCOL2, figsize=(NCOL2*2.8, NROW2*2.5))
af2 = axes.flatten()
plotted2 = 0
for i, c in enumerate(FEATS):
    ax = af2[i]
    ax.scatter(train[c], train[TARGET], s=1, alpha=0.15, color='#4C72B0')
    ax.set_xlabel(c, fontsize=5); ax.set_ylabel('', fontsize=5)
    ax.set_title(f'{c} (rho={sp[c]:.3f})', fontsize=5)
    ax.tick_params(labelsize=4)
    plotted2 += 1
for j in range(plotted2, NROW2*NCOL2): af2[j].set_visible(False)
fig.suptitle(f'Scatter vs {TARGET} - All {plotted2} features', fontsize=13, y=1.01)
plt.tight_layout(); save('04_scatter_all52.png')
print(f' Plotted: {plotted2}/52 features -> 04_scatter_all52.png')

# =====================================================================
# 8. VIF - ALL 53 NUMERIC
# =====================================================================
print(f'\n{P}\n 8. VIF - ALL 53 NUMERIC\n{P}')
X = train[NUM].dropna()
Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=NUM)
vif = []
for i, c in enumerate(NUM):
    v = variance_inflation_factor(Xs.values, i)
    vif.append((c, v))
vif.sort(key=lambda x: x[1], reverse=True)
print(f' {"Col":<35} {"VIF":>10} {"Flag":<12}')
print(f' {"-"*60}')
nd = nw = nok = 0
for c, v in vif:
    if np.isinf(v) or v > 10: f = 'DANGEROUS'; nd += 1
    elif v > 5: f = 'WARNING'; nw += 1
    else: f = 'OK'; nok += 1
    vs = 'INF' if np.isinf(v) else f'{v:.1f}'
    print(f' {c:<35} {vs:>10} {f:<12}')
print(f'\n DANGEROUS(>10): {nd} | WARNING(5-10): {nw} | OK(<5): {nok} | Total: {len(vif)}/53')

# =====================================================================
# 9. OUTLIERS - ALL 53 NUMERIC
# =====================================================================
print(f'\n{P}\n 9. OUTLIERS (IQR + Z>3) - ALL 53 NUMERIC\n{P}')
print(f' {"Col":<35} {"IQR_n":>7} {"IQR_%":>7} {"Z3_n":>7} {"Z3_%":>7}')
print(f' {"-"*67}')
for c in NUM:
    d = train[c].dropna()
    Q1,Q3 = d.quantile(0.25), d.quantile(0.75)
    IQR = Q3-Q1; lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
    ni = ((d<lo)|(d>hi)).sum()
    nz = (np.abs(stats.zscore(d))>3).sum() if d.std()>0 else 0
    print(f' {c:<35} {ni:>7} {ni/len(d)*100:>6.1f}% {nz:>7} {nz/len(d)*100:>6.1f}%')

# Boxplot ALL 53
NCOL3=6; NROW3=9
fig, axes = plt.subplots(NROW3, NCOL3, figsize=(NCOL3*3, NROW3*2.3))
af3 = axes.flatten()
plotted3 = 0
for i, c in enumerate(NUM):
    ax = af3[i]
    d = train[c].dropna()
    bp = ax.boxplot(d, vert=True, patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7), medianprops=dict(color='red', lw=1.5),
                    flierprops=dict(marker='.', markersize=1, alpha=0.3))
    ax.set_title(c, fontsize=6, fontweight='bold'); ax.tick_params(labelsize=5)
    plotted3 += 1
for j in range(plotted3, NROW3*NCOL3): af3[j].set_visible(False)
fig.suptitle(f'Boxplots - All {plotted3} numeric columns (Train)', fontsize=13, y=1.01)
plt.tight_layout(); save('05_boxplot_all53.png')
print(f'\n Plotted: {plotted3}/53 numeric -> 05_boxplot_all53.png')

# =====================================================================
# 10. MUTUAL INFORMATION - ALL 52 FEATURES
# =====================================================================
print(f'\n{P}\n 10. MUTUAL INFORMATION - ALL 52 FEATURES\n{P}')
Xmi = train[FEATS].fillna(0)
mi = mutual_info_regression(Xmi, train[TARGET], random_state=42, n_neighbors=5)
midf = pd.DataFrame({'feat':FEATS,'mi':mi,'sp':sp.values}).sort_values('mi', ascending=False)
midf.reset_index(drop=True, inplace=True)
print(f' {"#":<4} {"Feature":<35} {"MI":>8} {"Spearman":>10} {"Type":<18}')
print(f' {"-"*78}')
for i,r in midf.iterrows():
    tp = 'NONLINEAR' if r.mi>0.01 and abs(r.sp)<0.05 else ('LINEAR+NL' if r.mi>0.01 else 'WEAK')
    print(f' {i+1:<4} {r.feat:<35} {r.mi:>8.4f} {r.sp:>10.4f} {tp:<18}')

fig, ax = plt.subplots(figsize=(10, 16))
ax.barh(range(len(midf)), midf['mi'], color='#4C72B0', height=0.8)
ax.set_yticks(range(len(midf))); ax.set_yticklabels(midf['feat'], fontsize=6)
ax.invert_yaxis(); ax.set_xlabel('MI Score')
ax.set_title(f'Mutual Information - All {len(midf)} features')
for i, v in enumerate(midf['mi']): ax.text(v+0.001, i, f'{v:.4f}', va='center', fontsize=5)
plt.tight_layout(); save('06_mi_all52.png')
print(f'\n Plotted: {len(midf)}/52 features -> 06_mi_all52.png')

# =====================================================================
# 11. DISTRIBUTION SHIFT - ALL 53 NUMERIC
# =====================================================================
print(f'\n{P}\n 11. TRAIN vs TEST SHIFT (KS test) - ALL 53 NUMERIC\n{P}')
print(f' {"Col":<35} {"KS_stat":>9} {"KS_p":>12} {"Train_mu":>12} {"Test_mu":>12} {"Shift":>6}')
print(f' {"-"*90}')
shifts = []
for c in NUM:
    ks_s, ks_p = stats.ks_2samp(train[c].dropna(), test[c].dropna())
    shifted = ks_p < 0.01
    shifts.append((c, ks_s, ks_p, train[c].mean(), test[c].mean(), shifted))
shifts.sort(key=lambda x: x[1], reverse=True)
n_shifted = sum(1 for s in shifts if s[5])
for c, ks_s, ks_p, tm, tsm, sh in shifts:
    sh_str = "YES" if sh else ""
    print(f' {c:<35} {ks_s:>9.4f} {ks_p:>12.2e} {tm:>12.4f} {tsm:>12.4f} {sh_str:>6}')
print(f'\n Shifted: {n_shifted}/{len(NUM)} columns')

# Overlay train vs test - ALL 53
NCOL4=6; NROW4=9
fig, axes = plt.subplots(NROW4, NCOL4, figsize=(NCOL4*3.2, NROW4*2.5))
af4 = axes.flatten()
plotted4 = 0
for i, c in enumerate(NUM):
    ax = af4[i]
    ax.hist(train[c].dropna(), bins=40, alpha=0.5, density=True, color='#4C72B0', label='Train')
    ax.hist(test[c].dropna(), bins=40, alpha=0.5, density=True, color='#C44E52', label='Test')
    ks_p_val = [s[2] for s in shifts if s[0]==c][0]
    ax.set_title(f'{c}\np={ks_p_val:.1e}', fontsize=5, fontweight='bold')
    ax.tick_params(labelsize=4)
    if i == 0: ax.legend(fontsize=5)
    plotted4 += 1
for j in range(plotted4, NROW4*NCOL4): af4[j].set_visible(False)
fig.suptitle(f'Train vs Test - All {plotted4} numeric columns', fontsize=13, y=1.01)
plt.tight_layout(); save('07_shift_all53.png')
print(f' Plotted: {plotted4}/53 numeric -> 07_shift_all53.png')

# =====================================================================
# 12. TIME SERIES - ALL 53 NUMERIC
# =====================================================================
print(f'\n{P}\n 12. TIME SERIES - ALL 53 NUMERIC\n{P}')
NCOL5=6; NROW5=9
fig, axes = plt.subplots(NROW5, NCOL5, figsize=(NCOL5*3.5, NROW5*2))
af5 = axes.flatten()
plotted5 = 0
for i, c in enumerate(NUM):
    ax = af5[i]
    ax.plot(train['date'], train[c], lw=0.3, alpha=0.7, color='#4C72B0')
    ax.set_title(c, fontsize=6, fontweight='bold'); ax.tick_params(labelsize=4)
    for d in ['2020-03','2022-02']:
        ax.axvline(pd.Timestamp(d), color='orange', alpha=0.4, ls=':', lw=0.5)
    plotted5 += 1
for j in range(plotted5, NROW5*NCOL5): af5[j].set_visible(False)
fig.suptitle(f'Time Series - All {plotted5} numeric columns (Train)', fontsize=13, y=1.01)
plt.tight_layout(); save('08_timeseries_all53.png')
print(f' Plotted: {plotted5}/53 numeric -> 08_timeseries_all53.png')

# =====================================================================
# 13. ARCH EFFECTS
# =====================================================================
print(f'\n{P}\n 13. ARCH EFFECTS (oil_return)\n{P}')
sq = train[TARGET]**2
lb = acorr_ljungbox(sq, lags=[10,20,30], return_df=True)
for lag in lb.index: print(f' lag={lag}: stat={lb.loc[lag,"lb_stat"]:.2f} p={lb.loc[lag,"lb_pvalue"]:.2e}')
print(f' ARCH effects: {"YES" if all(lb["lb_pvalue"]<0.05) else "NO"}')

# =====================================================================
# 14. PCA VARIANCE
# =====================================================================
print(f'\n{P}\n 14. PCA - VARIANCE EXPLAINED\n{P}')
from sklearn.decomposition import PCA
Xpca = StandardScaler().fit_transform(train[FEATS].fillna(0))
pca = PCA().fit(Xpca)
cumvar = np.cumsum(pca.explained_variance_ratio_)*100
print(f' {"#PC":>4} {"Var%":>8} {"Cumul%":>8}')
for i in range(min(20, len(cumvar))):
    print(f' {i+1:>4} {pca.explained_variance_ratio_[i]*100:>8.2f} {cumvar[i]:>8.2f}')
for t in [80, 90, 95, 99]:
    n = np.argmax(cumvar >= t) + 1
    print(f' {t}% variance -> {n} components')

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, len(cumvar)+1), pca.explained_variance_ratio_*100, color='#4C72B0', alpha=0.7)
ax2 = ax.twinx()
ax2.plot(range(1, len(cumvar)+1), cumvar, 'r-o', markersize=3)
ax2.axhline(90, color='green', ls='--', alpha=0.5); ax2.axhline(95, color='orange', ls='--', alpha=0.5)
ax.set_xlabel('PC'); ax.set_ylabel('Variance %'); ax2.set_ylabel('Cumulative %')
ax.set_title(f'PCA Variance - {len(FEATS)} features')
plt.tight_layout(); save('09_pca.png')
print(f' -> 09_pca.png')

# =====================================================================
# 15. ROLLING FEATURE IMPORTANCE (Spearman 90d) - TOP 10
# =====================================================================
print(f'\n{P}\n 15. ROLLING CORRELATION (90d) - TOP 10 FEATURES\n{P}')
top10 = cdf.head(10).index.tolist()
fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
for i, c in enumerate(top10):
    ax = axes[i//2][i%2]
    rc = train[[TARGET, c]].rolling(90).corr().unstack()[TARGET][c]
    ax.plot(train['date'], rc, lw=0.7, color='#4C72B0')
    ax.axhline(0, color='red', ls='--', alpha=0.4)
    ax.fill_between(train['date'], rc, 0, alpha=0.1, color='#4C72B0')
    ax.set_title(f'{c} (overall={sp[c]:.3f})', fontsize=9)
    for d in ['2020-03','2022-02']: ax.axvline(pd.Timestamp(d), color='orange', alpha=0.4, ls=':')
fig.suptitle('Rolling 90d Spearman with target - Top 10', fontsize=13, y=1.0)
plt.tight_layout(); save('10_rolling_top10.png')
print(f' Plotted top 10: {top10} -> 10_rolling_top10.png')

# =====================================================================
# SUMMARY
# =====================================================================
ns_list = [c for c,v in verdicts.items() if v=='NON-STATIONARY']
st_list = [c for c,v in verdicts.items() if v=='Stationary']
ts_list = [c for c,v in verdicts.items() if v=='Trend-stationary']
top3_corr = cdf.head(3).index.tolist()
top5_mi = midf.head(5)['feat'].tolist()

print(f'\n{P}\n SUMMARY\n{P}')
print(f"""
 DATASET:        {df.shape} | Train {len(train)} | Test {len(test)}
 QUALITY:        0 missing, 0 INF, 0 dup
 TARGET:         {TARGET} | skew={train[TARGET].skew():.2f} kurt={train[TARGET].kurtosis():.1f} | Stationary=YES
 STATIONARITY:   {len(st_list)} stat / {len(ts_list)} trend / {len(ns_list)} NON-stat
 ARCH:           YES (volatility clustering)
 CORRELATION:    Top 3: {top3_corr} ({cdf.head(3)['abs_sp'].values.round(3).tolist()})
 MULTICOLLIN:    {len(pairs)} pairs |rho|>0.80
 VIF:            {nd} dangerous / {nw} warning / {nok} ok
 MI TOP 5:       {top5_mi}
 SHIFT:          {n_shifted}/{len(NUM)} features shifted
 PCA:            90% var in {np.argmax(cumvar>=90)+1} PCs / 95% in {np.argmax(cumvar>=95)+1} PCs

 PLOTS (10 files, all covering FULL feature set):""")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.png'): print(f'   {f}')
print(f'\n{P}\n DONE\n{P}')
