"""Create the EDA notebook for oil price classification."""
import nbformat as nbf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(md("""# EDA — Oil Price Direction Classification

**Bài toán**: Binary classification dự đoán xu hướng **tăng (1) / giảm (0)** của giá dầu ngày tiếp theo.

**Dataset**: `data/processed/dataset_final.csv` — 2923 rows × 33 features + date

**Quy tắc quan trọng**: Mọi phân tích EDA chỉ thực hiện trên tập **TRAIN** (< 2023-01-01).

---
### Nội dung
1. [Setup & Load Data](#1-setup--load-data)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Quality Checks](#3-data-quality-checks)
4. [Target Analysis](#4-target-analysis)
5. [Feature Distributions](#5-feature-distributions)
6. [Time-series Analysis](#6-time-series-analysis)
7. [Feature–Target Relationship](#7-featuretarget-relationship)
8. [Leakage & Split Checks](#8-leakage--split-checks)
9. [Key Findings & Recommendations](#9-key-findings--recommendations)
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
cells.append(code("""\
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ks_2samp
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE = Path(".").resolve()
if not (BASE / "data").exists():
    BASE = BASE.parent
FIG_DIR = BASE / "figures"; FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = BASE / "tables";  TBL_DIR.mkdir(exist_ok=True)
REP_DIR = BASE / "reports"; REP_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF = pd.Timestamp("2023-01-01")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
EVENTS = {"COVID crash":"2020-03-09","Ukraine war":"2022-02-24","Fed hike":"2022-03-16"}
print("Setup complete. Working dir:", BASE)
"""))

# ── Cell 2: Load ──────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup & Load Data"))
cells.append(code("""\
df = pd.read_csv(BASE / "data/processed/dataset_final.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Target definition: direction = 1 if return > 0 (Up), else 0 (Down)
df["direction"] = (df["oil_return"] > 0).astype(int)

train = df[df["date"] < TRAIN_CUTOFF].copy().reset_index(drop=True)
test  = df[df["date"] >= TRAIN_CUTOFF].copy().reset_index(drop=True)

print(f"Full dataset : {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Train set    : {len(train)} rows  ({train.date.min().date()} → {train.date.max().date()})")
print(f"Test  set    : {len(test)} rows  ({test.date.min().date()} → {test.date.max().date()})")
print(f"Missing (all): {df.isnull().sum().sum()}")
"""))

# ── Cell 3: Overview ──────────────────────────────────────────────────────────
cells.append(md("## 2. Dataset Overview"))
cells.append(code("""\
# Feature groups
FEATURE_GROUPS = {
    "Market":    ["oil_return","usd_return","sp500_return","vix_return","oil_volatility_7d"],
    "Macro":     ["yield_spread","cpi_lag","unemployment_lag","fed_rate_change","fed_rate_regime","real_rate"],
    "Supply":    ["inventory_zscore","inventory_change_pct","production_change_pct","net_imports_change_pct"],
    "Sentiment": ["gdelt_tone_7d","gdelt_tone_30d","gdelt_goldstein","gdelt_goldstein_7d",
                  "gdelt_events","conflict_event_count","fatalities","conflict_intensity_7d",
                  "fatalities_7d","geopolitical_stress_index"],
    "Lag":       ["oil_return_lag1","oil_return_lag2","vix_lag1","gdelt_tone_lag1","gdelt_volume_lag1"],
    "Temporal":  ["day_of_week","month"],
}
for grp, cols in FEATURE_GROUPS.items():
    print(f"  {grp:10s}: {cols}")
"""))

cells.append(code("""\
# Summary statistics (train set)
print("\\n--- Describe (train set) ---")
train.describe().round(3)
"""))

cells.append(code("""\
# Timeline plot
img = mpimg.imread(FIG_DIR / "timeline_coverage.png")
fig, ax = plt.subplots(figsize=(14, img.shape[0]/img.shape[1]*14))
ax.imshow(img); ax.axis("off")
plt.tight_layout(); plt.show()
"""))

# ── Cell 4: Quality ───────────────────────────────────────────────────────────
cells.append(md("## 3. Data Quality Checks"))
cells.append(code("""\
print("Missing values:", train.isnull().sum().sum())
print("Duplicate rows:", train.duplicated().sum())
print("Duplicate dates:", train['date'].duplicated().sum())
num_cols = train.select_dtypes(include=[np.number]).columns
inf_found = {c: np.isinf(train[c]).sum() for c in num_cols if np.isinf(train[c]).sum()>0}
print("INF values:", inf_found or "None")
print("\\n✓ All clean!" if not inf_found and train.isnull().sum().sum()==0 else "⚠️ Issues found!")
"""))

cells.append(code("""\
# Outlier summary
out_df = pd.read_csv(TBL_DIR / "outlier_summary.csv")
print("Top IQR outlier features (fat tails expected in financial data):")
out_df
"""))

cells.append(code("""\
# Lag consistency check
lag1_ok = np.allclose(df["oil_return_lag1"].values[1:], df["oil_return"].values[:-1],
                      equal_nan=True, rtol=1e-3)
print(f"oil_return_lag1 == oil_return.shift(1): {'✓ OK' if lag1_ok else '✗ FAIL'}")
"""))

# ── Cell 5: Target ────────────────────────────────────────────────────────────
cells.append(md("## 4. Target Analysis"))
cells.append(code("""\
ret = train["oil_return"]; dir_ = train["direction"]
cc = dir_.value_counts().sort_index(); cr = dir_.value_counts(normalize=True).sort_index()
print(f"Skewness : {ret.skew():.3f}")
print(f"Kurtosis : {ret.kurtosis():.3f}  (excess kurtosis — >3 = fat tails)")
print(f"Class    : Down={cc[0]} ({cr[0]:.2%})  Up={cc[1]} ({cr[1]:.2%})")
print(f"Imbalanced: {abs(cr[0]-cr[1])>0.10}")
"""))

cells.append(code("""\
for fname in ["target_distribution.png","returns_by_class.png","target_temporal_pattern.png","streak_distribution.png"]:
    fp = FIG_DIR/fname
    if not fp.exists(): continue
    img = mpimg.imread(fp)
    fig, ax = plt.subplots(figsize=(13, img.shape[0]/img.shape[1]*13))
    ax.imshow(img); ax.axis("off"); ax.set_title(fname.replace(".png","").replace("_"," ").title())
    plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# Regime analysis
pd.read_csv(TBL_DIR / "regime_analysis.csv")
"""))

# ── Cell 6: Distributions ────────────────────────────────────────────────────
cells.append(md("## 5. Feature Distributions\n\nHistogram + KDE per feature group (train set only)."))
cells.append(code("""\
for grp in ["market","macro","supply","sentiment","lag"]:
    fp = FIG_DIR / f"hist_kde_{grp}.png"
    if not fp.exists(): continue
    img = mpimg.imread(fp)
    fig, ax = plt.subplots(figsize=(14, img.shape[0]/img.shape[1]*14))
    ax.imshow(img); ax.axis("off"); ax.set_title(f"Feature Distributions — {grp.title()} Group")
    plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# Skewness/kurtosis summary
feat_stats = pd.read_csv(TBL_DIR / "feature_stats_summary.csv")
print("Highly skewed features (|skew|>1):")
print(feat_stats.loc[feat_stats["skewness"].abs()>1, ["feature","skewness","kurt_val"]].to_string(index=False))
"""))

# ── Cell 7: Time-series ──────────────────────────────────────────────────────
cells.append(md("## 6. Time-series Analysis"))
cells.append(code("""\
# ADF stationarity results
adf_tbl = pd.read_csv(TBL_DIR / "adf_test_results.csv")
print("ADF Test (p<0.05 → stationary):")
print(adf_tbl.to_string(index=False))
"""))

cells.append(code("""\
# ACF/PACF oil_return
ret_ts = train.set_index("date")["oil_return"].dropna()
fig, axes = plt.subplots(1,2,figsize=(13,4))
plot_acf(ret_ts, lags=30, ax=axes[0], title="ACF — oil_return (train)")
plot_pacf(ret_ts, lags=30, ax=axes[1], title="PACF — oil_return (train)", method="ywm")
plt.tight_layout(); plt.savefig(FIG_DIR/"acf_pacf_oil_return.png",dpi=120); plt.show()
print("\\nInterpretation: Values near 0 → market close to random walk → low autocorrelation in returns")
"""))

cells.append(code("""\
# Rolling stats + seasonality
for fname in ["rolling_statistics.png","rolling_correlation.png","seasonality_heatmap.png"]:
    fp = FIG_DIR/fname
    if not fp.exists(): continue
    img = mpimg.imread(fp)
    fig, ax = plt.subplots(figsize=(13, img.shape[0]/img.shape[1]*13))
    ax.imshow(img); ax.axis("off"); ax.set_title(fname.replace(".png","").replace("_"," ").title())
    plt.tight_layout(); plt.show()
"""))

# ── Cell 8: Feature–Target ────────────────────────────────────────────────────
cells.append(md("## 7. Feature–Target Relationship"))
cells.append(code("""\
# Correlation heatmap
fp = FIG_DIR/"correlation_heatmap.png"
img = mpimg.imread(fp)
fig, ax = plt.subplots(figsize=(14, 12))
ax.imshow(img); ax.axis("off"); ax.set_title("Spearman Correlation Matrix (feature×feature)")
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# VIF
vif_tbl = pd.read_csv(TBL_DIR/"vif_results.csv").sort_values("VIF",ascending=False)
print("VIF (top 15 — VIF>10 = severe multicollinearity):")
print(vif_tbl.head(15).to_string(index=False))
"""))

cells.append(code("""\
# Point-biserial + Mann-Whitney
ft_tbl = pd.read_csv(TBL_DIR/"feature_target_correlation.csv").sort_values("pointbiserial_r",key=abs,ascending=False)
print(f"Features significant by MW test (p<0.05): {ft_tbl['significant_MW'].sum()}/{len(ft_tbl)}")
print("\\nTop 12 by |point-biserial r|:")
print(ft_tbl.head(12).to_string(index=False))
"""))

cells.append(code("""\
# Mutual Information
mi_tbl = pd.read_csv(TBL_DIR/"mutual_information.csv").sort_values("mutual_info",ascending=False)
fig, ax = plt.subplots(figsize=(12,5))
top = mi_tbl.head(20)
ax.barh(top["feature"][::-1], top["mutual_info"][::-1], color="steelblue")
ax.set_title("Mutual Information — Top 20 features vs direction"); ax.set_xlabel("MI score")
plt.tight_layout(); plt.savefig(FIG_DIR/"mutual_information.png",dpi=120); plt.show()
"""))

# ── Cell 9: Leakage ───────────────────────────────────────────────────────────
cells.append(md("## 8. Leakage & Split Checks"))
cells.append(code("""\
# Leakage risk assessment
leak_tbl = pd.read_csv(TBL_DIR/"leakage_risk_assessment.csv")
print(leak_tbl.to_string(index=False))
"""))

cells.append(code("""\
# Train/test integrity
train_up = (train["oil_return"]>0).mean()
test_up  = (test["oil_return"]>0).mean()
print(f"Class ratio Up — train: {train_up:.2%}  test: {test_up:.2%}  diff: {abs(train_up-test_up):.2%}")
print(f"Date overlap       : {len(set(train.date)&set(test.date))} dates (should be 0)")
print(f"Train ends before  : {train.date.max() < test.date.min()}")
"""))

cells.append(code("""\
# Distribution shift (KS test)
ks_tbl = pd.read_csv(TBL_DIR/"distribution_shift_ks_test.csv").sort_values("ks_stat",ascending=False)
n_shift = ks_tbl["shift_detected"].sum()
print(f"{n_shift}/{len(ks_tbl)} features show significant distribution shift train→test (KS p<0.05)")
print("\\nTop shifted features:")
print(ks_tbl.head(10).to_string(index=False))
"""))

cells.append(code("""\
for fname in ["train_test_distribution_shift.png","class_ratio_train_vs_test.png"]:
    fp = FIG_DIR/fname
    if not fp.exists(): continue
    img = mpimg.imread(fp)
    fig, ax = plt.subplots(figsize=(13, img.shape[0]/img.shape[1]*13))
    ax.imshow(img); ax.axis("off"); ax.set_title(fname.replace(".png","").replace("_"," ").title())
    plt.tight_layout(); plt.show()
"""))

# ── Cell 10: Summary ──────────────────────────────────────────────────────────
cells.append(md("## 9. Key Findings & Recommendations"))
cells.append(code("""\
rank_tbl = pd.read_csv(TBL_DIR/"feature_ranking.csv")
print("Feature Ranking (top 20):")
print(rank_tbl[["feature","abs_r","mutual_info","VIF","recommendation"]].head(20).to_string(index=False))
"""))

cells.append(md("""\
### Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | **Class balance** | Up=51.4% / Down=48.6% — acceptable |
| 2 | **Returns dist** | kurtosis≈12 (fat tails), skew≈−0.39 — financial norm |
| 3 | **Random walk** | ACF oil_return ≈ 0 → prediction is genuinely hard |
| 4 | **Top predictors** | vix_return (r=−0.21), sp500_return (0.21), usd_return (−0.11) |
| 5 | **Multicollinearity** | GDELT/conflict features VIF>10 — prune redundants |
| 6 | **Distribution shift** | 23/31 features drifted train→test (macro regime change) |
| 7 | **Leakage** | oil_return/direction excluded; usd/sp500/vix_return are same-day (medium risk) |
| 8 | **Seasonal** | Rolling Up-ratio fluctuates by regime; avg streak ≈ 2 days |

### Modeling Recommendations

```
Metric       : AUC-ROC + F1-weighted (NOT accuracy alone)
CV           : TimeSeriesSplit(n_splits=5)  — NO random k-fold
Scaling      : RobustScaler
Baseline     : "always predict Up" → 51.4%
Models order : Logistic Regression → Random Forest → XGBoost
Class imbal  : class_weight='balanced' (minimal imbalance)
Strict setup : replace vix_return/sp500_return/usd_return with _lag1 versions
```
"""))

nb.cells = cells
out = BASE / "notebooks/eda_oil_price_classification.ipynb"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written: {out}")
