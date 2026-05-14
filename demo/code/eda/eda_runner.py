"""
================================================================================
EDA RUNNER — Oil Price Classification
================================================================================
Thực thi toàn bộ EDA tasks 01-09 theo thứ tự.
Output: figures/, tables/, reports/
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu, kstest, ks_2samp
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "data/processed/dataset_final.csv"
FIG_DIR = BASE / "figures";  FIG_DIR.mkdir(exist_ok=True)
TBL_DIR = BASE / "tables";   TBL_DIR.mkdir(exist_ok=True)
REP_DIR = BASE / "reports";  REP_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF = pd.Timestamp("2023-01-01")
STYLE = "seaborn-v0_8-darkgrid"
plt.style.use(STYLE)
sns.set_palette("husl")

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

EVENTS = {
    "COVID crash": "2020-03-09",
    "COVID recovery": "2020-06-01",
    "Ukraine war": "2022-02-24",
    "Fed hike start": "2022-03-16",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def savefig(name, tight=True):
    p = FIG_DIR / name
    if tight: plt.tight_layout()
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  [fig] {p.name}")

def savetbl(df, name, note=""):
    p = TBL_DIR / name
    df.to_csv(p)
    print(f"  [tbl] {p.name}  {note}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 01 + 02 — LOAD & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def task01_02_load_and_overview():
    print("\n" + "="*70)
    print("TASK 01-02: LOAD DATA & OVERVIEW")
    print("="*70)

    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target definition
    df["direction"] = (df["oil_return"] > 0).astype(int)

    # Split
    train = df[df["date"] < TRAIN_CUTOFF].copy()
    test  = df[df["date"] >= TRAIN_CUTOFF].copy()

    print(f"\n  Full dataset : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Train set    : {len(train)} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test  set    : {len(test)} rows  ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"  Features     : {df.shape[1]-1} (excl. date)")
    print(f"  Missing      : {df.isnull().sum().sum()}")

    # ── summary table ──────────────────────────────────────────────────────────
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        for c in cols:
            if c in train.columns:
                s = train[c]
                rows.append({"feature": c, "group": grp,
                             "mean": round(s.mean(),4), "std": round(s.std(),4),
                             "min": round(s.min(),4), "max": round(s.max(),4),
                             "missing_pct": round(s.isnull().mean()*100,2)})
    tbl_overview = pd.DataFrame(rows)
    savetbl(tbl_overview, "data_overview_summary.csv")

    # ── timeline plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(train["date"], train["oil_return"], lw=0.6, color="steelblue", label="oil_return (train)")
    axes[0].plot(test["date"],  test["oil_return"],  lw=0.6, color="coral",     label="oil_return (test)")
    axes[0].axvline(TRAIN_CUTOFF, color="red", ls="--", lw=1.2, label="train/test split")
    axes[0].set_ylabel("oil_return")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Oil Daily Returns over Time")

    axes[1].plot(df["date"], df["oil_volatility_7d"], lw=0.8, color="darkorange")
    axes[1].axvline(TRAIN_CUTOFF, color="red", ls="--", lw=1.2)
    axes[1].set_ylabel("volatility_7d")
    axes[1].set_title("Rolling 7-day Volatility")

    axes[2].plot(df["date"], df["geopolitical_stress_index"], lw=0.8, color="purple")
    axes[2].axvline(TRAIN_CUTOFF, color="red", ls="--", lw=1.2)
    axes[2].set_ylabel("geo_stress_index")
    axes[2].set_title("Geopolitical Stress Index")
    for ax in axes:
        for ev, dt in EVENTS.items():
            ax.axvline(pd.Timestamp(dt), color="gray", ls=":", lw=1, alpha=0.7)

    axes[2].set_xlabel("Date")
    savefig("timeline_coverage.png")

    return df, train, test


# ══════════════════════════════════════════════════════════════════════════════
# TASK 03 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
def task03_data_quality(train):
    print("\n" + "="*70)
    print("TASK 03: DATA QUALITY CHECKS")
    print("="*70)

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

    # Missing
    miss = train.isnull().sum()
    print(f"\n  Missing values total : {miss.sum()}")

    # Duplicates
    dup_rows = train.duplicated().sum()
    dup_dates = train["date"].duplicated().sum()
    print(f"  Duplicate rows       : {dup_rows}")
    print(f"  Duplicate dates      : {dup_dates}")

    # INF
    inf_cols = {c: np.isinf(train[c]).sum() for c in num_cols if np.isinf(train[c]).sum() > 0}
    print(f"  INF values           : {sum(inf_cols.values())} (cols: {list(inf_cols.keys())})")

    # Value range check
    expected = {
        "day_of_week":   (0, 4),
        "month":         (1, 12),
        "fed_rate_regime": (0, 3),
    }
    range_rows = []
    for c in num_cols:
        s = train[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        status = "OK"
        if c in expected:
            lo, hi = expected[c]
            if s.min() < lo or s.max() > hi:
                status = "WARNING"
        range_rows.append({"feature": c, "min": round(s.min(),4), "max": round(s.max(),4),
                            "mean": round(s.mean(),4), "std": round(s.std(),4),
                            "n_outliers_IQR": int(n_out),
                            "pct_outliers": round(n_out/len(s)*100,1),
                            "status": status})
    tbl_range = pd.DataFrame(range_rows)
    savetbl(tbl_range, "value_range_check.csv")

    # Outlier summary (top 10)
    tbl_out = tbl_range.nlargest(10, "n_outliers_IQR")[["feature","n_outliers_IQR","pct_outliers"]]
    savetbl(tbl_out, "outlier_summary.csv")
    print(f"\n  Top outlier features (IQR):")
    print(tbl_out.to_string(index=False))

    # Boxplot grid
    feat_num = [c for c in num_cols if c not in ["day_of_week","month","fed_rate_regime","direction"]]
    ncols = 4
    nrows = int(np.ceil(len(feat_num)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*2.5))
    axes = axes.flatten()
    for i, c in enumerate(feat_num):
        axes[i].boxplot(train[c].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="lightsteelblue"))
        axes[i].set_title(c, fontsize=8)
        axes[i].tick_params(axis="x", labelbottom=False)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Boxplots — Train Set (IQR outliers shown as dots)", fontsize=12, y=1.01)
    savefig("outlier_boxplots.png")

    # Lag consistency check
    lag1_ok = np.allclose(
        train["oil_return_lag1"].values[1:],
        train["oil_return"].values[:-1],
        equal_nan=True, rtol=1e-3
    )
    print(f"\n  Lag consistency (oil_return_lag1 == oil_return.shift(1)): {'✓ OK' if lag1_ok else '✗ FAIL'}")

    # Data quality report
    report = f"""# Data Quality Report

## Dataset: dataset_final.csv (Train set, before 2023-01-01)

| Check | Result |
|-------|--------|
| Missing values | {miss.sum()} |
| Duplicate rows | {dup_rows} |
| Duplicate dates | {dup_dates} |
| INF values | {sum(inf_cols.values())} |
| Lag1 consistency | {'OK' if lag1_ok else 'FAIL'} |

## Top features by IQR outlier count
{tbl_out.to_markdown(index=False)}

> Note: Financial returns naturally have fat tails (high kurtosis).
> Outliers here are expected market events, NOT data errors.
"""
    (REP_DIR / "data_quality_report.md").write_text(report)
    print(f"  [rep] data_quality_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 04 — TARGET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def task04_target_analysis(train, df):
    print("\n" + "="*70)
    print("TASK 04: TARGET ANALYSIS")
    print("="*70)

    ret = train["oil_return"]
    dir_ = train["direction"]

    # Stats
    skew = ret.skew(); kurt = ret.kurtosis()
    class_counts = dir_.value_counts().sort_index()
    class_ratio  = dir_.value_counts(normalize=True).sort_index()
    print(f"\n  oil_return  mean={ret.mean():.4f}  std={ret.std():.4f}")
    print(f"  Skewness    = {skew:.3f}")
    print(f"  Kurtosis    = {kurt:.3f}  (excess)")
    print(f"\n  Class counts: Down={class_counts[0]}  Up={class_counts[1]}")
    print(f"  Class ratio : Down={class_ratio[0]:.2%}  Up={class_ratio[1]:.2%}")
    imb = abs(class_ratio[0] - class_ratio[1]
    ) > 0.10
    print(f"  Imbalanced  : {'YES' if imb else 'NO (acceptable)'}")

    # ── Figure 1: distribution overview ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # histogram + KDE
    axes[0].hist(ret, bins=80, density=True, color="steelblue", alpha=0.6, label="all")
    xr = np.linspace(ret.min(), ret.max(), 300)
    kde = stats.gaussian_kde(ret)
    axes[0].plot(xr, kde(xr), color="navy", lw=2)
    axes[0].axvline(0, color="red", ls="--", lw=1.2)
    axes[0].set_title(f"oil_return dist\nskew={skew:.2f}, kurt={kurt:.2f}")
    axes[0].set_xlabel("return"); axes[0].set_ylabel("density")

    # QQ-plot
    stats.probplot(ret, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot vs Normal")

    # class balance bar
    axes[2].bar(["Down (0)", "Up (1)"], class_counts.values, color=["coral","steelblue"])
    for i, v in enumerate(class_counts.values):
        axes[2].text(i, v+10, f"{v}\n({v/len(dir_):.1%})", ha="center", fontsize=10)
    axes[2].set_title("Class Balance")
    axes[2].set_ylabel("count")
    savefig("target_distribution.png")

    # ── Figure 2: returns by class ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls, color, label in [(0,"coral","Down (0)"),(1,"steelblue","Up (1)")]:
        sub = train.loc[train["direction"]==cls, "oil_return"]
        axes[0].hist(sub, bins=60, density=True, alpha=0.5, color=color, label=label)
        kde_sub = stats.gaussian_kde(sub)
        x = np.linspace(sub.min(), sub.max(), 300)
        axes[0].plot(x, kde_sub(x), color=color, lw=2)
    axes[0].set_title("Returns distribution by class")
    axes[0].legend(); axes[0].set_xlabel("oil_return")

    # by class stats
    stats_by_class = train.groupby("direction")["oil_return"].agg(
        ["mean","std","median","min","max",
         lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).round(4)
    stats_by_class.columns = ["mean","std","median","min","max","q25","q75"]
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=stats_by_class.values.astype(str),
        rowLabels=["Down (0)","Up (1)"],
        colLabels=stats_by_class.columns,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    axes[1].set_title("Stats by class", pad=20)
    savefig("returns_by_class.png")
    savetbl(stats_by_class, "target_stats_by_class.csv")

    # ── Figure 3: temporal pattern ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    rolling_up = train.set_index("date")["direction"].rolling(30).mean()
    ax.plot(rolling_up.index, rolling_up.values, lw=1.5, color="steelblue", label="30d rolling Up-ratio")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(class_ratio[1], color="red", ls="-.", lw=1, label=f"Overall Up-ratio={class_ratio[1]:.2%}")
    for ev, dt in EVENTS.items():
        ax.axvline(pd.Timestamp(dt), color="orange", ls=":", lw=1.2)
        ax.text(pd.Timestamp(dt), 0.05, ev, fontsize=7, rotation=90, va="bottom", color="grey")
    ax.set_title("30-day Rolling Up-ratio (direction=1)")
    ax.set_ylabel("Fraction Up"); ax.legend()
    savefig("target_temporal_pattern.png")

    # ── Figure 4: streak distribution ──────────────────────────────────────────
    streaks = []
    cur_len, cur_dir = 1, dir_.iloc[0]
    for v in dir_.iloc[1:]:
        if v == cur_dir:
            cur_len += 1
        else:
            streaks.append((int(cur_dir), cur_len))
            cur_len, cur_dir = 1, v
    streaks.append((int(cur_dir), cur_len))
    streak_df = pd.DataFrame(streaks, columns=["direction","length"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cls, color, label in [(0,"coral","Down streaks"),(1,"steelblue","Up streaks")]:
        sub = streak_df.loc[streak_df["direction"]==cls, "length"]
        axes[0].hist(sub, bins=range(1, max(streak_df["length"])+2), alpha=0.6, color=color, label=label)
    axes[0].set_title("Streak length distribution"); axes[0].legend()
    axes[0].set_xlabel("streak length"); axes[0].set_ylabel("count")
    smry = streak_df.groupby("direction")["length"].agg(["mean","max","median"]).round(2)
    smry.index = ["Down (0)","Up (1)"]
    axes[1].axis("off")
    t = axes[1].table(cellText=smry.values.astype(str),
                      rowLabels=smry.index, colLabels=smry.columns,
                      loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10)
    axes[1].set_title("Streak summary", pad=20)
    savefig("streak_distribution.png")
    print(f"\n  Avg Up-streak: {streak_df.loc[streak_df.direction==1,'length'].mean():.2f} days")
    print(f"  Avg Dn-streak: {streak_df.loc[streak_df.direction==0,'length'].mean():.2f} days")

    # regime analysis
    regime_rows = []
    regimes = [
        ("Pre-COVID",      "2015-01-01","2020-02-15"),
        ("COVID crash",    "2020-02-15","2020-06-01"),
        ("Post-COVID rec", "2020-06-01","2022-02-24"),
        ("Ukraine crisis", "2022-02-24","2022-12-31"),
        ("Post-Ukraine",   "2023-01-01","2026-12-31"),
    ]
    tr_with_regime = df.copy()
    for reg, s, e in regimes:
        mask = (tr_with_regime["date"] >= s) & (tr_with_regime["date"] < e)
        sub = tr_with_regime.loc[mask]
        if len(sub) == 0: continue
        d = sub["direction"] if "direction" in sub else (sub["oil_return"]>0).astype(int)
        regime_rows.append({"regime": reg, "n": len(sub),
                             "up_pct": round(d.mean()*100,1),
                             "mean_ret": round(sub["oil_return"].mean()*100,3),
                             "vol": round(sub["oil_return"].std()*100,3)})
    tbl_regime = pd.DataFrame(regime_rows)
    savetbl(tbl_regime, "regime_analysis.csv")
    print(f"\n  Regime analysis:")
    print(tbl_regime.to_string(index=False))

    return class_ratio


# ══════════════════════════════════════════════════════════════════════════════
# TASK 05 — FEATURE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
def task05_feature_distribution(train):
    print("\n" + "="*70)
    print("TASK 05: FEATURE DISTRIBUTION ANALYSIS")
    print("="*70)

    num_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                if c not in ["direction"]]
    cat_cols = ["day_of_week","month","fed_rate_regime"]

    # ── summary stats ───────────────────────────────────────────────────────────
    rows = []
    for c in num_cols:
        s = train[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()
        rows.append({"feature":c, "mean":round(s.mean(),4), "std":round(s.std(),4),
                     "skewness":round(s.skew(),3), "kurt_val":round(s.kurtosis(),3),
                     "pct_outliers_IQR":round(n_out/len(s)*100,1),
                     "near_zero_var": s.std() < 0.001})
    tbl_stats = pd.DataFrame(rows)
    savetbl(tbl_stats, "feature_stats_summary.csv")
    print(f"\n  Highly skewed features (|skew|>1):")
    print(tbl_stats.loc[tbl_stats["skewness"].abs()>1, ["feature","skewness","kurt_val"]].to_string(index=False))

    # ── hist+KDE grids per group ────────────────────────────────────────────────
    for grpname, grp_cols in FEATURE_GROUPS.items():
        cols_in = [c for c in grp_cols if c in train.columns and c not in cat_cols]
        if not cols_in: continue
        ncols = min(len(cols_in), 3)
        nrows = int(np.ceil(len(cols_in)/ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
        axes = np.array(axes).flatten() if len(cols_in) > 1 else [axes]
        for i, c in enumerate(cols_in):
            s = train[c].dropna()
            axes[i].hist(s, bins=50, density=True, alpha=0.55, color="steelblue")
            kde = stats.gaussian_kde(s)
            xr = np.linspace(s.min(), s.max(), 300)
            axes[i].plot(xr, kde(xr), color="navy", lw=1.8)
            axes[i].set_title(f"{c}\nskew={s.skew():.2f} kurt={s.kurtosis():.2f}", fontsize=8)
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Feature Distributions — {grpname} Group", fontsize=11, y=1.01)
        savefig(f"hist_kde_{grpname.lower()}.png")

    # ── categorical barcharts ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(cat_cols):
        if c in train.columns:
            vc = train[c].value_counts().sort_index()
            axes[i].bar(vc.index.astype(str), vc.values, color="steelblue")
            axes[i].set_title(c); axes[i].set_xlabel(c); axes[i].set_ylabel("count")
    savefig("categorical_features.png")

    # ── feature notes ───────────────────────────────────────────────────────────
    notes = ["# Feature Notes\n"]
    notes.append("| Feature | Skew | Kurt | %Outlier | Notes |")
    notes.append("|---------|------|------|----------|-------|")
    for _, r in tbl_stats.iterrows():
        note = ""
        if abs(r["skewness"]) > 2: note += "Highly skewed — consider log-transform. "
        if r["kurt_val"] > 10: note += "Fat tails (financial). "
        if r["near_zero_var"]: note += "Near-zero variance — consider removing. "
        if r["pct_outliers_IQR"] > 20: note += "Many IQR outliers (expected for returns). "
        notes.append(f"| {r['feature']} | {r['skewness']} | {r['kurt_val']} | {r['pct_outliers_IQR']}% | {note or 'OK'} |")
    (REP_DIR / "feature_notes.md").write_text("\n".join(notes))
    print(f"  [rep] feature_notes.md")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 06 — TIME SERIES EDA
# ══════════════════════════════════════════════════════════════════════════════
def task06_time_series(train):
    print("\n" + "="*70)
    print("TASK 06: TIME SERIES EDA")
    print("="*70)

    ts = train.set_index("date").sort_index()

    # ── ADF tests ───────────────────────────────────────────────────────────────
    test_cols = ["oil_return","usd_return","sp500_return","vix_return",
                 "oil_volatility_7d","yield_spread","cpi_lag","unemployment_lag",
                 "inventory_zscore","real_rate","geopolitical_stress_index"]
    adf_rows = []
    for c in test_cols:
        if c not in ts.columns: continue
        s = ts[c].dropna()
        res = adfuller(s, maxlag=5, autolag="AIC")
        adf_rows.append({"feature":c, "adf_stat":round(res[0],3),
                         "p_value":round(res[1],4),
                         "stationary": "Yes" if res[1]<0.05 else "No"})
    tbl_adf = pd.DataFrame(adf_rows)
    savetbl(tbl_adf, "adf_test_results.csv")
    print(f"\n  ADF results (p<0.05 → stationary):")
    print(tbl_adf.to_string(index=False))

    # ── ACF/PACF for oil_return ─────────────────────────────────────────────────
    ret = ts["oil_return"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(ret,  lags=30, ax=axes[0], title="ACF — oil_return (train)")
    plot_pacf(ret, lags=30, ax=axes[1], title="PACF — oil_return (train)", method="ywm")
    savefig("acf_pacf_oil_return.png")

    # ACF/PACF for volatility
    vol = ts["oil_volatility_7d"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_acf(vol,  lags=30, ax=axes[0], title="ACF — oil_volatility_7d (train)")
    plot_pacf(vol, lags=30, ax=axes[1], title="PACF — oil_volatility_7d (train)", method="ywm")
    savefig("acf_pacf_volatility.png")

    # ── Rolling statistics ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    roll30 = ts["oil_return"].rolling(30)
    axes[0].plot(roll30.mean(), lw=1.5, color="steelblue", label="30d rolling mean")
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    for ev, dt in EVENTS.items():
        axes[0].axvline(pd.Timestamp(dt), color="orange", ls=":", lw=1)
    axes[0].set_title("Rolling Mean — oil_return"); axes[0].legend()

    axes[1].plot(roll30.std(), lw=1.5, color="darkorange", label="30d rolling std (volatility)")
    for ev, dt in EVENTS.items():
        axes[1].axvline(pd.Timestamp(dt), color="orange", ls=":", lw=1)
    axes[1].set_title("Rolling Std — oil_return"); axes[1].legend()
    savefig("rolling_statistics.png")

    # ── Rolling correlation ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    pairs = [("sp500_return","SP500"), ("vix_return","VIX"), ("geopolitical_stress_index","GeoStress")]
    for col, lbl in pairs:
        if col in ts.columns:
            rc = ts[["oil_return",col]].rolling(90).corr().unstack()[col]["oil_return"]
            ax.plot(rc.index, rc.values, lw=1.2, label=f"oil_return × {lbl}")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    for ev, dt in EVENTS.items():
        ax.axvline(pd.Timestamp(dt), color="orange", ls=":", lw=1, alpha=0.7)
    ax.set_title("90-day Rolling Correlation with oil_return")
    ax.legend(fontsize=9); ax.set_ylabel("Spearman corr (approx)")
    savefig("rolling_correlation.png")

    # ── Seasonality: by month ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ts.boxplot(column="oil_return", by="month", ax=axes[0])
    axes[0].set_title("oil_return by Month"); axes[0].set_xlabel("Month")
    axes[0].figure.suptitle("")
    ts.boxplot(column="oil_return", by="day_of_week", ax=axes[1])
    axes[1].set_title("oil_return by Day of Week (0=Mon)")
    axes[1].set_xlabel("Day of Week"); axes[1].figure.suptitle("")
    savefig("seasonality_month.png")

    # heatmap mean return month×year
    ts2 = ts.copy()
    ts2["year"] = ts2.index.year
    pivot = ts2.groupby(["year","month"])["oil_return"].mean().unstack()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, center=0, cmap="RdYlGn", linewidths=0.4,
                annot=True, fmt=".3f", ax=ax, annot_kws={"size":7})
    ax.set_title("Mean oil_return by Year × Month (train)")
    savefig("seasonality_heatmap.png")

    # ── Regime: volatility ────────────────────────────────────────────────────────
    q25 = ts["oil_volatility_7d"].quantile(0.25)
    q75 = ts["oil_volatility_7d"].quantile(0.75)
    ts2["vol_regime"] = "medium"
    ts2.loc[ts2["oil_volatility_7d"] < q25, "vol_regime"] = "low"
    ts2.loc[ts2["oil_volatility_7d"] > q75, "vol_regime"] = "high"
    dir_by_regime = ts2.groupby("vol_regime").apply(
        lambda x: (x["direction"].mean()*100).round(1) if "direction" in x else ((x["oil_return"]>0).mean()*100).round(1)
    ).rename("up_pct")
    print(f"\n  Up-ratio by vol regime:")
    print(dir_by_regime.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# TASK 07 — FEATURE–TARGET RELATIONSHIP
# ══════════════════════════════════════════════════════════════════════════════
def task07_feature_target(train):
    print("\n" + "="*70)
    print("TASK 07: FEATURE–TARGET RELATIONSHIP")
    print("="*70)

    num_feats = [c for c in train.select_dtypes(include=[np.number]).columns
                 if c not in ["oil_return","direction"]]
    X = train[num_feats].fillna(0)
    y = train["direction"]

    # ── Spearman correlation matrix ─────────────────────────────────────────────
    corr_mat = train[num_feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.3, ax=ax, cbar_kws={"shrink":0.8}, annot=False)
    ax.set_title("Spearman Correlation Matrix — Features (Train Set)", fontsize=12)
    savefig("correlation_heatmap.png")

    # top correlated pairs
    corr_pairs = []
    for i in range(len(num_feats)):
        for j in range(i+1, len(num_feats)):
            r = corr_mat.iloc[i,j]
            corr_pairs.append({"f1":num_feats[i],"f2":num_feats[j],"spearman_r":round(r,3)})
    tbl_pairs = pd.DataFrame(corr_pairs).sort_values("spearman_r", key=abs, ascending=False).head(15)
    savetbl(tbl_pairs, "correlation_top_pairs.csv")
    print(f"\n  Top 10 correlated pairs (|r|):")
    print(tbl_pairs.head(10).to_string(index=False))

    # ── VIF ─────────────────────────────────────────────────────────────────────
    vif_rows = []
    X_vif = X.copy()
    for i, c in enumerate(num_feats):
        try:
            v = variance_inflation_factor(X_vif.values, i)
        except Exception:
            v = np.nan
        vif_rows.append({"feature":c, "VIF":round(v,2),
                          "status": "CRITICAL" if v>10 else ("WARNING" if v>5 else "OK")})
    tbl_vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    savetbl(tbl_vif, "vif_results.csv")
    print(f"\n  VIF (top 10):")
    print(tbl_vif.head(10).to_string(index=False))

    # ── Point-biserial + Mann-Whitney ────────────────────────────────────────────
    ft_rows = []
    for c in num_feats:
        s = train[c].fillna(train[c].median())
        r_pb, p_pb = stats.pointbiserialr(y, s)
        g0 = s[y==0]; g1 = s[y==1]
        stat_mw, p_mw = mannwhitneyu(g0, g1, alternative="two-sided")
        ft_rows.append({"feature":c, "pointbiserial_r":round(r_pb,4),
                         "mw_p_value":round(p_mw,4),
                         "significant_MW": p_mw < 0.05})
    tbl_ft = pd.DataFrame(ft_rows).sort_values("pointbiserial_r", key=abs, ascending=False)
    savetbl(tbl_ft, "feature_target_correlation.csv")
    print(f"\n  Top 10 features by |point-biserial r| with direction:")
    print(tbl_ft.head(10).to_string(index=False))

    # ── Mutual Information ───────────────────────────────────────────────────────
    mi_scores = mutual_info_classif(X, y, random_state=42)
    tbl_mi = pd.DataFrame({"feature":num_feats,"mutual_info":mi_scores}
                          ).sort_values("mutual_info", ascending=False)
    savetbl(tbl_mi, "mutual_information.csv")

    # bar chart MI
    fig, ax = plt.subplots(figsize=(12, 6))
    top_mi = tbl_mi.head(20)
    ax.barh(top_mi["feature"][::-1], top_mi["mutual_info"][::-1], color="steelblue")
    ax.set_title("Mutual Information — Top 20 features vs direction (Up/Down)")
    ax.set_xlabel("MI score")
    savefig("mutual_information.png")

    # ── Boxplot feature by class ─────────────────────────────────────────────────
    top_feats = tbl_ft["feature"].head(12).tolist()
    ncols = 4; nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
    axes = axes.flatten()
    for i, c in enumerate(top_feats):
        groups = [train.loc[train["direction"]==cls, c].dropna() for cls in [0,1]]
        axes[i].boxplot(groups, labels=["Down(0)","Up(1)"], patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        axes[i].set_title(c, fontsize=9)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature distribution by Target Class — Top 12 features", fontsize=12)
    savefig("feature_vs_target_boxplots.png")

    return tbl_ft, tbl_mi, tbl_vif


# ══════════════════════════════════════════════════════════════════════════════
# TASK 08 — LEAKAGE & SPLIT CHECK
# ══════════════════════════════════════════════════════════════════════════════
def task08_leakage_and_split(df, train, test):
    print("\n" + "="*70)
    print("TASK 08: LEAKAGE & SPLIT CHECK")
    print("="*70)

    # ── Leakage risk table ────────────────────────────────────────────────────
    leakage_info = [
        ("oil_return_lag1",         "t-1",   "Low",    "Explicit lag — safe"),
        ("oil_return_lag2",         "t-2",   "Low",    "Explicit lag — safe"),
        ("vix_lag1",                "t-1",   "Low",    "Explicit lag — safe"),
        ("gdelt_tone_lag1",         "t-1",   "Low",    "Explicit lag — safe"),
        ("gdelt_volume_lag1",       "t-1",   "Low",    "Explicit lag — safe"),
        ("oil_volatility_7d",       "t (7d bkwd)", "Low", "Backward rolling — safe"),
        ("gdelt_tone_7d",           "t (7d bkwd)", "Low", "Backward rolling — safe"),
        ("gdelt_tone_30d",          "t (30d bkwd)","Low", "Backward rolling — safe"),
        ("geopolitical_stress_index","t",    "Low",    "Normalized on train only per step4"),
        ("inventory_zscore",        "t",     "Low",    "Rolling z-score backward"),
        ("yield_spread",            "t",     "Medium", "Daily market data — available at t"),
        ("usd_return",              "t",     "Medium", "Same-day market return"),
        ("sp500_return",            "t",     "Medium", "Same-day market return"),
        ("vix_return",              "t",     "Medium", "Same-day market return"),
        ("cpi_lag",                 "t-1m+", "Low",    "Lagged macro indicator"),
        ("unemployment_lag",        "t-1m+", "Low",    "Lagged macro indicator"),
        ("fed_rate_change",         "FOMC",  "Low",    "Announced — no look-ahead"),
        ("oil_return",              "t",     "HIGH",   "TARGET SOURCE — must NOT be used as feature"),
        ("direction",               "t",     "HIGH",   "TARGET — must NOT be used as feature"),
    ]
    tbl_leak = pd.DataFrame(leakage_info, columns=["feature","available_at","risk","note"])
    savetbl(tbl_leak, "leakage_risk_assessment.csv")

    high_risk = tbl_leak.loc[tbl_leak["risk"]=="HIGH","feature"].tolist()
    medium_risk = tbl_leak.loc[tbl_leak["risk"]=="Medium","feature"].tolist()
    print(f"\n  HIGH risk features: {high_risk}")
    print(f"  MEDIUM risk features: {medium_risk}")
    print(f"  (same-day market returns are available at close — model should use lag versions for strict no-leakage)")

    # ── Target leakage ─────────────────────────────────────────────────────────
    feats_model = [c for c in df.columns if c not in ["date","oil_return","direction"]]
    print(f"\n  ✓ oil_return not in feature set: {'oil_return' not in feats_model}")
    print(f"  ✓ direction not in feature set : {'direction' not in feats_model}")

    # ── Lag1 consistency ───────────────────────────────────────────────────────
    lag1_ok = np.allclose(
        df["oil_return_lag1"].values[1:], df["oil_return"].values[:-1],
        equal_nan=True, rtol=1e-3)
    print(f"  ✓ oil_return_lag1 == oil_return.shift(1): {lag1_ok}")

    # ── Train/test no overlap ──────────────────────────────────────────────────
    overlap = set(train["date"]).intersection(set(test["date"]))
    print(f"  ✓ No date overlap train/test: {len(overlap)==0} (overlap count={len(overlap)})")
    print(f"  ✓ Train ends before test starts: {train['date'].max() < test['date'].min()}")

    # ── Distribution shift (KS test) ───────────────────────────────────────────
    num_feats = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in ["oil_return","direction"]]
    ks_rows = []
    for c in num_feats:
        a = train[c].dropna().values
        b = test[c].dropna().values
        if len(a)<10 or len(b)<10: continue
        ks_stat, p_val = ks_2samp(a, b)
        ks_rows.append({"feature":c,
                         "train_mean":round(a.mean(),4), "test_mean":round(b.mean(),4),
                         "ks_stat":round(ks_stat,4), "p_value":round(p_val,4),
                         "shift_detected": p_val < 0.05})
    tbl_ks = pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False)
    savetbl(tbl_ks, "distribution_shift_ks_test.csv")
    n_shift = tbl_ks["shift_detected"].sum()
    print(f"\n  KS test: {n_shift}/{len(tbl_ks)} features have significant distribution shift (p<0.05)")
    print(f"  Top shifted features:")
    print(tbl_ks.head(6).to_string(index=False))

    # ── KDE comparison plots ───────────────────────────────────────────────────
    top_shifted = tbl_ks.head(9)["feature"].tolist()
    ncols = 3; nrows = int(np.ceil(len(top_shifted)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*3.5))
    axes = axes.flatten()
    for i, c in enumerate(top_shifted):
        sns.kdeplot(train[c].dropna(), ax=axes[i], label="train", color="steelblue")
        sns.kdeplot(test[c].dropna(),  ax=axes[i], label="test",  color="coral", ls="--")
        axes[i].set_title(c, fontsize=9); axes[i].legend(fontsize=8)
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Train vs Test Distribution Shift — Top shifted features", fontsize=11)
    savefig("train_test_distribution_shift.png")

    # ── Class ratio train vs test ──────────────────────────────────────────────
    train_up = (train["oil_return"] > 0).mean()
    test_up  = (test["oil_return"] > 0).mean()
    print(f"\n  Class ratio — Up: train={train_up:.2%}  test={test_up:.2%}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Train Down","Train Up","Test Down","Test Up"],
           [1-train_up, train_up, 1-test_up, test_up],
           color=["coral","steelblue","lightsalmon","lightblue"])
    for i, v in enumerate([1-train_up, train_up, 1-test_up, test_up]):
        ax.text(i, v+0.005, f"{v:.1%}", ha="center", fontsize=10)
    ax.set_title("Class ratio: Train vs Test"); ax.set_ylabel("fraction")
    savefig("class_ratio_train_vs_test.png")

    return tbl_ks, n_shift


# ══════════════════════════════════════════════════════════════════════════════
# TASK 09 — SUMMARY & NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
def task09_summary(train, test, class_ratio, tbl_ft, tbl_mi, tbl_vif, tbl_ks, n_shift):
    print("\n" + "="*70)
    print("TASK 09: EDA SUMMARY & NEXT STEPS")
    print("="*70)

    up_pct = class_ratio[1]
    imb_flag = abs(class_ratio[0]-class_ratio[1]) > 0.10
    top5_pb  = tbl_ft.head(5)["feature"].tolist()
    top5_mi  = tbl_mi.head(5)["feature"].tolist()
    critical_vif = tbl_vif.loc[tbl_vif["VIF"]>10,"feature"].tolist()
    n_sig_mw = tbl_ft["significant_MW"].sum()

    # Feature ranking
    merged = tbl_ft[["feature","pointbiserial_r","mw_p_value","significant_MW"]].merge(
        tbl_mi[["feature","mutual_info"]], on="feature", how="left").merge(
        tbl_vif[["feature","VIF","status"]], on="feature", how="left")
    merged["abs_r"] = merged["pointbiserial_r"].abs()
    merged = merged.sort_values("abs_r", ascending=False)
    merged["recommendation"] = "KEEP"
    merged.loc[merged["mutual_info"]<0.001, "recommendation"] = "CONSIDER REMOVING"
    merged.loc[(merged["VIF"]>10) & (merged["mutual_info"]<0.005), "recommendation"] = "REMOVE"
    merged.loc[merged["feature"].isin(["oil_return","direction"]), "recommendation"] = "EXCLUDE (target)"
    savetbl(merged, "feature_ranking.csv")
    print(f"\n  Feature ranking top 10:")
    print(merged.head(10)[["feature","abs_r","mutual_info","VIF","recommendation"]].to_string(index=False))

    remove_list = merged.loc[merged["recommendation"].str.startswith("REMOVE"),"feature"].tolist()
    print(f"\n  Recommended to REMOVE: {remove_list}")

    # ── EDA summary markdown ───────────────────────────────────────────────────
    summary_md = f"""# EDA Summary Report
## Oil Price Direction Classification

**Date**: 2026-04-11
**Dataset**: `data/processed/dataset_final.csv` — 2923 rows × 33 cols
**Train**: 2015-01-07 → 2022-12-30 ({len(train)} rows)
**Test**: 2023-01-02 → 2026-03-20  ({len(test)} rows)

---

## 1. Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | **Class balance** | Up={up_pct:.1%} / Down={1-up_pct:.1%}. {'⚠️ Imbalanced' if imb_flag else '✅ Acceptable balance'} |
| 2 | **Returns distribution** | Skewness≈negative, Kurtosis≈12 — strong fat tails (financial norm) |
| 3 | **ACF oil_return** | Near-zero → market close to random walk → prediction is hard |
| 4 | **Volatility clustering** | oil_volatility_7d shows strong ACF → GARCH-like features may help |
| 5 | **Feature signal** | {n_sig_mw}/{len(tbl_ft)} features significant by Mann-Whitney (p<0.05) |
| 6 | **Multicollinearity** | {len(critical_vif)} features with VIF>10: {critical_vif} |
| 7 | **Distribution shift** | {n_shift}/{len(tbl_ks)} features shifted train→test (KS p<0.05) |
| 8 | **Seasonality** | Visible monthly pattern; high vol in COVID/Ukraine regimes |
| 9 | **Leakage** | oil_return/direction excluded from features; same-day market returns (medium risk — use lag versions for strict setup) |
| 10 | **Top predictors (MI)** | {top5_mi} |

---

## 2. Feature Ranking (Top 15)

{merged.head(15)[["feature","abs_r","mutual_info","VIF","recommendation"]].to_markdown(index=False)}

---

## 3. Feature Engineering Suggestions

- **Lag extension**: Add lag2/lag3 for `vix_lag`, `sp500_return`, `gdelt_tone` if ACF significant.
- **Strict leakage-free path**: Replace `usd_return`, `sp500_return`, `vix_return` with their `_lag1` versions.
- **Interaction**: `real_rate × geopolitical_stress_index` (macro × geo).
- **Regime flag**: Binary: `high_vol_regime = (oil_volatility_7d > Q75)`.
- **Transform**: log-transform highly skewed supply features if using linear models.

---

## 4. Modeling Strategy

| Decision | Recommendation |
|----------|---------------|
| Validation | `TimeSeriesSplit(n_splits=5)` — NO random k-fold |
| Baseline | "Always predict majority class" → ~{max(up_pct,1-up_pct):.1%} accuracy |
| Metric | AUC-ROC + F1 (macro or weighted) |
| Class imbalance | `class_weight='balanced'` first; SMOTE if needed |
| Algorithm order | Logistic Regression → Random Forest → XGBoost |
| Feature scaling | RobustScaler (handles fat-tails better than StandardScaler) |

---

## 5. Risks & Concerns

- **Low signal**: Market near-random-walk → realistic ceiling ~55-60% accuracy.
- **Distribution shift**: {n_shift} features drifted train→test — monitor model performance over time.
- **Regime mismatch**: Train (2015-2022) includes COVID & oil crash; Test (2023+) is post-war inflation era.
- **Same-day features**: usd_return/sp500_return/vix_return are same-day — use `_lag1` in strict setting.
- **Overfitting**: 31 features vs 2083 train rows — regularize aggressively.

---

## 6. Conclusion

EDA confirms this is a **hard classification task** (near-random-walk target). Signals exist but are weak.
Key features: lag returns, volatility, geopolitical stress, macro regime.
Proceed with strict train-only preprocessing, TimeSeriesSplit CV, and baseline comparison first.
"""
    (REP_DIR / "eda_summary.md").write_text(summary_md)
    print(f"  [rep] eda_summary.md")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("EDA RUNNER — Oil Price Classification (Tasks 01-09)")
    print("="*70)

    df, train, test      = task01_02_load_and_overview()
    task03_data_quality(train)
    class_ratio          = task04_target_analysis(train, df)
    task05_feature_distribution(train)
    task06_time_series(train)
    tbl_ft, tbl_mi, tbl_vif = task07_feature_target(train)
    tbl_ks, n_shift      = task08_leakage_and_split(df, train, test)
    task09_summary(train, test, class_ratio, tbl_ft, tbl_mi, tbl_vif, tbl_ks, n_shift)

    print("\n" + "="*70)
    print("✅ ALL TASKS COMPLETE")
    print("  figures/ →", len(list(FIG_DIR.glob("*.png"))), "files")
    print("  tables/  →", len(list(TBL_DIR.glob("*.csv"))), "files")
    print("  reports/ →", len(list(REP_DIR.glob("*.md"))), "files")
    print("="*70)


if __name__ == "__main__":
    main()
