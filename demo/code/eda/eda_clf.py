"""
EDA Classification - Oil Price Direction (UP/DOWN)

Upgraded classification EDA with:
  - forward-target aware splitting by oil_return_fwd1_date
  - data-quality and drift tables
  - leakage-aware feature risk assessment
  - grouped feature plots
  - signal + stability ranking

Usage:
  EDA_DATA_PATH=... EDA_OUT_DIR=... EDA_PREFIX=... python eda_clf.py
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    plot_acf = None
    plot_pacf = None
    adfuller = None
    HAS_STATSMODELS = False


plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "figure.facecolor": "white"})
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("EDA_DATA_PATH", str(ROOT / "data/processed/dataset_step4_transformed.csv")))
OUT = Path(os.getenv("EDA_OUT_DIR", str(Path(__file__).resolve().parent))).resolve()
PREFIX = os.getenv("EDA_PREFIX", "").strip()
RUN_LABEL = os.getenv("EDA_RUN_LABEL", PREFIX or DATA.stem).strip()
TARGET_RET = "oil_return_fwd1"
TARGET_DATE = "oil_return_fwd1_date"
SPLIT = pd.Timestamp(os.getenv("EDA_SPLIT_DATE", "2023-01-01"))

TABLE_DIR = OUT / "tables"
REPORT_DIR = OUT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

P = "=" * 90

EVENTS = [
    ("COVID", "2020-03-09"),
    ("Ukraine", "2022-02-24"),
    ("Train/Test", "2023-01-01"),
]


def out_name(name):
    return f"{PREFIX}_{name}" if PREFIX else name


def save_fig(name):
    path = OUT / out_name(name)
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close("all")
    print("    -> {}".format(path.name))


def save_table(df, name):
    path = TABLE_DIR / out_name(name)
    df.to_csv(path, index=False)
    print("    -> {}".format(path.relative_to(OUT)))


def save_report(text, name):
    path = REPORT_DIR / out_name(name)
    path.write_text(text)
    print("    -> {}".format(path.relative_to(OUT)))


def report_table(df, columns=None, max_rows=None):
    subset = df.copy()
    if columns is not None:
        subset = subset[columns]
    if max_rows is not None:
        subset = subset.head(max_rows)
    try:
        return subset.to_markdown(index=False)
    except Exception:
        return subset.to_string(index=False)


def safe_unique(series):
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return np.nan


def safe_skew(series):
    s = series.dropna()
    if len(s) < 3:
        return np.nan
    return float(s.skew())


def safe_kurt(series):
    s = series.dropna()
    if len(s) < 4:
        return np.nan
    return float(s.kurtosis())


def safe_outlier_pct(series):
    s = series.dropna()
    if len(s) < 10:
        return 0.0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    return float(mask.mean() * 100.0)


def safe_ks(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, pval = stats.ks_2samp(a, b)
    return float(stat), float(pval)


def safe_mwu(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, pval = mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(pval)


def safe_pointbiserial(y, x):
    x = pd.Series(x)
    mask = x.notna()
    if mask.sum() < 3 or pd.Series(y)[mask].nunique() < 2:
        return np.nan, np.nan
    r, p = stats.pointbiserialr(pd.Series(y)[mask], x[mask])
    return float(r), float(p)


def safe_cohens_d(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def normalize(series):
    s = pd.Series(series, dtype=float).fillna(0.0)
    mx = s.max()
    if mx <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s / mx


def safe_adf(series, maxlag=5):
    s = pd.Series(series).dropna()
    if len(s) < 20 or not HAS_STATSMODELS:
        return np.nan, np.nan
    try:
        stat, pval, *_ = adfuller(s, maxlag=maxlag, autolag="AIC")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def safe_vif_table(frame):
    if frame.empty:
        return pd.DataFrame(columns=["feature", "vif", "status"])
    filled = frame.copy()
    filled = filled.fillna(filled.median(numeric_only=True)).fillna(0.0)
    rows = []
    values = filled.to_numpy(dtype=float)
    for i, feature in enumerate(filled.columns):
        try:
            y = values[:, i]
            X = np.delete(values, i, axis=1)
            if X.shape[1] == 0:
                vif = 1.0
            else:
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                if r2 >= 0.999999:
                    vif = np.inf
                else:
                    vif = float(1.0 / (1.0 - r2))
        except Exception:
            vif = np.nan
        status = "ok"
        if not np.isnan(vif):
            if vif > 10:
                status = "critical"
            elif vif > 5:
                status = "warning"
        rows.append({"feature": feature, "vif": vif, "status": status})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def infer_group(feature):
    name = feature.lower()
    if name.startswith("day_of_week") or name.startswith("month"):
        return "Calendar"
    if "gdelt" in name:
        return "GDELT"
    if "conflict" in name or "fatalit" in name:
        return "Conflict"
    if "inventory" in name or "production" in name or "import" in name:
        return "Supply"
    if "cpi" in name or "fed" in name or "unemployment" in name or "yield" in name or "real_rate" in name:
        return "Macro"
    if "oil" in name or "sp500" in name or "usd" in name or "vix" in name or "wti" in name:
        return "Market"
    return "Other"


def leakage_risk(feature):
    name = feature.lower()
    if "fwd" in name or "target" in name:
        return "high", "Forward or target-derived field."
    if any(x in name for x in ["cpi", "unemployment", "fed_funds", "fed_rate", "real_rate", "recession"]):
        return "high", "Macro timing risk if not aligned to publication date."
    if name in {"oil_close", "usd_close", "sp500_close", "vix_close", "wti_fred"}:
        return "medium", "Same-day level feature; safe only for end-of-day forecasting."
    if name in {"oil_return", "usd_return", "sp500_return"} or name.startswith("vix_return"):
        return "medium", "Same-day return feature; safe only for end-of-day T -> T+1 setup."
    if "lag" in name or "_7d" in name or "_30d" in name or "zscore" in name or "log1p" in name or "slog1p" in name:
        return "low", "Backward-looking or deterministic transformed feature."
    if name.endswith("_sin") or name.endswith("_cos"):
        return "low", "Calendar cyclical encoding."
    return "low", "No obvious future or publication-timing risk in the feature name."


def build_group_map(features):
    rows = []
    for feature in features:
        rows.append({"feature": feature, "group": infer_group(feature)})
    return pd.DataFrame(rows)


def grouped_features(features):
    fmap = {}
    for feature in features:
        group = infer_group(feature)
        fmap.setdefault(group, []).append(feature)
    return fmap


print("\n{}\n LOAD\n{}".format(P, P))
print("  Run label: {}".format(RUN_LABEL))
print("  Dataset: {}".format(DATA))

parse_dates = ["date"]
probe = pd.read_csv(DATA, nrows=0)
if TARGET_DATE in probe.columns:
    parse_dates.append(TARGET_DATE)
df = pd.read_csv(DATA, parse_dates=parse_dates).sort_values("date").reset_index(drop=True)

if TARGET_RET not in df.columns:
    raise ValueError("{} not found in dataset {}".format(TARGET_RET, DATA))

df["direction"] = (df[TARGET_RET] > 0).astype(int)
split_col = TARGET_DATE if TARGET_DATE in df.columns else "date"
split_values = pd.to_datetime(df[split_col])

train = df[split_values < SPLIT].copy()
test = df[split_values >= SPLIT].copy()
meta_cols = ["date", TARGET_RET, TARGET_DATE, "direction"]
NUM = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

print("  Shape: {} | Train: {} | Test: {}".format(df.shape, len(train), len(test)))
print("  Numeric features: {}".format(len(NUM)))
if split_col != "date":
    print("  Train targets: {} -> {}".format(train[split_col].iloc[0].date(), train[split_col].iloc[-1].date()))
    print("  Test targets:  {} -> {}".format(test[split_col].iloc[0].date(), test[split_col].iloc[-1].date()))


print("\n{}\n 0. DATA QUALITY + OVERVIEW\n{}".format(P, P))
group_df = build_group_map(NUM)
overview_rows = []
for feature in NUM:
    s_full = df[feature]
    s_train = train[feature]
    s_test = test[feature]
    ks_shift, ks_shift_p = safe_ks(s_train, s_test)
    overview_rows.append({
        "feature": feature,
        "group": infer_group(feature),
        "dtype": str(df[feature].dtype),
        "missing_pct": round(float(s_full.isna().mean() * 100.0), 3),
        "n_unique": safe_unique(s_full),
        "train_mean": round(float(s_train.mean()), 6),
        "train_std": round(float(s_train.std()), 6),
        "test_mean": round(float(s_test.mean()), 6),
        "test_std": round(float(s_test.std()), 6),
        "skew": round(safe_skew(s_train), 6),
        "kurtosis": round(safe_kurt(s_train), 6),
        "outlier_pct_iqr": round(safe_outlier_pct(s_train), 3),
        "train_test_ks": round(ks_shift, 6) if not np.isnan(ks_shift) else np.nan,
        "train_test_ks_p": round(ks_shift_p, 6) if not np.isnan(ks_shift_p) else np.nan,
    })
overview_df = pd.DataFrame(overview_rows).sort_values(["group", "feature"]).reset_index(drop=True)
save_table(overview_df, "data_overview_summary.csv")
save_table(
    pd.DataFrame([
        {
            "run_label": RUN_LABEL,
            "dataset": str(DATA),
            "rows": len(df),
            "cols": df.shape[1],
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train["date"].iloc[0].date()),
            "train_end": str(train["date"].iloc[-1].date()),
            "test_start": str(test["date"].iloc[0].date()),
            "test_end": str(test["date"].iloc[-1].date()),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_dates": int(df["date"].duplicated().sum()),
            "inf_values": int(np.isinf(df[NUM].to_numpy(dtype=float)).sum()) if NUM else 0,
            "missing_values": int(df[NUM].isna().sum().sum()) if NUM else 0,
        }
    ]),
    "data_quality_summary.csv",
)

leak_rows = []
for feature in NUM:
    risk, why = leakage_risk(feature)
    leak_rows.append({"feature": feature, "group": infer_group(feature), "risk": risk, "why": why})
leak_df = pd.DataFrame(leak_rows).sort_values(["risk", "group", "feature"]).reset_index(drop=True)
save_table(leak_df, "leakage_risk_assessment.csv")

top_outliers = overview_df.sort_values("outlier_pct_iqr", ascending=False).head(15)
top_shift = overview_df.sort_values("train_test_ks", ascending=False).head(15)
group_counts = group_df["group"].value_counts().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].barh(top_outliers["feature"], top_outliers["outlier_pct_iqr"], color="#4C72B0")
axes[0].invert_yaxis()
axes[0].set_title("Top Train Outlier Rates (IQR)")
axes[0].set_xlabel("% rows")

axes[1].barh(top_shift["feature"], top_shift["train_test_ks"], color="#DD8452")
axes[1].invert_yaxis()
axes[1].set_title("Top Train/Test Shift (KS)")
axes[1].set_xlabel("KS stat")

axes[2].bar(group_counts.index, group_counts.values, color="#55A868")
axes[2].set_title("Feature Group Counts")
axes[2].tick_params(axis="x", rotation=35)

fig.suptitle("Data Quality and Coverage Overview", fontsize=14)
plt.tight_layout()
save_fig("00_data_quality_overview.png")


print("\n{}\n 1. CLASS DISTRIBUTION\n{}".format(P, P))
for name, subset in [("Full", df), ("Train", train), ("Test", test)]:
    vc = subset["direction"].value_counts()
    print("  {:>6}: UP={} ({:.1%})  DOWN={} ({:.1%})".format(
        name,
        vc.get(1, 0),
        vc.get(1, 0) / len(subset),
        vc.get(0, 0),
        vc.get(0, 0) / len(subset),
    ))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, subset) in enumerate([("Full", df), ("Train", train), ("Test", test)]):
    vc = subset["direction"].value_counts().sort_index()
    axes[i].bar(["DOWN (0)", "UP (1)"], vc.values, color=["#C44E52", "#55A868"], edgecolor="white")
    axes[i].set_title("{} (n={})".format(name, len(subset)))
    for j, value in enumerate(vc.values):
        axes[i].text(j, value + 5, "{}\n({:.1%})".format(value, value / len(subset)), ha="center", fontsize=9)
fig.suptitle("Class Distribution: UP vs DOWN", fontsize=14)
plt.tight_layout()
save_fig("01_class_distribution.png")

yearly = train.groupby(train["date"].dt.year)["direction"].value_counts().unstack(fill_value=0)
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
yearly.plot(kind="bar", stacked=True, ax=axes[0], color=["#C44E52", "#55A868"])
axes[0].set_title("Class Count by Year (Train)")
axes[0].legend(["DOWN", "UP"])
yearly_pct.plot(kind="bar", stacked=True, ax=axes[1], color=["#C44E52", "#55A868"])
axes[1].set_title("Class Proportion by Year (Train)")
axes[1].axhline(0.5, color="black", ls="--", alpha=0.5)
plt.tight_layout()
save_fig("02_class_by_year.png")

save_table(
    yearly_pct.reset_index().rename(columns={"date": "year", 0: "down_pct", 1: "up_pct"}),
    "class_ratio_by_year.csv",
)


print("\n{}\n 2. TARGET OVER TIME\n{}".format(P, P))
rolling_up = train.set_index("date")["direction"].rolling(90).mean()
streaks = []
current_value = int(train["direction"].iloc[0])
current_len = 1
for value in train["direction"].iloc[1:]:
    value = int(value)
    if value == current_value:
        current_len += 1
    else:
        streaks.append({"direction": current_value, "length": current_len})
        current_value = value
        current_len = 1
streaks.append({"direction": current_value, "length": current_len})
streak_df = pd.DataFrame(streaks)
monthly = train.set_index("date").resample("M")["direction"].value_counts().unstack(fill_value=0)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
axes[0].plot(rolling_up.index, rolling_up.values, color="#4C72B0", lw=1.4)
axes[0].axhline(0.5, color="red", ls="--", alpha=0.6)
axes[0].set_ylabel("P(UP)")
axes[0].set_title("Rolling 90-day UP Proportion (Train)")
for label, date_str in EVENTS:
    axes[0].axvline(pd.Timestamp(date_str), color="orange", ls=":", alpha=0.6)
    axes[0].text(pd.Timestamp(date_str), 0.75, label, fontsize=8, rotation=90, va="top")

axes[1].bar(monthly.index, monthly.get(0, 0), width=20, color="#C44E52", label="DOWN", alpha=0.8)
axes[1].bar(monthly.index, monthly.get(1, 0), width=20, bottom=monthly.get(0, 0), color="#55A868", label="UP", alpha=0.8)
axes[1].set_title("Monthly Class Counts (Train)")
axes[1].legend(fontsize=8)

for cls, color, label in [(0, "#C44E52", "Down streaks"), (1, "#55A868", "Up streaks")]:
    subset = streak_df.loc[streak_df["direction"] == cls, "length"]
    axes[2].hist(subset, bins=range(1, int(streak_df["length"].max()) + 2), alpha=0.55, color=color, label=label)
axes[2].set_title("Streak Length Distribution (Train)")
axes[2].set_xlabel("Consecutive days")
axes[2].legend(fontsize=8)
plt.tight_layout()
save_fig("03_target_over_time.png")
save_table(streak_df, "streak_distribution.csv")


print("\n{}\n 3. FEATURE DISTRIBUTIONS BY CLASS\n{}".format(P, P))
nc = 6
nr = (len(NUM) + nc - 1) // nc
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.5))
axes = np.array(axes).flatten()
ks_rows = []
mw_rows = []
cohen_rows = []
for i, feature in enumerate(NUM):
    ax = axes[i]
    up = train.loc[train["direction"] == 1, feature].dropna()
    down = train.loc[train["direction"] == 0, feature].dropna()
    ax.hist(down, bins=40, alpha=0.5, density=True, color="#C44E52", label="DOWN")
    ax.hist(up, bins=40, alpha=0.5, density=True, color="#55A868", label="UP")
    ks_stat, ks_p = safe_ks(up, down)
    mw_stat, mw_p = safe_mwu(up, down)
    d = safe_cohens_d(up, down)
    ax.set_title("{}\nKS p={:.2e}".format(feature, ks_p if not np.isnan(ks_p) else np.nan), fontsize=6, fontweight="bold")
    ax.tick_params(labelsize=5)
    if i == 0:
        ax.legend(fontsize=5)
    ks_rows.append({"feature": feature, "ks_stat": ks_stat, "ks_p": ks_p, "sig": ks_p < 0.05 if not np.isnan(ks_p) else False})
    mw_rows.append({"feature": feature, "mw_u": mw_stat, "mw_p": mw_p, "sig": mw_p < 0.05 if not np.isnan(mw_p) else False})
    cohen_rows.append({"feature": feature, "cohens_d": d, "abs_d": abs(d) if not np.isnan(d) else np.nan})
for j in range(len(NUM), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Feature Distributions: UP vs DOWN (Train)", fontsize=13, y=1.01)
plt.tight_layout()
save_fig("04_dist_by_class_all.png")

ks_df = pd.DataFrame(ks_rows).sort_values("ks_stat", ascending=False)
mw_df = pd.DataFrame(mw_rows).sort_values("mw_p", ascending=True)
edf = pd.DataFrame(cohen_rows).sort_values("abs_d", ascending=False)
save_table(ks_df, "feature_separability_ks.csv")
save_table(mw_df, "feature_separability_mannwhitney.csv")
save_table(edf, "feature_effect_size_cohens_d.csv")

print("  Features significantly different between UP/DOWN (KS): {}/{}".format(int(ks_df["sig"].sum()), len(NUM)))
print("  Features significantly different between UP/DOWN (MWU): {}/{}".format(int(mw_df["sig"].sum()), len(NUM)))

fig, ax = plt.subplots(figsize=(10, min(18, 0.35 * len(edf) + 2)))
colors = ["#C44E52" if x < 0 else "#55A868" for x in edf["cohens_d"].fillna(0)]
ax.barh(range(len(edf)), edf["cohens_d"].fillna(0), color=colors, height=0.8)
ax.set_yticks(range(len(edf)))
ax.set_yticklabels(edf["feature"], fontsize=6)
ax.invert_yaxis()
ax.axvline(0, color="black", lw=0.5)
ax.axvline(0.2, color="green", ls=":", alpha=0.4)
ax.axvline(-0.2, color="green", ls=":", alpha=0.4)
ax.set_title("Cohen's d: UP vs DOWN")
plt.tight_layout()
save_fig("05_cohens_d.png")


print("\n{}\n 4. POINT-BISERIAL + MUTUAL INFORMATION\n{}".format(P, P))
pb_rows = []
for feature in NUM:
    rpb, pval = safe_pointbiserial(train["direction"], train[feature])
    pb_rows.append({"feature": feature, "rpb": rpb, "p": pval, "abs_rpb": abs(rpb) if not np.isnan(rpb) else np.nan})
pb_df = pd.DataFrame(pb_rows).sort_values("abs_rpb", ascending=False)

mi_input = train[NUM].copy()
mi_input = mi_input.fillna(mi_input.median(numeric_only=True)).fillna(0.0)
mi = mutual_info_classif(mi_input, train["direction"], random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({"feature": NUM, "MI": mi}).sort_values("MI", ascending=False)

save_table(pb_df, "feature_target_pointbiserial.csv")
save_table(mi_df, "mutual_information.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, max(7, 0.28 * len(NUM))))

colors = ["#C44E52" if x < 0 else "#55A868" for x in pb_df["rpb"].fillna(0)]
axes[0].barh(range(len(pb_df)), pb_df["rpb"].fillna(0), color=colors, height=0.8)
axes[0].set_yticks(range(len(pb_df)))
axes[0].set_yticklabels(pb_df["feature"], fontsize=6)
axes[0].invert_yaxis()
axes[0].axvline(0, color="black", lw=0.5)
axes[0].set_title("Point-Biserial Correlation")

axes[1].barh(range(len(mi_df)), mi_df["MI"], color="#4C72B0", height=0.8)
axes[1].set_yticks(range(len(mi_df)))
axes[1].set_yticklabels(mi_df["feature"], fontsize=6)
axes[1].invert_yaxis()
axes[1].set_title("Mutual Information")

plt.tight_layout()
save_fig("06_signal_scores.png")


print("\n{}\n 4B. TIME-SERIES + MULTICOLLINEARITY CHECKS\n{}".format(P, P))
adf_candidates = []
for candidate in [TARGET_RET, "oil_return", "yield_spread", "inventory_change_pct", "gdelt_tone_30d", "conflict_event_count_log1p"]:
    if candidate in df.columns and candidate not in adf_candidates:
        adf_candidates.append(candidate)
adf_rows = []
for feature in adf_candidates:
    adf_stat, adf_p = safe_adf(train[feature])
    adf_rows.append({
        "feature": feature,
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "stationary": bool(adf_p < 0.05) if not np.isnan(adf_p) else np.nan,
    })
adf_df = pd.DataFrame(adf_rows)
save_table(adf_df, "adf_test_results.csv")

target_series = train.set_index("date")[TARGET_RET].dropna()
if len(target_series) >= 30:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if HAS_STATSMODELS:
        plot_acf(target_series, lags=30, ax=axes[0], title="ACF - {}".format(TARGET_RET))
        plot_pacf(target_series, lags=30, ax=axes[1], title="PACF - {}".format(TARGET_RET), method="ywm")
    else:
        lags = np.arange(1, 31)
        acf_vals = [target_series.autocorr(lag=int(lag)) for lag in lags]
        axes[0].bar(lags, acf_vals, color="#4C72B0")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title("ACF Approx - {}".format(TARGET_RET))
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("Autocorr")
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "PACF/ADF skipped:\nstatsmodels not installed",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC"),
        )
    plt.tight_layout()
    save_fig("06b_acf_pacf_target.png")

corr_mat_full = train[NUM].corr(method="spearman")
corr_pairs = []
for i, f1 in enumerate(NUM):
    for j in range(i + 1, len(NUM)):
        f2 = NUM[j]
        corr_pairs.append({"feature_1": f1, "feature_2": f2, "spearman_r": corr_mat_full.iloc[i, j], "abs_r": abs(corr_mat_full.iloc[i, j])})
corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_r", ascending=False)
save_table(corr_pairs_df.head(25), "correlation_top_pairs.csv")

vif_df = safe_vif_table(train[NUM])
save_table(vif_df, "vif_results.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_corr = corr_pairs_df.head(15).iloc[::-1]
axes[0].barh(range(len(top_corr)), top_corr["abs_r"], color="#8172B2")
axes[0].set_yticks(range(len(top_corr)))
axes[0].set_yticklabels(
    ["{} vs {}".format(a, b) for a, b in zip(top_corr["feature_1"], top_corr["feature_2"])],
    fontsize=7,
)
axes[0].set_title("Top Spearman Correlated Feature Pairs")
axes[0].set_xlabel("|r|")

top_vif = vif_df.head(15).iloc[::-1]
axes[1].barh(range(len(top_vif)), top_vif["vif"].fillna(0), color="#937860")
axes[1].set_yticks(range(len(top_vif)))
axes[1].set_yticklabels(top_vif["feature"], fontsize=7)
axes[1].set_title("Top VIF Features")
axes[1].set_xlabel("VIF")
axes[1].axvline(5, color="orange", ls=":", alpha=0.6)
axes[1].axvline(10, color="red", ls=":", alpha=0.6)
plt.tight_layout()
save_fig("06c_multicollinearity.png")

calendar_rows = []
if "day_of_week_sin" in train.columns:
    train_calendar = train.copy()
    train_calendar["year"] = train_calendar["date"].dt.year
    train_calendar["month_num"] = train_calendar["date"].dt.month
    train_calendar["day_num"] = train_calendar["date"].dt.dayofweek
    calendar_rows = (
        train_calendar.groupby(["year", "month_num"])["direction"].mean().unstack().fillna(np.nan)
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(calendar_rows, center=0.5, cmap="RdYlGn", linewidths=0.3, annot=True, fmt=".2f", ax=axes[0], cbar=False, annot_kws={"fontsize": 7})
    axes[0].set_title("Mean UP Ratio by Year x Month")
    dow = train_calendar.groupby("day_num")["direction"].mean()
    axes[1].bar(dow.index.astype(str), dow.values, color="#4C72B0")
    axes[1].axhline(train_calendar["direction"].mean(), color="red", ls="--", alpha=0.6)
    axes[1].set_title("Mean UP Ratio by Day of Week")
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("P(UP)")
    plt.tight_layout()
    save_fig("06d_calendar_patterns.png")


print("\n{}\n 5. TRAIN/TEST SHIFT + STABILITY-AWARE RANKING\n{}".format(P, P))
shift_rows = []
for feature in NUM:
    ks_stat, ks_p = safe_ks(train[feature], test[feature])
    shift_rows.append({"feature": feature, "train_test_ks": ks_stat, "train_test_ks_p": ks_p})
shift_df = pd.DataFrame(shift_rows).sort_values("train_test_ks", ascending=False)
save_table(shift_df, "distribution_shift_ks_test.csv")

combined = pd.DataFrame({"feature": NUM})
combined = combined.merge(group_df, on="feature", how="left")
combined = combined.merge(pb_df[["feature", "abs_rpb", "rpb", "p"]], on="feature", how="left")
combined = combined.merge(mi_df[["feature", "MI"]], on="feature", how="left")
combined = combined.merge(ks_df[["feature", "ks_stat", "ks_p", "sig"]], on="feature", how="left")
combined = combined.merge(edf[["feature", "cohens_d", "abs_d"]], on="feature", how="left")
combined = combined.merge(shift_df, on="feature", how="left")

combined["signal_score"] = (
    normalize(combined["abs_rpb"]).fillna(0.0)
    + normalize(combined["MI"]).fillna(0.0)
    + normalize(combined["ks_stat"]).fillna(0.0)
    + normalize(combined["abs_d"]).fillna(0.0)
) / 4.0
combined["stability_score"] = 1.0 - normalize(combined["train_test_ks"]).fillna(0.0)
combined["research_score"] = 0.7 * combined["signal_score"] + 0.3 * combined["stability_score"]
combined["risk"] = combined["feature"].map(leak_df.set_index("feature")["risk"])
combined["risk_reason"] = combined["feature"].map(leak_df.set_index("feature")["why"])
combined.sort_values(["research_score", "signal_score"], ascending=False, inplace=True)
combined.reset_index(drop=True, inplace=True)
combined.to_csv(OUT / out_name("feature_ranking_clf.csv"), index=False)
save_table(combined, "feature_ranking_clf_full.csv")

print("  Top signal feature: {}".format(combined.iloc[0]["feature"]))
print("  Most shifted feature: {}".format(shift_df.iloc[0]["feature"]))

top_shift_plot = shift_df.head(20)
top_rank_plot = combined.head(20)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].barh(range(len(top_rank_plot)), top_rank_plot["research_score"], color="#4C72B0")
axes[0].set_yticks(range(len(top_rank_plot)))
axes[0].set_yticklabels(top_rank_plot["feature"], fontsize=7)
axes[0].invert_yaxis()
axes[0].set_title("Top 20 Features by Research Score")
axes[0].set_xlabel("research_score")

axes[1].barh(range(len(top_shift_plot)), top_shift_plot["train_test_ks"], color="#DD8452")
axes[1].set_yticks(range(len(top_shift_plot)))
axes[1].set_yticklabels(top_shift_plot["feature"], fontsize=7)
axes[1].invert_yaxis()
axes[1].set_title("Top 20 Train/Test Shift (KS)")
axes[1].set_xlabel("train_test_ks")

plt.tight_layout()
save_fig("07_ranking_and_shift.png")


print("\n{}\n 6. TOP FEATURE VISUALS\n{}".format(P, P))
top20 = combined.head(20)["feature"].tolist()
fig, axes = plt.subplots(4, 5, figsize=(22, 16))
axes = np.array(axes).flatten()
for i, feature in enumerate(top20):
    ax = axes[i]
    data = [
        train.loc[train["direction"] == 0, feature].dropna(),
        train.loc[train["direction"] == 1, feature].dropna(),
    ]
    bp = ax.boxplot(
        data,
        labels=["DOWN", "UP"],
        patch_artist=True,
        widths=0.6,
        boxprops=dict(alpha=0.7),
        medianprops=dict(color="black", lw=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    bp["boxes"][0].set_facecolor("#C44E52")
    bp["boxes"][1].set_facecolor("#55A868")
    ax.set_title("{}\nscore={:.3f}".format(feature, combined.set_index("feature").loc[feature, "research_score"]), fontsize=8)
for j in range(len(top20), len(axes)):
    axes[j].axis("off")
fig.suptitle("Boxplot by Class - Top 20 Features (Train)", fontsize=14)
plt.tight_layout()
save_fig("08_boxplot_top20.png")

top12 = combined.head(12)["feature"].tolist()
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = np.array(axes).flatten()
for i, feature in enumerate(top12):
    ax = axes[i]
    parts = ax.violinplot(
        [
            train.loc[train["direction"] == 0, feature].dropna(),
            train.loc[train["direction"] == 1, feature].dropna(),
        ],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
    )
    for j, body in enumerate(parts["bodies"]):
        body.set_facecolor("#C44E52" if j == 0 else "#55A868")
        body.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DOWN", "UP"])
    ax.set_title(feature, fontsize=9)
fig.suptitle("Violin Plot by Class - Top 12 Features (Train)", fontsize=14)
plt.tight_layout()
save_fig("09_violin_top12.png")

top15 = combined.head(15)["feature"].tolist()
corr_mat = train[top15].corr(method="spearman")
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=0.5, annot_kws={"fontsize": 6}, ax=ax)
ax.set_title("Spearman Correlation - Top 15 Features")
ax.tick_params(labelsize=7)
plt.tight_layout()
save_fig("10_corr_top15.png")

pairs = [(top12[i], top12[j]) for i in range(min(4, len(top12))) for j in range(i + 1, min(4, len(top12)))]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = np.array(axes).flatten()
for i, (f1, f2) in enumerate(pairs[:6]):
    ax = axes[i]
    down_mask = train["direction"] == 0
    up_mask = train["direction"] == 1
    ax.scatter(train.loc[down_mask, f1], train.loc[down_mask, f2], s=4, alpha=0.18, color="#C44E52", label="DOWN")
    ax.scatter(train.loc[up_mask, f1], train.loc[up_mask, f2], s=4, alpha=0.18, color="#55A868", label="UP")
    ax.set_xlabel(f1, fontsize=7)
    ax.set_ylabel(f2, fontsize=7)
    ax.set_title("{} vs {}".format(f1, f2), fontsize=8)
    if i == 0:
        ax.legend(fontsize=6, markerscale=3)
for j in range(len(pairs[:6]), len(axes)):
    axes[j].axis("off")
fig.suptitle("Feature Pair Scatter - Top Feature Pairs", fontsize=14)
plt.tight_layout()
save_fig("11_scatter_pairs.png")


print("\n{}\n 7. GROUP-WISE FEATURE PLOTS\n{}".format(P, P))
for group_name, feats in grouped_features(NUM).items():
    feats = [f for f in feats if f in combined["feature"].tolist()]
    feats = list(combined[combined["feature"].isin(feats)].head(8)["feature"])
    if not feats:
        continue
    ncols = min(4, len(feats))
    nrows = int(np.ceil(len(feats) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else np.array([axes])
    for i, feature in enumerate(feats):
        ax = axes[i]
        train_s = train[feature].dropna()
        test_s = test[feature].dropna()
        ax.hist(train_s, bins=35, density=True, alpha=0.45, color="#4C72B0", label="Train")
        ax.hist(test_s, bins=35, density=True, alpha=0.45, color="#DD8452", label="Test")
        ks_stat, ks_p = safe_ks(train_s, test_s)
        ax.set_title("{}\nshift KS={:.3f}".format(feature, ks_stat if not np.isnan(ks_stat) else 0.0), fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")
    fig.suptitle("{} Group - Train vs Test Distributions".format(group_name), fontsize=13)
    plt.tight_layout()
    save_fig("12_group_{}.png".format(group_name.lower()))


print("\n{}\n 8. TRAIN vs TEST SHIFT - PER CLASS\n{}".format(P, P))
class_shift_rows = []
for feature in combined.head(20)["feature"]:
    up_ks, up_p = safe_ks(train.loc[train["direction"] == 1, feature], test.loc[test["direction"] == 1, feature])
    dn_ks, dn_p = safe_ks(train.loc[train["direction"] == 0, feature], test.loc[test["direction"] == 0, feature])
    class_shift_rows.append({
        "feature": feature,
        "up_ks": up_ks,
        "up_p": up_p,
        "down_ks": dn_ks,
        "down_p": dn_p,
    })
class_shift_df = pd.DataFrame(class_shift_rows).sort_values(["up_ks", "down_ks"], ascending=False)
save_table(class_shift_df, "distribution_shift_by_class.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].barh(range(len(class_shift_df)), class_shift_df["up_ks"], color="#55A868")
axes[0].set_yticks(range(len(class_shift_df)))
axes[0].set_yticklabels(class_shift_df["feature"], fontsize=7)
axes[0].invert_yaxis()
axes[0].set_title("Train/Test Shift for UP Class")
axes[0].set_xlabel("KS stat")

axes[1].barh(range(len(class_shift_df)), class_shift_df["down_ks"], color="#C44E52")
axes[1].set_yticks(range(len(class_shift_df)))
axes[1].set_yticklabels(class_shift_df["feature"], fontsize=7)
axes[1].invert_yaxis()
axes[1].set_title("Train/Test Shift for DOWN Class")
axes[1].set_xlabel("KS stat")
plt.tight_layout()
save_fig("13_class_shift.png")


sig_ks = int(ks_df["sig"].sum())
sig_mwu = int(mw_df["sig"].sum())
sig_rpb = int((pb_df["p"] < 0.05).sum())
low_risk = int((combined["risk"] == "low").sum())
med_risk = int((combined["risk"] == "medium").sum())
high_risk = int((combined["risk"] == "high").sum())
critical_vif = int((vif_df["vif"] > 10).sum()) if not vif_df.empty else 0
warning_vif = int(((vif_df["vif"] > 5) & (vif_df["vif"] <= 10)).sum()) if not vif_df.empty else 0
stationary_count = int(adf_df["stationary"].sum()) if not adf_df.empty else 0

recommended_keep = combined[(combined["risk"] == "low")].head(10)[["feature", "group", "research_score", "train_test_ks"]]
recommended_caution = combined[(combined["train_test_ks"] > combined["train_test_ks"].median()) | (combined["risk"] == "medium")].head(10)[["feature", "group", "research_score", "train_test_ks", "risk"]]
recommended_drop = combined[combined["risk"] == "high"][["feature", "group", "risk_reason"]]

summary = """
# EDA Summary

## Run
- run label: {run_label}
- dataset: `{dataset}`
- split column: `{split_col}`
- train rows: {train_rows}
- test rows: {test_rows}

## Class Balance
- train: UP={train_up:.1%}, DOWN={train_down:.1%}
- test: UP={test_up:.1%}, DOWN={test_down:.1%}

## Signal Summary
- KS significant features: {sig_ks}/{n_features}
- Mann-Whitney significant features: {sig_mwu}/{n_features}
- Point-biserial significant features: {sig_rpb}/{n_features}
- Cohen's d > 0.2: {small_d}/{n_features}

## Time-Series / Collinearity
- stationary series in ADF sample: {stationary_count}/{adf_total}
- VIF > 10: {critical_vif}
- VIF 5-10: {warning_vif}

## Leakage Risk
- low risk features: {low_risk}
- medium risk features: {med_risk}
- high risk features: {high_risk}

## Top 10 Features By Research Score
{top10_table}

## Recommended Keep
{keep_table}

## Use With Caution
{caution_table}

## High-Risk Features
{drop_table}

## Notes
- `signal_score` measures class separability on the train period.
- `stability_score` penalizes features that shift strongly from train to test.
- `research_score = 0.7 * signal_score + 0.3 * stability_score`.
- This EDA is forward-target aware: the split uses `oil_return_fwd1_date` when available.
- Same-day market returns are marked medium risk because they are only valid for end-of-day T -> T+1 forecasting.
""".strip().format(
    run_label=RUN_LABEL,
    dataset=DATA,
    split_col=split_col,
    train_rows=len(train),
    test_rows=len(test),
    train_up=train["direction"].mean(),
    train_down=1.0 - train["direction"].mean(),
    test_up=test["direction"].mean(),
    test_down=1.0 - test["direction"].mean(),
    sig_ks=sig_ks,
    sig_mwu=sig_mwu,
    sig_rpb=sig_rpb,
    n_features=len(NUM),
    small_d=int((edf["abs_d"] > 0.2).sum()),
    stationary_count=stationary_count,
    adf_total=len(adf_df),
    critical_vif=critical_vif,
    warning_vif=warning_vif,
    low_risk=low_risk,
    med_risk=med_risk,
    high_risk=high_risk,
    top10_table=report_table(combined, ["feature", "group", "signal_score", "stability_score", "research_score", "risk"], max_rows=10),
    keep_table=report_table(recommended_keep) if not recommended_keep.empty else "None",
    caution_table=report_table(recommended_caution) if not recommended_caution.empty else "None",
    drop_table=report_table(recommended_drop) if not recommended_drop.empty else "None",
)
save_report(summary, "eda_summary.md")

print("\n{}\n SUMMARY\n{}".format(P, P))
print(
    """
  RUN LABEL: {run_label}
  TARGET: direction = ({target_ret} > 0)
  SPLIT: {split_col} < {split_date} => train, else test

  CLASS BALANCE:
    Train: UP={train_up:.1%} DOWN={train_down:.1%}
    Test:  UP={test_up:.1%} DOWN={test_down:.1%}

  SIGNAL:
    KS significant:           {sig_ks}/{n_features}
    Mann-Whitney significant: {sig_mwu}/{n_features}
    Point-biserial significant: {sig_rpb}/{n_features}
    Cohen's d > 0.2:          {small_d}/{n_features}

  TIME-SERIES / COLLINEARITY:
    Stationary ADF sample:    {stationary_count}/{adf_total}
    VIF > 10:                 {critical_vif}
    VIF 5-10:                 {warning_vif}

  TOP 5 RESEARCH FEATURES:
    1. {f1:<30} score={s1:.4f}
    2. {f2:<30} score={s2:.4f}
    3. {f3:<30} score={s3:.4f}
    4. {f4:<30} score={s4:.4f}
    5. {f5:<30} score={s5:.4f}

  OUTPUTS:
    figures: {fig_count} png
    tables:  {tbl_count} csv
    reports: {rep_count} md
""".format(
        run_label=RUN_LABEL,
        target_ret=TARGET_RET,
        split_col=split_col,
        split_date=SPLIT.date(),
        train_up=train["direction"].mean(),
        train_down=1.0 - train["direction"].mean(),
        test_up=test["direction"].mean(),
        test_down=1.0 - test["direction"].mean(),
        sig_ks=sig_ks,
        sig_mwu=sig_mwu,
        sig_rpb=sig_rpb,
        small_d=int((edf["abs_d"] > 0.2).sum()),
        stationary_count=stationary_count,
        adf_total=len(adf_df),
        critical_vif=critical_vif,
        warning_vif=warning_vif,
        n_features=len(NUM),
        f1=combined.iloc[0]["feature"],
        s1=combined.iloc[0]["research_score"],
        f2=combined.iloc[1]["feature"],
        s2=combined.iloc[1]["research_score"],
        f3=combined.iloc[2]["feature"],
        s3=combined.iloc[2]["research_score"],
        f4=combined.iloc[3]["feature"],
        s4=combined.iloc[3]["research_score"],
        f5=combined.iloc[4]["feature"],
        s5=combined.iloc[4]["research_score"],
        fig_count=len(list(OUT.glob("*.png"))),
        tbl_count=len(list(TABLE_DIR.glob("*.csv"))),
        rep_count=len(list(REPORT_DIR.glob("*.md"))),
    )
)

print("{}\n DONE\n{}".format(P, P))
