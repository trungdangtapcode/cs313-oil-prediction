"""
Generate presentation-quality plots for oil price classification EDA.
Output: plot_present/
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.feature_selection import mutual_info_classif

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
OUT      = BASE / "plot_present"; OUT.mkdir(exist_ok=True)
FIG_DIR  = BASE / "figures"
TBL_DIR  = BASE / "tables"
DATA     = BASE / "data/processed/dataset_final.csv"
TRAIN_CUT = pd.Timestamp("2023-01-01")

SLIDE_DPI = 150
SLIDE_W, SLIDE_H = 14, 7        # widescreen slide ratio

# Palette
C_UP   = "#2196F3"   # blue
C_DN   = "#EF5350"   # red
C_LINE = "#212121"
C_EVNT = "#FF9800"   # orange for event lines
C_SPLIT= "#4CAF50"   # green for train/test split

# Each event: (label, date, y_fraction_offset)
# y_frac controls vertical placement so close events don't overlap
EVENTS = [
    ("COVID crash (Mar 2020)", "2020-03-09", 0.03),
    ("Ukraine war (Feb 2022)", "2022-02-24", 0.03),
    ("Fed hike (Mar 2022)",    "2022-03-16", 0.30),   # staggered up
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA",
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
})

def save(name):
    p = OUT / name
    plt.savefig(p, dpi=SLIDE_DPI, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  ✓ {name}")

def annotate_events(ax, ymin=None, ymax=None):
    ylim = ax.get_ylim()
    y0 = ymin if ymin is not None else ylim[0]
    y1 = ymax if ymax is not None else ylim[1]
    for label, dt, yfrac in EVENTS:
        x = pd.Timestamp(dt)
        ax.axvline(x, color=C_EVNT, ls="--", lw=1.3, alpha=0.8, zorder=2)
        ypos = y0 + (y1 - y0) * yfrac
        ax.text(x, ypos, f" {label}",
                fontsize=8, color=C_EVNT, va="bottom", ha="left", style="italic",
                bbox=dict(facecolor="white", alpha=0.55, linewidth=0, pad=1))

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["direction"] = (df["oil_return"] > 0).astype(int)
train = df[df["date"] < TRAIN_CUT].copy().reset_index(drop=True)
test  = df[df["date"] >= TRAIN_CUT].copy().reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 01 — Oil Price Returns Over Time (timeline overview)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[01] timeline_overview")
fig, axes = plt.subplots(2, 1, figsize=(SLIDE_W, SLIDE_H+1), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

# Returns — vectorized: plot all Up bars then all Down bars (much faster than per-row loop)
up_mask = df["direction"] == 1
axes[0].bar(df.loc[up_mask, "date"],  df.loc[up_mask,  "oil_return"],
            color=C_UP, alpha=0.55, width=1.5, label="Up (direction=1)")
axes[0].bar(df.loc[~up_mask, "date"], df.loc[~up_mask, "oil_return"],
            color=C_DN, alpha=0.55, width=1.5, label="Down (direction=0)")
axes[0].axhline(0, color=C_LINE, lw=0.8)
axes[0].axvline(TRAIN_CUT, color=C_SPLIT, lw=2.5, ls="-", zorder=5)
axes[0].set_ylabel("Daily Return", fontsize=12)
axes[0].set_title("Brent Oil Daily Returns 2015–2026  (Blue = Up, Red = Down)",
                  fontsize=13, fontweight="bold", pad=8)
annotate_events(axes[0])
split_line = plt.Line2D([0], [0], color=C_SPLIT, lw=2.5, label="Train/Test split (2023-01-01)")
handles, labels = axes[0].get_legend_handles_labels()
handles.append(split_line); labels.append("Train/Test split (2023-01-01)")
axes[0].legend(handles=handles, labels=labels, loc="upper left", fontsize=9,
               framealpha=0.85)

# Volatility
axes[1].plot(df["date"], df["oil_volatility_7d"], lw=1.2, color="#7B1FA2")
axes[1].fill_between(df["date"], df["oil_volatility_7d"], alpha=0.2, color="#7B1FA2")
axes[1].axvline(TRAIN_CUT, color=C_SPLIT, lw=2.5, ls="-", zorder=5)
axes[1].set_ylabel("7d Volatility", fontsize=11)
axes[1].set_xlabel("Date")
annotate_events(axes[1])
axes[1].set_title("7-day Rolling Volatility", fontsize=12)

fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, hspace=0.28)
save("01_oil_returns_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 02 — Target Definition & Class Balance
# ══════════════════════════════════════════════════════════════════════════════
print("[02] target_class_balance")
ret = train["oil_return"]
dir_ = train["direction"]
cc = dir_.value_counts().sort_index()
cr = dir_.value_counts(normalize=True).sort_index()
skew, kurt = ret.skew(), ret.kurtosis()

fig = plt.figure(figsize=(SLIDE_W, 5.5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# Sub1: histogram + KDE
ax1 = fig.add_subplot(gs[0, 0:2])
bins = np.linspace(ret.min(), ret.max(), 80)
# Down bars
mask_dn = dir_ == 0
ax1.hist(ret[mask_dn], bins=bins, density=True, color=C_DN, alpha=0.55, label="Down (0)")
ax1.hist(ret[~mask_dn], bins=bins, density=True, color=C_UP, alpha=0.55, label="Up (1)")
xr = np.linspace(ret.min(), ret.max(), 300)
ax1.plot(xr, stats.gaussian_kde(ret)(xr), color=C_LINE, lw=2, label="Overall KDE")
ax1.axvline(0, color="black", ls="--", lw=1.5, label="return = 0")
ax1.set_xlabel("oil_return (daily)")
ax1.set_ylabel("Density")
ax1.set_title(f"Return Distribution by Class\nskewness={skew:.2f}   excess kurtosis={kurt:.1f} (fat tails)", fontsize=12)
ax1.legend(fontsize=9)

# Add annotation box
props = dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", alpha=0.9)
ax1.text(0.98, 0.97,
         f"⚠ Kurtosis={kurt:.0f}\n(vs Normal=3)\n→ Fat tails\n→ Extreme events\n   more frequent",
         transform=ax1.transAxes, fontsize=9, va="top", ha="right", bbox=props)

# Sub2: class bar
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.bar(["Down (0)\nGiảm", "Up (1)\nTăng"], cc.values,
               color=[C_DN, C_UP], width=0.5, edgecolor="white", linewidth=1.5)
for bar, v, pct in zip(bars, cc.values, cr.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{v:,}\n({pct:.1%})", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax2.set_ylim(0, cc.max() * 1.25)
ax2.set_title("Class Balance\n(Train set)", fontsize=12)
ax2.set_ylabel("Count")
ax2.yaxis.grid(True, alpha=0.5)

# Balance label
balance_note = "✅ Balanced\n(~51/49)" if abs(cr[0]-cr[1]) < 0.10 else "⚠️ Imbalanced"
ax2.text(0.5, 0.85, balance_note, transform=ax2.transAxes,
         ha="center", fontsize=11, color="#2E7D32", fontweight="bold",
         bbox=dict(facecolor="#E8F5E9", alpha=0.9, boxstyle="round,pad=0.4"))

fig.suptitle("Target Variable Analysis — direction = (oil_return > 0).astype(int)",
             fontsize=13, fontweight="bold", y=1.01)
save("02_target_class_balance.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 03 — Target Over Time (temporal pattern)
# ══════════════════════════════════════════════════════════════════════════════
print("[03] target_over_time")
fig, axes = plt.subplots(2, 1, figsize=(SLIDE_W, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [1.5, 1]})

# Rolling up-ratio (30d, 90d)
ts = df.set_index("date").sort_index()
roll30 = ts["direction"].rolling(30).mean()
roll90 = ts["direction"].rolling(90).mean()
overall = dir_.mean()

axes[0].fill_between(roll30.index, 0.5, roll30.values,
                     where=roll30.values >= 0.5, alpha=0.25, color=C_UP)
axes[0].fill_between(roll30.index, 0.5, roll30.values,
                     where=roll30.values < 0.5, alpha=0.25, color=C_DN)
axes[0].plot(roll30.index, roll30.values, lw=1.2, color=C_UP, alpha=0.7, label="30d rolling Up-ratio")
axes[0].plot(roll90.index, roll90.values, lw=2.0, color="#1565C0", label="90d rolling Up-ratio")
axes[0].axhline(0.5, color="gray", ls="--", lw=1.2, label="50% line")
axes[0].axhline(overall, color="black", ls="-.", lw=1, alpha=0.6,
                label=f"Train overall = {overall:.1%}")
axes[0].axvline(TRAIN_CUT, color=C_SPLIT, lw=2, label="Train/Test split")

# Regime annotations
regimes = [
    ("Pre-COVID\nbull", "2015-01-01", "2018-06-01"),
    ("COVID\ncrash", "2020-02-01", "2020-06-01"),
    ("Post-COVID\nbull", "2020-06-01", "2022-02-01"),
    ("Ukraine\nhigh-vol", "2022-02-24", "2023-01-01"),
]
ypos = 0.08
for name, s, e in regimes:
    axes[0].axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color="gray")

for label, dt, _yfrac in EVENTS:
    axes[0].axvline(pd.Timestamp(dt), color=C_EVNT, ls=":", lw=1.2, alpha=0.9)

axes[0].set_ylim(0.2, 0.8)
axes[0].set_ylabel("Fraction of Up days")
axes[0].set_title("Market Direction Over Time — Rolling Up-Ratio", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=9, loc="upper right", ncol=3)

# Geopolitical stress
axes[1].plot(df["date"], df["geopolitical_stress_index"], lw=1.2, color="#7B1FA2", label="Geopolitical Stress Index")
axes[1].fill_between(df["date"], df["geopolitical_stress_index"], alpha=0.15, color="#7B1FA2")
axes[1].axvline(TRAIN_CUT, color=C_SPLIT, lw=2)
axes[1].set_ylabel("Geo Stress")
axes[1].set_xlabel("Date")
axes[1].legend(fontsize=9)

plt.tight_layout()
save("03_target_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 04 — Data Quality Summary (compact, slide-friendly)
# ══════════════════════════════════════════════════════════════════════════════
print("[04] data_quality")
num_cols = train.select_dtypes(include=[np.number]).columns
out_rows = []
for c in num_cols:
    s = train[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    n = ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()
    out_rows.append({"feature": c, "n_outliers": n, "pct": n/len(s)*100,
                     "skew": s.skew(), "kurt": s.kurtosis()})
out_df = pd.DataFrame(out_rows).sort_values("n_outliers", ascending=False).head(10)

fig, axes = plt.subplots(1, 3, figsize=(SLIDE_W, 5))

# Panel A: Missing / dup / INF checklist
ax = axes[0]
ax.axis("off")
checks = [
    ("Missing values",    "0",     "✅"),
    ("Duplicate rows",    "0",     "✅"),
    ("Duplicate dates",   "0",     "✅"),
    ("INF values",        "0",     "✅"),
    ("Lag1 consistency",  "OK",    "✅"),
    ("Train rows",        "2,083", "ℹ️"),
    ("Test rows",         "840",   "ℹ️"),
    ("Total features",    "32",    "ℹ️"),
]
ax.set_title("Data Integrity Checklist", fontsize=13, fontweight="bold", pad=10)
for i, (chk, val, icon) in enumerate(checks):
    y = 0.88 - i*0.11
    ax.text(0.0,  y, icon,  transform=ax.transAxes, fontsize=14, va="center")
    ax.text(0.12, y, chk,  transform=ax.transAxes, fontsize=11, va="center")
    ax.text(0.85, y, val,  transform=ax.transAxes, fontsize=11, va="center",
            ha="right", fontweight="bold",
            color="#2E7D32" if icon=="✅" else "#1565C0")

# Panel B: Outlier % bar chart (top features)
ax = axes[1]
top_out = out_df.head(8)
bars = ax.barh(top_out["feature"][::-1], top_out["pct"][::-1],
               color="#EF9A9A", edgecolor="white")
ax.axvline(5, color="gray", ls="--", lw=1, label="5% threshold")
ax.set_xlabel("% IQR outliers"); ax.set_title("Outlier rate per feature\n(financial fat-tails expected)", fontsize=12)
for bar, pct in zip(bars, top_out["pct"][::-1]):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f"{pct:.1f}%", va="center", fontsize=9)
ax.legend(fontsize=9)
note = "⚠️ Outliers here are\nmarket events (COVID, Ukraine)\nNOT data errors"
ax.text(0.97, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#E65100",
        bbox=dict(facecolor="#FFF3E0", boxstyle="round,pad=0.4", alpha=0.9))

# Panel C: Skewness bar chart
ax = axes[2]
skew_df = pd.DataFrame(out_rows).sort_values("skew", key=abs, ascending=False).head(10)
colors = [C_DN if s > 0 else C_UP for s in skew_df["skew"][::-1]]
ax.barh(skew_df["feature"][::-1], skew_df["skew"][::-1], color=colors, edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.axvline(1, color="gray", ls="--", lw=1, alpha=0.7)
ax.axvline(-1, color="gray", ls="--", lw=1, alpha=0.7)
ax.set_xlabel("Skewness")
ax.set_title("Feature skewness\n(|skew|>1 → consider transform)", fontsize=12)

fig.suptitle("Data Quality Overview — Train Set (< 2023-01-01)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("04_data_quality.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 05 — Top Feature Signals vs Target
# ══════════════════════════════════════════════════════════════════════════════
print("[05] feature_signals")
num_feats = [c for c in train.select_dtypes(include=[np.number]).columns
             if c not in ["oil_return","direction"]]

# Compute point-biserial + MI
from scipy.stats import pointbiserialr
ft_rows = []
for c in num_feats:
    s = train[c].fillna(train[c].median())
    r, _ = pointbiserialr(train["direction"], s)
    g0 = s[train["direction"]==0]; g1 = s[train["direction"]==1]
    _, p_mw = mannwhitneyu(g0, g1, alternative="two-sided")
    ft_rows.append({"feature": c, "r": r, "abs_r": abs(r), "sig": p_mw<0.05})
ft_df = pd.DataFrame(ft_rows).sort_values("abs_r", ascending=False)

X = train[num_feats].fillna(0)
y = train["direction"]
mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({"feature": num_feats, "mi": mi}).sort_values("mi", ascending=False)

# TOP 8 by abs_r
top8 = ft_df.head(8)["feature"].tolist()

fig = plt.figure(figsize=(SLIDE_W, SLIDE_H+1))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel A (left, tall): Point-biserial bar chart
ax_pb = fig.add_subplot(gs[:, 0])
top_r = ft_df.head(12)
colors_bar = [C_UP if r > 0 else C_DN for r in top_r["r"][::-1]]
bars = ax_pb.barh(top_r["feature"][::-1], top_r["r"][::-1], color=colors_bar, edgecolor="white")
ax_pb.axvline(0, color="black", lw=1)
# significance dots
for i, (_, row) in enumerate(top_r[::-1].iterrows()):
    if row["sig"]:
        ax_pb.text(row["r"] + (0.002 if row["r"]>=0 else -0.002),
                   11-i, "*", ha="center", va="center", color="black", fontsize=13)
ax_pb.set_xlabel("Point-biserial r\n(* = MW p<0.05)", fontsize=10)
ax_pb.set_title("Feature–Target\nCorrelation (r)\n", fontsize=12, fontweight="bold")
ax_pb.set_xlim(-0.28, 0.28)
note = "Blue → Up signal\nRed → Down signal"
ax_pb.text(0.5, -0.12, note, transform=ax_pb.transAxes,
           ha="center", fontsize=9, color="gray")

# Panel B (right top): MI bar
ax_mi = fig.add_subplot(gs[0, 1:])
top_mi = mi_df.head(10)
ax_mi.barh(top_mi["feature"][::-1], top_mi["mi"][::-1], color="#7B1FA2", alpha=0.75, edgecolor="white")
ax_mi.set_xlabel("Mutual Information score"); ax_mi.set_title("Mutual Information vs direction\n(non-linear signal)", fontsize=12, fontweight="bold")
ax_mi.text(0.97, 0.05,
           "Higher MI → more\npredictive information",
           transform=ax_mi.transAxes, ha="right", fontsize=9, color="#4A148C",
           bbox=dict(facecolor="#F3E5F5", boxstyle="round,pad=0.3", alpha=0.9))

# Panel C (right bottom): boxplot top 4 features
ax_box = fig.add_subplot(gs[1, 1:])
top4 = ft_df.head(4)["feature"].tolist()
positions_all = []
data_all = []
labels_all = []
colors_all = []
spacing = 2.5

for i, feat in enumerate(top4):
    base = i * spacing
    g0 = train.loc[train["direction"]==0, feat].dropna().values
    g1 = train.loc[train["direction"]==1, feat].dropna().values
    # Normalize to z-score for fair comparison
    mn, sd = np.concatenate([g0,g1]).mean(), np.concatenate([g0,g1]).std()
    sd = sd if sd > 0 else 1
    data_all.extend([(g0-mn)/sd, (g1-mn)/sd])
    positions_all.extend([base, base+1])
    short_name = feat.replace("_return","_ret").replace("_lag","_lg")
    labels_all.extend([f"{short_name}\nDown", f"{short_name}\nUp"])
    colors_all.extend([C_DN, C_UP])

bp = ax_box.boxplot(data_all, positions=positions_all, patch_artist=True,
                    widths=0.7, showfliers=False,
                    medianprops=dict(color="black", lw=2))
for patch, col in zip(bp["boxes"], colors_all):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax_box.set_xticks(positions_all)
ax_box.set_xticklabels(labels_all, fontsize=8, rotation=15)
ax_box.set_ylabel("z-score")
ax_box.set_title("Top 4 features — distribution Up vs Down\n(z-scored for comparison)", fontsize=12, fontweight="bold")
ax_box.axhline(0, color="gray", lw=0.8, ls="--")

fig.suptitle("Feature Signals for Classification — Train Set Only",
             fontsize=14, fontweight="bold", y=1.01)
save("05_feature_signals.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 06 — Time-series Properties (ACF + seasonality)
# ══════════════════════════════════════════════════════════════════════════════
print("[06] timeseries_properties")
from statsmodels.tsa.stattools import adfuller

fig, axes = plt.subplots(2, 2, figsize=(SLIDE_W, SLIDE_H))
ret_ts = train.set_index("date")["oil_return"].dropna()

# A: ACF oil_return
plot_acf(ret_ts, lags=25, ax=axes[0,0], title="")
axes[0,0].set_title(
    "ACF — oil_return (train)\n→ Near-zero: market close to random walk",
    fontsize=11,
    fontweight="bold",
    pad=6,
)
axes[0,0].axhline(0, color="black", lw=0.8)
axes[0,0].text(
    0.96, 0.62,
    "ACF ≈ 0 at all lags\n→ Returns are\n   nearly i.i.d.",
    transform=axes[0,0].transAxes,
    ha="right",
    va="bottom",
    fontsize=8.5,
    bbox=dict(facecolor="#FFF9C4", boxstyle="round,pad=0.3", alpha=0.9),
)

# B: ACF volatility
vol_ts = train.set_index("date")["oil_volatility_7d"].dropna()
plot_acf(vol_ts, lags=25, ax=axes[0,1], title="", color=C_DN)
axes[0,1].set_title(
    "ACF — oil_volatility_7d (train)\n→ Strong persistence: volatility clustering",
    fontsize=11,
    fontweight="bold",
    pad=6,
)

# C: Seasonality by month (up-ratio)
monthly = train.groupby("month")["direction"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
bar_colors = [C_UP if v >= 0.5 else C_DN for v in monthly.values]
axes[1,0].bar(range(1,13), monthly.values, color=bar_colors, alpha=0.75, edgecolor="white")
axes[1,0].axhline(0.5, color="gray", ls="--", lw=1.2, label="50% baseline")
axes[1,0].set_xticks(range(1,13)); axes[1,0].set_xticklabels(month_names, fontsize=9)
axes[1,0].set_ylabel("Fraction Up"); axes[1,0].set_ylim(0.35, 0.70)
axes[1,0].set_title("Up-ratio by Month\n→ Mild seasonal pattern", fontsize=11, fontweight="bold")
axes[1,0].legend(fontsize=9)

# D: Up-ratio by day of week
daily = train.groupby("day_of_week")["direction"].mean()
days = ["Mon","Tue","Wed","Thu","Fri"]
bar_colors2 = [C_UP if v >= 0.5 else C_DN for v in daily.values]
axes[1,1].bar(range(5), daily.values, color=bar_colors2, alpha=0.75, edgecolor="white")
axes[1,1].axhline(0.5, color="gray", ls="--", lw=1.2)
axes[1,1].set_xticks(range(5)); axes[1,1].set_xticklabels(days)
axes[1,1].set_ylabel("Fraction Up"); axes[1,1].set_ylim(0.35, 0.70)
axes[1,1].set_title("Up-ratio by Day of Week\n→ Slight mid-week bias", fontsize=11, fontweight="bold")

fig.suptitle(
    "Time-Series Properties — Stationarity & Seasonality (Train Set)",
    fontsize=14,
    fontweight="bold",
    y=0.985,
)
plt.tight_layout(rect=(0, 0, 0.91, 0.92), h_pad=1.8, w_pad=1.2)
fig.text(
    0.975, 0.765,
    "High ACF for 20+ lags\n→ Volatility regime\npersists → useful\nfor model",
    ha="right",
    va="top",
    fontsize=8.5,
    bbox=dict(facecolor="#E8F5E9", boxstyle="round,pad=0.3", alpha=0.9),
)
save("06_timeseries_properties.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 07 — Train/Test Split & Leakage Check
# ══════════════════════════════════════════════════════════════════════════════
print("[07] split_leakage_check")
fig, axes = plt.subplots(1, 3, figsize=(SLIDE_W, 5.5))

# Panel A: Class ratio train vs test
train_up = (train["oil_return"]>0).mean()
test_up  = (test["oil_return"]>0).mean()
categories = ["Train\n(2015–2022)", "Test\n(2023–2026)"]
up_vals    = [train_up, test_up]
dn_vals    = [1-train_up, 1-test_up]
x = np.arange(2)
w = 0.32
bar1 = axes[0].bar(x-w/2, dn_vals, w, label="Down", color=C_DN, alpha=0.75, edgecolor="white")
bar2 = axes[0].bar(x+w/2, up_vals, w, label="Up",   color=C_UP, alpha=0.75, edgecolor="white")
for bar, v in zip(list(bar1)+list(bar2), dn_vals+up_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(categories)
axes[0].set_ylim(0, 0.72); axes[0].set_ylabel("Fraction")
axes[0].set_title("Target Class Ratio\nTrain vs Test", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
drift_pct = abs(train_up - test_up)
color_drift = "#2E7D32" if drift_pct < 0.05 else "#E65100"
axes[0].text(0.5, 0.92, f"Class drift: {drift_pct:.1%}",
             transform=axes[0].transAxes, ha="center", color=color_drift,
             fontweight="bold", fontsize=11,
             bbox=dict(facecolor="#F1F8E9" if drift_pct<0.05 else "#FFF3E0",
                       boxstyle="round,pad=0.3", alpha=0.9))

# Panel B: KDE shift for top macro features
from scipy.stats import ks_2samp
shifted_feats = ["cpi_lag","real_rate","yield_spread","fed_rate_regime"]
colors_kd = ["#1565C0","#C62828","#2E7D32","#6A1B9A"]
for feat, col in zip(shifted_feats, colors_kd):
    sns.kdeplot(train[feat].dropna(), ax=axes[1], color=col, lw=2,
                ls="-", label=f"{feat} (train)")
    sns.kdeplot(test[feat].dropna(),  ax=axes[1], color=col, lw=2,
                ls="--", alpha=0.7)
axes[1].set_title("Distribution Shift\nTrain (solid) vs Test (dashed)\nTop macro features", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Feature value"); axes[1].set_ylabel("Density")
axes[1].legend(fontsize=8, title="Feature")
axes[1].text(0.97, 0.97,
             "⚠️ Macro features shift\nsignificantly train→test\n→ Model must generalize\n   across regimes",
             transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
             color="#BF360C",
             bbox=dict(facecolor="#FBE9E7", boxstyle="round,pad=0.4", alpha=0.9))

# Panel C: Leakage risk table
ax3 = axes[2]; ax3.axis("off")
ax3.set_title("Leakage Risk Assessment", fontsize=12, fontweight="bold", pad=10)
risk_data = [
    ("oil_return [t]",     "🔴 HIGH",   "Is the target source — EXCLUDE"),
    ("direction [t]",      "🔴 HIGH",   "Is the target — EXCLUDE"),
    ("vix_return [t]",     "🟡 MEDIUM", "Same-day close → use lag1"),
    ("sp500_return [t]",   "🟡 MEDIUM", "Same-day close → use lag1"),
    ("usd_return [t]",     "🟡 MEDIUM", "Same-day close → use lag1"),
    ("oil_return_lag1 [t-1]","🟢 LOW",  "Explicit lag — safe"),
    ("vix_lag1 [t-1]",     "🟢 LOW",   "Explicit lag — safe"),
    ("oil_volatility_7d",  "🟢 LOW",   "Backward rolling — safe"),
    ("cpi_lag",            "🟢 LOW",   "Lagged macro — safe"),
    ("geopolitical_stress","🟢 LOW",   "Train-only normalized"),
]
for i, (feat, risk, note) in enumerate(risk_data):
    y = 0.93 - i*0.092
    ax3.text(0.0,  y, risk,  transform=ax3.transAxes, fontsize=10, va="center")
    ax3.text(0.22, y, feat,  transform=ax3.transAxes, fontsize=9,  va="center", style="italic")
    ax3.text(0.55, y, note,  transform=ax3.transAxes, fontsize=8,  va="center", color="gray")

fig.suptitle("Data Split Integrity & Leakage Check",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
save("07_split_leakage_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
plots = sorted(OUT.glob("*.png"))
print(f"\n✅ Done. {len(plots)} plots in {OUT}")
for p in plots:
    print(f"   {p.name}")
