"""
Visualize dữ liệu từ FRED, EIA, GDELT để hiểu structure và ý nghĩa
Chạy script này TRƯỚC khi có API key - dùng sample data mô phỏng đúng format thật.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================
# 📦 Tạo sample data mô phỏng đúng format thật
# ============================================================
dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
n = len(dates)

# --- Giá dầu WTI (Yahoo Finance - cậu đã có) ---
# Mô phỏng: crash COVID 2020, phục hồi 2021, spike 2022 (Ukraine)
oil_base = 60 + np.cumsum(np.random.normal(0, 1.2, n))
oil_base[50:100]   -= 40   # COVID crash Mar-Apr 2020
oil_base[100:300]  += np.linspace(0, 30, 200)  # Phục hồi
oil_base[550:620]  += 40   # Ukraine spike 2022
oil_base = np.clip(oil_base, 20, 130)
oil_price = pd.Series(oil_base, index=dates, name="oil_price")

# --- FRED: Lãi suất Fed ---
fed_rate_vals = np.zeros(n)
fed_rate_vals[:200]  = 0.08   # COVID gần 0
fed_rate_vals[200:300] = 0.08
fed_rate_vals[300:400] = 0.5
fed_rate_vals[400:500] = 2.5   # Tăng mạnh 2022
fed_rate_vals[500:600] = 4.5
fed_rate_vals[600:]  = 5.3   # Peak 2023-2024
fed_rate = pd.Series(fed_rate_vals, index=dates, name="fed_funds_rate")

# --- FRED: Yield spread (chỉ báo recession) ---
spread_vals = 1.5 - np.linspace(0, 3, n) + np.random.normal(0, 0.1, n)
spread_vals[600:] += 0.5  # Cải thiện 2024
yield_spread = pd.Series(spread_vals, index=dates, name="yield_spread")

# --- FRED: CPI ---
cpi_vals = 260 + np.linspace(0, 30, n) + np.random.normal(0, 0.5, n)
cpi_vals[400:600] += np.linspace(0, 20, 200)  # Lạm phát 2022
cpi = pd.Series(cpi_vals, index=dates, name="cpi")

# --- EIA: Tồn kho dầu (Weekly) ---
# Tồn kho thường ngược chiều giá
inv_base = 480 - oil_base * 0.5 + np.random.normal(0, 8, n) + np.cumsum(np.random.normal(0, 0.3, n))
inv_base = np.clip(inv_base, 380, 560)
# Giả lập weekly: chỉ có giá trị mỗi thứ Tư, còn lại NaN → ffill
inventory_raw = pd.Series(np.nan, index=dates, name="crude_inventory")
wed_mask = dates.dayofweek == 2  # Wednesday
inventory_raw[wed_mask] = inv_base[wed_mask]
inventory_ffill = inventory_raw.ffill()

inv_change = inventory_ffill.pct_change(5) * 100
inv_change.name = "inventory_change_pct"

# --- EIA: Sản lượng khai thác ---
prod_vals = 11 + np.linspace(0, 2, n) + np.random.normal(0, 0.2, n)
prod_vals[50:150] -= 1.5  # COVID impact
production = pd.Series(prod_vals, index=dates, name="crude_production")

# --- GDELT: Sentiment truyền thông ---
# Tone âm = tin tức tiêu cực về Trung Đông
gdelt_tone_base = -2 + np.random.normal(0, 1, n)
gdelt_tone_base[550:620] -= 3   # Ukraine war → negative coverage tăng
gdelt_tone_base[50:120]  -= 2   # COVID news
gdelt_tone = pd.Series(gdelt_tone_base, index=dates, name="gdelt_tone")
gdelt_tone_7d = gdelt_tone.rolling(7).mean()
gdelt_tone_7d.name = "gdelt_tone_7d"

# Volume tin tức
gdelt_vol_base = 50 + np.abs(np.random.normal(0, 10, n)) + np.abs(gdelt_tone_base) * 5
gdelt_volume = pd.Series(gdelt_vol_base, index=dates, name="gdelt_volume")

# ============================================================
# 🎨 Vẽ Dashboard
# ============================================================
fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3,
                       left=0.07, right=0.96, top=0.94, bottom=0.04)

COLORS = {
    "oil"     : "#f0b429",
    "fed"     : "#4fc3f7",
    "yield"   : "#ef5350",
    "cpi"     : "#ab47bc",
    "inv"     : "#26a69a",
    "inv_chg" : "#ff7043",
    "prod"    : "#66bb6a",
    "tone"    : "#ec407a",
    "vol"     : "#42a5f5",
    "bg"      : "#1a1d2e",
    "grid"    : "#2a2d3e",
    "text"    : "#e0e0e0",
    "subtext" : "#9e9e9e",
}

def style_ax(ax, title, ylabel="", source_tag=""):
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_color(COLORS["grid"])
    ax.yaxis.label.set_color(COLORS["subtext"])
    ax.set_ylabel(ylabel, fontsize=8, color=COLORS["subtext"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.3, alpha=0.4)
    ax.set_title(title, color=COLORS["text"], fontsize=10, fontweight="bold", pad=8)
    if source_tag:
        ax.text(0.99, 0.02, source_tag, transform=ax.transAxes,
                color=COLORS["subtext"], fontsize=7, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2d3e", alpha=0.8))

# --- Title ---
fig.text(0.5, 0.97, "🛢️  Oil Price Prediction — Dữ liệu bổ sung: FRED · EIA · GDELT",
         ha="center", va="top", fontsize=15, fontweight="bold",
         color=COLORS["text"])
fig.text(0.5, 0.955, "Sample data mô phỏng đúng structure thật · 2020–2024",
         ha="center", va="top", fontsize=9, color=COLORS["subtext"])

# ============================================================
# ROW 0: Giá dầu (reference)
# ============================================================
ax0 = fig.add_subplot(gs[0, :])
ax0.fill_between(dates, oil_price, alpha=0.15, color=COLORS["oil"])
ax0.plot(dates, oil_price, color=COLORS["oil"], linewidth=1.5, label="WTI Oil Price")
ax0.axhline(oil_price.mean(), color=COLORS["oil"], linewidth=0.8, linestyle="--", alpha=0.5)

# Annotations
for dt, label, y_off in [
    ("2020-04-01", "COVID\nCrash", -25),
    ("2022-03-01", "Ukraine\nWar", 15),
]:
    x = pd.Timestamp(dt)
    y = oil_price.asof(x)
    ax0.annotate(label, xy=(x, y), xytext=(x, y + y_off),
                color=COLORS["subtext"], fontsize=7.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["subtext"], lw=0.8))

style_ax(ax0, "📌 Biến mục tiêu — Giá dầu WTI (Yahoo Finance · đã có sẵn)", "USD/barrel", "Yahoo Finance")
ax0.set_ylim(0, oil_price.max() * 1.2)

# ============================================================
# ROW 1: FRED
# ============================================================
ax1a = fig.add_subplot(gs[1, 0])
ax1b = ax1b_twin = ax1a.twinx()

ax1a.fill_between(dates, 0, fed_rate, alpha=0.2, color=COLORS["fed"])
ax1a.plot(dates, fed_rate, color=COLORS["fed"], linewidth=1.5, label="Fed Funds Rate")
ax1a.set_ylabel("Fed Rate (%)", fontsize=8, color=COLORS["fed"])
ax1a.tick_params(axis="y", colors=COLORS["fed"], labelsize=8)

ax1b.plot(dates, yield_spread, color=COLORS["yield"], linewidth=1, alpha=0.8, linestyle="--", label="Yield Spread")
ax1b.axhline(0, color=COLORS["yield"], linewidth=0.5, linestyle=":", alpha=0.5)
ax1b.set_ylabel("Yield Spread 10Y-2Y (%)", fontsize=8, color=COLORS["yield"])
ax1b.tick_params(axis="y", colors=COLORS["yield"], labelsize=8)

style_ax(ax1a, "📊 FRED — Lãi suất Fed & Yield Spread", source_tag="FRED API")
ax1a.set_facecolor(COLORS["bg"])
ax1a.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)

# Legend
lines = [mpatches.Patch(color=COLORS["fed"], label="Fed Funds Rate"),
         mpatches.Patch(color=COLORS["yield"], label="Yield Spread 10Y-2Y (âm = recession risk)")]
ax1a.legend(handles=lines, loc="upper left", fontsize=7,
           facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

# Annotation: Yield spread âm
ax1b.annotate("Yield spread âm\n→ Recession risk\n→ Nhu cầu dầu giảm",
              xy=(pd.Timestamp("2023-06-01"), -1.2),
              xytext=(pd.Timestamp("2022-01-01"), -2.2),
              color=COLORS["yield"], fontsize=7,
              arrowprops=dict(arrowstyle="->", color=COLORS["yield"], lw=0.7))

# ---
ax1c = fig.add_subplot(gs[1, 1])
ax1c.plot(dates, cpi, color=COLORS["cpi"], linewidth=1.5)
ax1c.fill_between(dates, cpi.min(), cpi, alpha=0.1, color=COLORS["cpi"])
style_ax(ax1c, "📊 FRED — CPI (Lạm phát)", "Index", "FRED API")

cpi_grad = cpi.diff().rolling(30).mean()
ax1c_twin = ax1c.twinx()
ax1c_twin.plot(dates, cpi_grad, color="#ffcc80", linewidth=1, alpha=0.7, linestyle="--")
ax1c_twin.set_ylabel("Tốc độ tăng CPI (30d MA)", fontsize=7, color="#ffcc80")
ax1c_twin.tick_params(axis="y", colors="#ffcc80", labelsize=7)
ax1c_twin.axhline(0, color="#ffcc80", linewidth=0.4, linestyle=":", alpha=0.5)

ax1c.text(0.02, 0.92, "CPI tăng → Lạm phát cao\n→ Fed tăng lãi suất\n→ USD mạnh → Dầu giảm",
         transform=ax1c.transAxes, color=COLORS["subtext"], fontsize=7,
         bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))

# ============================================================
# ROW 2: EIA — Tồn kho
# ============================================================
ax2a = fig.add_subplot(gs[2, 0])

# Show raw weekly (có gaps) vs sau ffill
ax2a.scatter(dates[wed_mask], inventory_raw[wed_mask], color=COLORS["inv"],
            s=8, zorder=5, label="Điểm đo thực tế (mỗi thứ Tư)")
ax2a.plot(dates, inventory_ffill, color=COLORS["inv"], linewidth=1, alpha=0.5,
         linestyle="--", label="Sau forward-fill (daily)")

style_ax(ax2a, "🏭 EIA — Tồn kho dầu thô Mỹ (Weekly → Daily ffill)", "Triệu thùng", "EIA API V2")

ax2a.legend(loc="upper right", fontsize=7,
           facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
ax2a.text(0.02, 0.05,
         "⚠️  Raw data chỉ có mỗi thứ Tư\nGiữa tuần dùng ffill\n→ Đây là cleaning skill cần làm!",
         transform=ax2a.transAxes, color=COLORS["subtext"], fontsize=7,
         bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))

# ---
ax2b = fig.add_subplot(gs[2, 1])
colors_bar = [COLORS["inv"] if v < 0 else COLORS["inv_chg"] for v in inv_change.fillna(0)]
ax2b.bar(dates, inv_change, color=colors_bar, alpha=0.7, width=1)
ax2b.axhline(0, color="white", linewidth=0.5, alpha=0.3)
style_ax(ax2b, "🏭 EIA — % Thay đổi tồn kho so với tuần trước", "%", "EIA API V2")

ax2b.text(0.02, 0.92,
         "Tồn kho tăng (cam) → Cung dư\n→ Giá dầu có xu hướng GIẢM\nTồn kho giảm (xanh) → Cung thiếu\n→ Giá dầu có xu hướng TĂNG",
         transform=ax2b.transAxes, color=COLORS["subtext"], fontsize=7,
         bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))

# ============================================================
# ROW 3: EIA — Sản lượng
# ============================================================
ax3a = fig.add_subplot(gs[3, 0])
ax3a.plot(dates, production, color=COLORS["prod"], linewidth=1.5)
ax3a.fill_between(dates, production.min() - 0.5, production, alpha=0.15, color=COLORS["prod"])
style_ax(ax3a, "🏭 EIA — Sản lượng khai thác dầu Mỹ", "Triệu thùng/ngày", "EIA API V2")
ax3a.annotate("COVID:\nKhai thác\ngiảm mạnh",
             xy=(pd.Timestamp("2020-06-01"), 10.2),
             xytext=(pd.Timestamp("2020-10-01"), 9.5),
             color=COLORS["subtext"], fontsize=7,
             arrowprops=dict(arrowstyle="->", color=COLORS["subtext"], lw=0.7))

# ---
ax3b = fig.add_subplot(gs[3, 1])

# Correlation plot: inventory change vs oil return
oil_return = oil_price.pct_change() * 100
scatter_x = inv_change.dropna()
scatter_y = oil_return.reindex(scatter_x.index).dropna()
scatter_x = scatter_x.reindex(scatter_y.index)

sc = ax3b.scatter(scatter_x, scatter_y, c=scatter_y, cmap="RdYlGn",
                 alpha=0.4, s=8)
ax3b.axhline(0, color="white", linewidth=0.3, alpha=0.3)
ax3b.axvline(0, color="white", linewidth=0.3, alpha=0.3)

# Trend line
z = np.polyfit(scatter_x.fillna(0), scatter_y.fillna(0), 1)
p = np.poly1d(z)
x_line = np.linspace(scatter_x.min(), scatter_x.max(), 100)
ax3b.plot(x_line, p(x_line), color="#ffcc80", linewidth=1.5, linestyle="--", alpha=0.8)

corr = scatter_x.corr(scatter_y)
style_ax(ax3b, f"📈 Correlation: Tồn kho vs Giá dầu (r = {corr:.2f})", source_tag="EIA + Yahoo")
ax3b.set_xlabel("% thay đổi tồn kho", fontsize=8, color=COLORS["subtext"])
ax3b.set_ylabel("% thay đổi giá dầu", fontsize=8, color=COLORS["subtext"])
ax3b.text(0.02, 0.92, "Nghịch chiều rõ ràng:\nInventory↑ → Oil Price↓",
         transform=ax3b.transAxes, color=COLORS["subtext"], fontsize=7,
         bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))

# ============================================================
# ROW 4: GDELT
# ============================================================
ax4a = fig.add_subplot(gs[4, 0])
ax4a.plot(dates, gdelt_tone, color=COLORS["tone"], linewidth=0.6, alpha=0.3)
ax4a.plot(dates, gdelt_tone_7d, color=COLORS["tone"], linewidth=1.8, label="7-day MA")
ax4a.axhline(0, color="white", linewidth=0.5, alpha=0.3, linestyle="--")
ax4a.fill_between(dates, 0, gdelt_tone_7d, where=(gdelt_tone_7d < 0),
                 color=COLORS["tone"], alpha=0.2, label="Negative territory")
style_ax(ax4a, "📰 GDELT — Media Tone (Trung Đông)", "Sentiment Score", "GDELT DOC API")
ax4a.legend(loc="upper right", fontsize=7,
           facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
ax4a.text(0.02, 0.05,
         "Tone âm = tin tức tiêu cực/căng thẳng\nDùng trực tiếp cho nhóm feature Geopolitical Risk",
         transform=ax4a.transAxes, color=COLORS["subtext"], fontsize=7,
         bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))

# ---
ax4b = fig.add_subplot(gs[4, 1])
ax4b.fill_between(dates, 0, gdelt_volume, alpha=0.25, color=COLORS["vol"])
ax4b.plot(dates, gdelt_volume.rolling(30).mean(), color=COLORS["vol"], linewidth=1.5, label="30d MA Volume")
style_ax(ax4b, "📰 GDELT — Media Volume (Trung Đông)", "Số bài viết/ngày", "GDELT DOC API")

# Dual axis: overlay oil price
ax4b_twin = ax4b.twinx()
ax4b_twin.plot(dates, oil_price, color=COLORS["oil"], linewidth=1, alpha=0.6, linestyle="--")
ax4b_twin.set_ylabel("Oil Price (USD)", fontsize=8, color=COLORS["oil"])
ax4b_twin.tick_params(axis="y", colors=COLORS["oil"], labelsize=7)

lines2 = [mpatches.Patch(color=COLORS["vol"], label="Media Volume"),
          mpatches.Patch(color=COLORS["oil"], label="Oil Price (ref.)")]
ax4b.legend(handles=lines2, loc="upper left", fontsize=7,
           facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

# ============================================================
# Save
# ============================================================
plt.savefig("oil_data_output/data_sources_overview.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("✅ Đã lưu: oil_data_output/data_sources_overview.png")
plt.close()
