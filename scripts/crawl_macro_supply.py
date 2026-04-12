"""
Script crawl dữ liệu bổ sung cho đồ án Dự đoán Giá Dầu
Nguồn: FRED, EIA
Giai đoạn output: 2015-01-01 đến 2026-03-20 (crawl lùi về 2013-12-01 để đủ lịch sử tính YoY)

Cài đặt thư viện trước khi chạy:
    pip install pandas requests fredapi
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ⚙️  CẤU HÌNH - Điền API key vào đây
# ============================================================
FRED_API_KEY = "dea911d6422f832a814ff0f3a3a13aa6"   # https://fred.stlouisfed.org/docs/api/api_key.html
EIA_API_KEY  = "3sFfSfhcUvNA9HwKAGmgAJeB9ZQqn1Yo6TY6sAnX"    # https://www.eia.gov/opendata/register.php

# Crawl lùi 13 tháng để đủ lịch sử tính YoY chuẩn ngay từ giai đoạn đầu output.
CRAWL_START_DATE  = "2013-12-01"
OUTPUT_START_DATE = "2015-01-01"
END_DATE          = "2026-03-20"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(RAW_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 📌 PHẦN 1: FRED DATA
# ============================================================
def fetch_fred_data(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Lấy các chỉ số kinh tế vĩ mô từ FRED.
    
    Series được lấy:
        FEDFUNDS  - Lãi suất Fed Funds (monthly → ffill về daily)
        CPIAUCSL  - Chỉ số giá tiêu dùng CPI (monthly → ffill)
        UNRATE    - Tỷ lệ thất nghiệp Mỹ (monthly → ffill)
        T10Y2Y    - Yield spread 10Y-2Y Treasury (daily, chỉ báo recession)
        DCOILWTICO- Giá dầu WTI từ FRED để cross-check với Yahoo Finance
    
    Lưu ý về publication lag:
        CPI tháng 1 thường được công bố giữa tháng 2 → cần shift(1) để tránh data leakage.
        FEDFUNDS tháng T công bố đầu tháng T+1 → tương tự.
    """
    series_config = {
        "FEDFUNDS" : {"name": "fed_funds_rate",  "freq": "monthly"},
        "CPIAUCSL" : {"name": "cpi",             "freq": "monthly"},
        "UNRATE"   : {"name": "unemployment",    "freq": "monthly"},
        "T10Y2Y"   : {"name": "yield_spread",    "freq": "daily"},
        "DCOILWTICO": {"name": "wti_fred",       "freq": "daily"},
    }

    dfs = []
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for series_id, cfg in series_config.items():
        print(f"  [FRED] Đang lấy {series_id} ({cfg['name']})...")
        params = {
            "series_id"        : series_id,
            "api_key"          : api_key,
            "file_type"        : "json",
            "observation_start": start,
            "observation_end"  : end,
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()["observations"]

            df = pd.DataFrame(data)[["date", "value"]]
            df.columns = ["date", cfg["name"]]
            df["date"] = pd.to_datetime(df["date"])
            # "." là giá trị missing trong FRED
            df[cfg["name"]] = pd.to_numeric(df[cfg["name"]], errors="coerce")
            df = df.set_index("date")

            # Shift 1 tháng để tránh data leakage với monthly data
            if cfg["freq"] == "monthly":
                df.index = df.index + pd.DateOffset(months=1)

            dfs.append(df)
            time.sleep(0.3)  # Tránh rate limit

        except Exception as e:
            print(f"  [FRED] ⚠️  Lỗi khi lấy {series_id}: {e}")

    if not dfs:
        return pd.DataFrame()

    # Tạo daily timeline rồi merge tất cả
    daily_index = pd.date_range(start=start, end=end, freq="B")  # Business days
    result = pd.DataFrame(index=daily_index)
    for df in dfs:
        result = result.join(df, how="left")

    # Forward fill: giá trị monthly/weekly "có hiệu lực" đến khi có báo cáo mới
    result = result.ffill()
    result = result.loc[result.index <= end]
    result.index.name = "date"
    
    # ========================================================================
    # [BƯỚC 2: DATA CLEANING] 
    # Lưu ý: Phần ffill ở trên làm ở bước crawl để thích hợp với những tần suất khác nhau_FRED monthly, EIA weekly)
    # Phần ffill này sẽ được xử lý thêm ở step2_cleaning.py khi merge với business day index
    # ========================================================================
    
    return result


# ============================================================
# 📌 PHẦN 2: EIA DATA
# ============================================================
def fetch_eia_data(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Lấy dữ liệu cung/cầu dầu từ EIA API V2.
    
    Series được lấy:
        PET.WCRSTUS1.W  - U.S. Crude Oil Stocks (tồn kho, weekly - mỗi thứ Tư)
        PET.WCRFPUS2.W  - U.S. Crude Oil Production (sản xuất, weekly)
        PET.WTTNTUS2.W  - U.S. Total Petroleum Net Imports (nhập khẩu ròng, weekly)

    Tồn kho dầu (inventory) là chỉ số thị trường theo dõi sát nhất:
        - Tồn kho tăng → cung dư thừa → giá dầu có xu hướng giảm
        - Tồn kho giảm → cung thiếu hụt → giá dầu có xu hướng tăng
    
    "Inventory surprise" = inventory thực tế - kỳ vọng thị trường
    → Feature này thường mạnh hơn bản thân con số tuyệt đối.
    """
    series_config = {
        "PET.WCRSTUS1.W": "crude_inventory_weekly",
        "PET.WCRFPUS2.W": "crude_production_weekly",
        "PET.WTTNTUS2.W": "net_imports_weekly",
    }

    dfs = []
    base_url = "https://api.eia.gov/v2/seriesid/{series_id}"

    for series_id, col_name in series_config.items():
        print(f"  [EIA] Đang lấy {series_id} ({col_name})...")
        url = f"https://api.eia.gov/v2/seriesid/{series_id}"
        params = {
            "api_key": api_key,
            "start"  : start,
            "end"    : end,
            "out"    : "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()

            # EIA V2 trả về nested JSON - cần flatten
            rows = raw.get("response", {}).get("data", [])
            if not rows:
                # Thử structure cũ
                rows = raw.get("request", {}).get("series", [{}])[0].get("data", [])

            df = pd.DataFrame(rows)
            if df.empty:
                print(f"  [EIA] ⚠️  Không có data cho {series_id}")
                continue

            # Chuẩn hóa tên cột (EIA V2 dùng "period" và "value")
            date_col  = "period" if "period" in df.columns else df.columns[0]
            value_col = "value"  if "value"  in df.columns else df.columns[1]

            df = df[[date_col, value_col]].copy()
            df.columns = ["date", col_name]
            df["date"]    = pd.to_datetime(df["date"])
            df[col_name]  = pd.to_numeric(df[col_name], errors="coerce")
            df = df.set_index("date").sort_index()

            dfs.append(df)
            time.sleep(0.3)

        except Exception as e:
            print(f"  [EIA] ⚠️  Lỗi khi lấy {series_id}: {e}")

    if not dfs:
        return pd.DataFrame()

    daily_index = pd.date_range(start=start, end=end, freq="B")
    result = pd.DataFrame(index=daily_index)
    for df in dfs:
        result = result.join(df, how="left")

    # Weekly data → ffill về daily (giống cách xử lý FRED monthly)
    result = result.ffill()

    # Feature engineering: % thay đổi tồn kho tuần so với tuần trước
    if "crude_inventory_weekly" in result.columns:
        result["inventory_change_pct"] = result["crude_inventory_weekly"].pct_change(5)  # 5 business days ≈ 1 week

    result = result.loc[result.index <= end]
    result.index.name = "date"
    return result


# ============================================================
# 🚀 MAIN - Chạy tất cả
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🛢️  Crawl dữ liệu bổ sung cho đồ án Dự đoán Giá Dầu")
    print(f"   Crawl range : {CRAWL_START_DATE} → {END_DATE}")
    print(f"   Output range: {OUTPUT_START_DATE} → {END_DATE}")
    print("=" * 60)

    all_dfs = {}

    # --- FRED ---
    if FRED_API_KEY != "YOUR_FRED_API_KEY":
        print("\n📊 [1/2] Crawl FRED...")
        fred_df = fetch_fred_data(FRED_API_KEY, CRAWL_START_DATE, END_DATE)
        if not fred_df.empty:
            path = os.path.join(OUTPUT_DIR, "fred_data.csv")
            fred_df.to_csv(path)
            print(f"  ✅ Lưu {path} — {len(fred_df)} rows, {len(fred_df.columns)} cols")
            print(f"  Columns: {list(fred_df.columns)}")
            all_dfs["FRED"] = fred_df
    else:
        print("\n⚠️  [1/2] Bỏ qua FRED (chưa có API key)")

    # --- EIA ---
    if EIA_API_KEY != "YOUR_EIA_API_KEY":
        print("\n🏭 [2/2] Crawl EIA...")
        eia_df = fetch_eia_data(EIA_API_KEY, CRAWL_START_DATE, END_DATE)
        if not eia_df.empty:
            path = os.path.join(OUTPUT_DIR, "eia_data.csv")
            eia_df.to_csv(path)
            print(f"  ✅ Lưu {path} — {len(eia_df)} rows, {len(eia_df.columns)} cols")
            print(f"  Columns: {list(eia_df.columns)}")
            all_dfs["EIA"] = eia_df
    else:
        print("\n⚠️  [2/2] Bỏ qua EIA (chưa có API key)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📋 Tóm tắt:")
    for name, df in all_dfs.items():
        missing = df.isnull().sum().sum()
        total   = df.shape[0] * df.shape[1]
        print(f"  {name}: {df.shape[0]} rows × {df.shape[1]} cols | Missing: {missing}/{total} ({missing/total*100:.1f}%)")
    print("\nDone! Kiểm tra thư mục:", OUTPUT_DIR)
