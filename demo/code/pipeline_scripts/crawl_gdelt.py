"""
GDELT Crawler v4 — Direct File Download (không dùng DOC API)
=============================================================
Thay vì gọi API (bị rate limit), ta download thẳng file CSV daily
từ GDELT 1.0 public dataset:
    http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Chiến lược:
  - Download 1 file/tuần (mỗi thứ Hai) → ~572 files cho 11 năm
  - Mỗi file ~1-3MB zip, unzip ~10-30MB
  - Filter ngay các sự kiện liên quan Trung Đông
  - Extract: AvgTone (sentiment), NumArticles (volume), GoldsteinScale
  - ffill về daily giống EIA

Thời gian ước tính: 15-30 phút tuỳ tốc độ mạng
"""

import pandas as pd
import requests
import zipfile
import io
import os
import shutil
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = str(RAW_DIR)
PROGRESS_FILE = str(RAW_DIR / "gdelt_v4_progress.json")
OUTPUT_FILE   = str(RAW_DIR / "gdelt_data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2015-01-01"
TARGET_END_DATE = "2026-03-20"
END_DATE = min(datetime.today().strftime("%Y-%m-%d"), TARGET_END_DATE)

# Country codes Trung Đông trong GDELT (ISO 3-letter)
MIDDLE_EAST_COUNTRIES = {
    "IRQ",  # Iraq
    "IRN",  # Iran
    "SAU",  # Saudi Arabia
    "SYR",  # Syria
    "YEM",  # Yemen
    "ISR",  # Israel
    "PSE",  # Palestine
    "LBN",  # Lebanon
    "KWT",  # Kuwait
    "ARE",  # UAE
    "QAT",  # Qatar
    "BHR",  # Bahrain
    "OMN",  # Oman
    "JOR",  # Jordan
}

# GDELT 1.0 column names (28 cột)
GDELT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type",
    "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_dates(start: str, end: str):
    """Trả về list các ngày trong khoảng [start, end]"""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    current = start_dt
    all_dates = []
    while current <= end_dt:
        all_dates.append(current)
        current += timedelta(days=1)
    return all_dates


def download_and_filter(date: datetime) -> dict | None:
    """
    Download GDELT file cho ngày cụ thể, filter Trung Đông,
    trả về dict {date, avg_tone, num_articles, goldstein, num_events}
    """
    date_str = date.strftime("%Y%m%d")
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)

        if resp.status_code == 404:
            # File không tồn tại (ngày lễ, etc.)
            return None
        resp.raise_for_status()

        # Đọc zip trong memory, không lưu file
        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    names=GDELT_COLS[:len(GDELT_COLS)],  # tự động match
                    on_bad_lines="skip",
                    dtype=str,
                    low_memory=False,
                )

        # Filter: chỉ lấy event liên quan Trung Đông
        me_mask = (
            df["Actor1CountryCode"].isin(MIDDLE_EAST_COUNTRIES) |
            df["Actor2CountryCode"].isin(MIDDLE_EAST_COUNTRIES) |
            df["ActionGeo_CountryCode"].isin(MIDDLE_EAST_COUNTRIES)
        )
        df_me = df[me_mask].copy()

        if df_me.empty:
            return None

        # Convert numeric
        for col in ["AvgTone", "GoldsteinScale", "NumArticles", "NumMentions"]:
            if col in df_me.columns:
                df_me[col] = pd.to_numeric(df_me[col], errors="coerce")

        return {
            "date"        : date.strftime("%Y-%m-%d"),
            "gdelt_tone"  : float(df_me["AvgTone"].mean())         if "AvgTone"       in df_me.columns else None,
            "gdelt_goldstein": float(df_me["GoldsteinScale"].mean()) if "GoldsteinScale" in df_me.columns else None,
            "gdelt_volume": int(df_me["NumArticles"].sum())       if "NumArticles"   in df_me.columns else None,
            "gdelt_events": int(len(df_me)),
        }

    except zipfile.BadZipFile:
        return None
    except Exception as e:
        print(f"\n      ⚠️  {type(e).__name__}: {str(e)[:60]}")
        return None


def load_progress() -> tuple[list, set[str]]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # Backward compatibility: format cũ chỉ là list rows.
            if isinstance(payload, list):
                rows = payload
                processed_dates = {r["date"] for r in rows if isinstance(r, dict) and "date" in r}
                return rows, processed_dates

            if isinstance(payload, dict):
                rows = payload.get("rows", [])
                processed_dates = set(payload.get("processed_dates", []))
                if not processed_dates:
                    processed_dates = {r["date"] for r in rows if isinstance(r, dict) and "date" in r}
                return rows, processed_dates

            return [], set()
        except json.JSONDecodeError:
            # Progress file có thể bị hỏng nếu lần chạy trước bị ngắt giữa lúc ghi.
            backup = PROGRESS_FILE + ".corrupt"
            try:
                os.replace(PROGRESS_FILE, backup)
                print(f"⚠️  Progress file hỏng, đã backup sang: {backup}")
            except OSError:
                pass
            return [], set()
    return [], set()


def save_progress(rows: list, processed_dates: set[str]):
    # Ghi atomically để tránh tạo JSON dở dang nếu process bị kill.
    tmp_file = PROGRESS_FILE + ".tmp"

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    payload = {
        "rows": rows,
        "processed_dates": sorted(processed_dates),
    }

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=_json_default)
    os.replace(tmp_file, PROGRESS_FILE)


def load_existing_rows_from_output() -> tuple[list, datetime | None]:
    """Đọc dữ liệu base đã có trong gdelt_data.csv để crawl nối phần thiếu."""
    if not os.path.exists(OUTPUT_FILE):
        return [], None

    try:
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
    except Exception:
        return [], None

    base_cols = ["gdelt_tone", "gdelt_goldstein", "gdelt_volume", "gdelt_events"]
    if "date" not in existing.columns or any(col not in existing.columns for col in base_cols):
        return [], None

    existing = existing[["date"] + base_cols].copy()
    existing = existing.dropna(subset=base_cols, how="all")
    if existing.empty:
        return [], None

    rows = []
    for _, row in existing.iterrows():
        rows.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "gdelt_tone": float(row["gdelt_tone"]) if pd.notna(row["gdelt_tone"]) else None,
                "gdelt_goldstein": float(row["gdelt_goldstein"]) if pd.notna(row["gdelt_goldstein"]) else None,
                "gdelt_volume": int(row["gdelt_volume"]) if pd.notna(row["gdelt_volume"]) else None,
                "gdelt_events": int(row["gdelt_events"]) if pd.notna(row["gdelt_events"]) else None,
            }
        )

    last_date = existing["date"].max()
    return rows, last_date


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📰 GDELT Crawler v4 — Direct File Download")
    print(f"   Source : data.gdeltproject.org/events/")
    print(f"   Period : {START_DATE} → {END_DATE}")
    print(f"   Freq   : Daily")
    print("=" * 60)

    # Load progress nếu có, nếu không thì nối tiếp từ file output cũ
    saved_rows, processed_dates = load_progress()
    all_rows = list(saved_rows)

    if processed_dates:
        crawl_start = START_DATE
        print(
            f"📂 Resume progress: đã xử lý {len(processed_dates)} ngày, "
            f"trong đó có {len(all_rows)} ngày có events Trung Đông...\n"
        )
    else:
        existing_rows, last_existing_date = load_existing_rows_from_output()
        all_rows = existing_rows
        if last_existing_date is not None:
            crawl_start = (last_existing_date + timedelta(days=1)).strftime("%Y-%m-%d")
            print(
                f"📂 Tìm thấy dữ liệu cũ tới {last_existing_date.strftime('%Y-%m-%d')}, "
                f"sẽ crawl nối từ {crawl_start}...\n"
            )
        else:
            crawl_start = START_DATE

    all_dates = get_dates(crawl_start, END_DATE)
    print(f"   Cần download thêm: {len(all_dates)} files\n")

    for i, current_date in enumerate(all_dates):
        date_str = current_date.strftime("%Y-%m-%d")

        if date_str in processed_dates:
            continue  # Skip đã có

        print(f"  [{i+1:4d}/{len(all_dates)}] {date_str} ...", end=" ", flush=True)

        result = download_and_filter(current_date)

        if result:
            all_rows.append(result)
            print(f"✅  tone={result['gdelt_tone']:.2f}  "
                  f"vol={result['gdelt_volume']:.0f}  "
                  f"events={result['gdelt_events']}")
        else:
            print("— (skip)")

        processed_dates.add(date_str)

        # Lưu progress mỗi 20 files
        if (i + 1) % 20 == 0:
            save_progress(all_rows, processed_dates)

        time.sleep(0.2)  # Nhẹ nhàng thôi, đây là file download không phải API

    # Save lần cuối
    save_progress(all_rows, processed_dates)

    # ── Build final DataFrame ──────────────────────────────
    print(f"\n📦 Build dataset từ {len(all_rows)} daily entries có events...")

    if not all_rows:
        print("❌ Không có data.")
        exit()

    daily_df = pd.DataFrame(all_rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.set_index("date").sort_index()

    # Align về business days, không ffill để giữ tín hiệu daily thật.
    daily_index = pd.date_range(START_DATE, END_DATE, freq="B")
    result = pd.DataFrame(index=daily_index)
    result.index.name = "date"
    result = result.join(daily_df, how="left")

    # Feature engineering
    if "gdelt_tone" in result.columns:
        result["gdelt_tone_7d"]    = result["gdelt_tone"].rolling(7,  min_periods=1).mean()
        result["gdelt_tone_30d"]   = result["gdelt_tone"].rolling(30, min_periods=1).mean()
        result["gdelt_tone_spike"] = (
            result["gdelt_tone"] < result["gdelt_tone_30d"] - 1
        ).astype(int)

    if "gdelt_volume" in result.columns:
        result["gdelt_volume_7d"]  = result["gdelt_volume"].rolling(7, min_periods=1).mean()

    if "gdelt_goldstein" in result.columns:
        result["gdelt_goldstein_7d"] = result["gdelt_goldstein"].rolling(7, min_periods=1).mean()

    # ========================================================================
    # [BƯỚC 4: DATA TRANSFORMATION]
    # Lưu ý: Rolling window aggregation ở trên (gdelt_tone_7d, gdelt_volume_7d, etc.)
    # được tính ở bước crawl vì phải có đủ dữ liệu daily trước.
    # Tuy nhiên, toàn bộ transformation logic sẽ được tập trung ở step4_transformation.py
    # để dễ theo dõi và bảo trì.
    # ========================================================================

    out = OUTPUT_FILE
    
    # Backup file cũ trước khi ghi đè (an toàn trong trường hợp interrupt)
    if os.path.exists(out):
        backup_path = out + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copy2(out, backup_path)
        except Exception as e:
            print(f"⚠️  Lỗi khi backup: {e}")
    
    # Ghi sang temporary file trước, sau đó atomic rename
    tmp_out = out + ".tmp"
    result.to_csv(tmp_out)
    os.replace(tmp_out, out)  # Atomic replace

    miss  = result.isnull().sum().sum()
    total = result.shape[0] * result.shape[1]
    print(f"\n✅ Lưu: {out}")
    print(f"   Shape  : {result.shape[0]} rows × {result.shape[1]} cols")
    print(f"   Missing: {miss}/{total} ({miss/total*100:.1f}%)")
    print(f"   Columns: {list(result.columns)}")
    print(f"\nSample (5 rows):")
    print(result.dropna().head(5).to_string())

    # Cleanup
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("\n✅ Done!")