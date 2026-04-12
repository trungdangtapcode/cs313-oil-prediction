# OIL PRICE PREDICTION PROJECT
**Dự đoán biến động giá dầu bằng dữ liệu kinh tế, thị trường và địa chính trị**

---

## 1) Tổng quan

Dự án xây dựng pipeline dữ liệu và mô hình để dự đoán biến động giá dầu (`oil_return`) từ nhiều nguồn:

- **Địa chính trị thực địa**: ACLED (sự kiện xung đột, thương vong)
- **Địa chính trị truyền thông**: GDELT (tone, goldstein, volume tin)
- **Thị trường tài chính**: Yahoo Finance (USD, S&P500, VIX)
- **Vĩ mô**: FRED (Fed Funds, CPI, Unemployment, Yield Spread)
- **Cung/cầu dầu**: EIA (inventory, production, net imports)

Mục tiêu chính là tạo bộ dữ liệu sạch, chống leakage, và đủ ổn định để đưa vào mô hình ML/Time-series.

---

## 2) Trạng thái hiện tại (đã chạy xong)

- `dataset_final.csv`: **2923 rows x 37 cols**
- Không có **NaN/INF**
- Train/Test split:
  - Train: 2083 rows (2015-01-07 -> 2022-12-30)
  - Test: 840 rows (2023-01-02 -> 2026-03-20)
- Đã chạy kiểm tra chất lượng bằng `scripts/step6_quality_check.py`

---

## 3) Cấu trúc thư mục

```text
OilPriceProject/
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── ACLED Data_2026-03-26.csv
│   │   ├── eia_data.csv
│   │   ├── fred_data.csv
│   │   ├── gdelt_data.csv
│   │   └── market_data.csv
│   └── processed/
│       ├── dataset_preprocessed.csv
│       ├── dataset_step2_cleaned.csv
│       ├── dataset_step3_integrated.csv
│       ├── dataset_step4_transformed.csv
│       ├── dataset_final_full.csv
│       └── dataset_final.csv
├── scripts/
│   ├── crawl_gdelt.py
│   ├── crawl_macro_supply.py
│   ├── ingest_data.py
│   ├── preprocess_data.py
│   ├── feature_engineering.py
│   ├── visualize_data.py
│   ├── step1_load_inspect.py
│   ├── step2_cleaning.py
│   ├── step3_integration.py
│   ├── step4_transformation.py
│   ├── step5_reduction.py
│   └── step6_quality_check.py
└── notebooks/
    └── step4b_eda.ipynb
```

---

## 4) Hướng dẫn nhanh

### 4.1 Cài môi trường

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Chạy full pipeline (khuyến nghị)

```bash
python scripts/step1_load_inspect.py
python scripts/step2_cleaning.py
python scripts/step3_integration.py
python scripts/step4_transformation.py
python scripts/step5_reduction.py
python scripts/step6_quality_check.py
```

---

## 5) Pipeline theo giai đoạn (dễ hiểu)

### Giai đoạn A - Thu thập dữ liệu thô

#### Bước A1: ACLED (thủ công)

1. Vào: https://acleddata.com/conflict-data/data-export-tool
2. Chọn khu vực Trung Đông (theo danh sách của đề tài)
3. Chọn thời gian: 2015-01-01 đến hiện tại
4. Export CSV vào `data/raw/` (đang dùng file `ACLED Data_2026-03-26.csv`)

#### Bước A2: Crawl FRED + EIA

```bash
python scripts/crawl_macro_supply.py --fred-key YOUR_FRED_KEY --eia-key YOUR_EIA_KEY
```

#### Bước A3: Crawl GDELT

```bash
python scripts/crawl_gdelt.py
```

#### Bước A4: Ingest dữ liệu thị trường

```bash
python scripts/ingest_data.py
```

---

### Giai đoạn B - Pipeline xử lý chính (Step 1 -> Step 6)

#### Step 1 - Load & Inspect

- Script: `scripts/step1_load_inspect.py`
- Input: `data/raw/*.csv`
- Mục tiêu:
  - Kiểm tra shape, dtype, date range, missing
  - Xác nhận dữ liệu đủ để vào cleaning
- Output: in console

#### Step 2 - Cleaning

- Script: `scripts/step2_cleaning.py`
- Input: dữ liệu raw
- Mục tiêu:
  - Fill missing theo từng nguồn
  - Xử lý ACLED cuối tuần -> dồn vào ngày giao dịch
  - Chuẩn hóa format ngày, bỏ duplicate
- Output: `data/processed/dataset_step2_cleaned.csv`

#### Step 3 - Integration

- Script: `scripts/step3_integration.py`
- Input: `dataset_step2_cleaned.csv`
- Mục tiêu:
  - Reindex business-day
  - Merge đa nguồn vào một timeline thống nhất
  - Kiểm tra redundancy cơ bản
- Output: `data/processed/dataset_step3_integrated.csv`

#### Step 4 - Transformation & Feature Engineering

- Script: `scripts/step4_transformation.py`
- Input: `dataset_step3_integrated.csv`
- Mục tiêu:
  - Tạo returns, rolling features, lag features, time features
  - Tạo derived features (macro/supply/sentiment/conflict)
  - Tạo `geopolitical_stress_index`
  - Winsorize volatility
- Output: `data/processed/dataset_step4_transformed.csv`

#### Step 5 - Data Reduction (Aggressive)

- Script: `scripts/step5_reduction.py`
- Input: `dataset_step4_transformed.csv`
- Mục tiêu:
  - Giảm đa cộng tuyến mạnh
  - Bỏ biến trung gian + biến nhiễu + biến raw không cần thiết
- Output:
  - `data/processed/dataset_final_full.csv` (giữ full)
  - `data/processed/dataset_final.csv` (model-ready)

**Kết quả hiện tại Step 5**
- Giảm từ 54 cột xuống 33 cột
- Tỷ lệ giữ lại: 68.5%

#### Step 6 - Final Quality Check

- Script: `scripts/step6_quality_check.py`
- Input: `dataset_final.csv`
- Mục tiêu:
  - Kiểm tra NaN/INF
  - Kiểm tra leakage
  - In train/test split và sample dữ liệu
- Output: báo cáo console cuối cùng trước modeling

---

## 6) Tóm tắt feature sau reduction

Bộ dữ liệu model-ready hiện giữ các nhóm thông tin chính:

- **Market/Returns**: `oil_return`, `usd_return`, `sp500_return`, `vix_return`, ...
- **Macro**: `real_rate`, `fed_rate_change`, `yield_spread`, `cpi_lag`, `unemployment_lag`, `fed_rate_regime`
- **Supply tương đối**: `inventory_zscore`, `production_change_pct`, `net_imports_change_pct`
- **Sentiment/Conflict**: `gdelt_tone_7d`, `gdelt_tone_lag1`, `gdelt_volume_lag1`, `gdelt_goldstein`, `gdelt_goldstein_7d`, `geopolitical_stress_index`, `conflict_intensity_7d`, `fatalities_7d`
- **Lag/Temporal**: `oil_return_lag1`, `oil_return_lag2`, `day_of_week`, `month`

---

## 7) Lưu ý quan trọng

- Dự án dự đoán **`oil_return`** (biến động), không phải trực tiếp giá tuyệt đối.
- Khi train mô hình, nên ưu tiên dùng biến **lag/return/change** để hạn chế leakage và non-stationary risk.
- Step 6 hiện cảnh báo khả năng leakage cho một số biến same-day (`vix_close`, `usd_close`, `sp500_close`); khi modeling có thể dùng phiên bản lag nếu muốn nghiêm ngặt hơn.

---

## 8) Notebook EDA

- Notebook chính: `notebooks/step4b_eda.ipynb`
- Nội dung: target analysis, ADF, ACF/PACF, correlation, VIF, rolling correlation, outlier analysis, summary table.

---

## 9) Tài liệu tham khảo

- ACLED: https://acleddata.com/
- GDELT: https://www.gdeltproject.org/
- FRED: https://fred.stlouisfed.org/
- EIA: https://www.eia.gov/
- Yahoo Finance: https://finance.yahoo.com/

---

## 10) Cập nhật gần nhất

- Last Updated: **2026-03-30**
