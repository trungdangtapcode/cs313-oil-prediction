# Scripts Pipeline Map

File này giải thích từng script trong thư mục `scripts/` tương ứng với bước nào trong pipeline dữ liệu, nên chạy theo thứ tự nào, và file nào là đầu ra chính.

## 1. Nhóm thu thập dữ liệu thô

### `ingest_data.py`
- Vai trò: lấy dữ liệu thị trường để tạo `market_data.csv`
- Thuộc bước: thu thập dữ liệu thị trường
- Output chính: `data/raw/market_data.csv`

### `crawl_macro_supply.py`
- Vai trò: crawl dữ liệu FRED và EIA
- Thuộc bước: thu thập dữ liệu vĩ mô và cung cầu dầu
- Output chính:
  - `data/raw/fred_data.csv`
  - `data/raw/eia_data.csv`

### `crawl_gdelt.py`
- Vai trò: crawl dữ liệu GDELT về tone, volume, event, goldstein
- Thuộc bước: thu thập dữ liệu truyền thông và địa chính trị
- Output chính: `data/raw/gdelt_data.csv`

## 2. Nhóm hỗ trợ hiểu dữ liệu

### `visualize_data.py`
- Vai trò: tạo sample dashboard để hiểu cấu trúc dữ liệu
- Thuộc bước: hỗ trợ khám phá dữ liệu
- Ghi chú: không phải bước bắt buộc của pipeline train

### `eda_full.py`
- Vai trò: EDA trên dataset đã xử lý
- Thuộc bước: phân tích dữ liệu sau pipeline
- Ghi chú: không phải bước bắt buộc để tạo dataset

### `feature_engineering.py`
- Vai trò: file tiện ích hoặc thử nghiệm cũ liên quan feature engineering
- Thuộc bước: script phụ trợ
- Ghi chú: pipeline chính hiện dùng `step4_transformation.py`

### `preprocess_data.py`
- Vai trò: pipeline tiền xử lý kiểu cũ hơn, gộp nhiều thao tác vào một script
- Thuộc bước: script phụ trợ hoặc legacy
- Ghi chú: pipeline chính hiện ưu tiên bộ `step1` đến `step6`

## 3. Pipeline xử lý chính

### `step1_load_inspect.py`
- Vai trò: load dữ liệu raw và kiểm tra shape, cột, kiểu dữ liệu, missing
- Thuộc bước: Bước 1 - Load and Inspect
- Input: `data/raw/*.csv`
- Output: báo cáo console

### `step2_cleaning.py`
- Vai trò: làm sạch dữ liệu, xử lý thiếu dữ liệu, chuẩn hóa format ngày, xử lý ACLED cuối tuần
- Thuộc bước: Bước 2 - Cleaning
- Input: dữ liệu raw
- Output chính: `data/processed/dataset_step2_cleaned.csv`

### `step3_integration.py`
- Vai trò: reindex business day và gộp nhiều nguồn lên cùng timeline
- Thuộc bước: Bước 3 - Integration
- Input: `data/processed/dataset_step2_cleaned.csv`
- Output chính: `data/processed/dataset_step3_integrated.csv`

### `step4_transformation.py`
- Vai trò: tạo feature như return, rolling, lag, macro derived, supply derived, stress index
- Thuộc bước: Bước 4 - Transformation / Feature Engineering
- Input: `data/processed/dataset_step3_integrated.csv`
- Output chính: `data/processed/dataset_step4_transformed.csv`

### `step4b_fix_leakage.py`
- Vai trò: tạo bản no-leakage bảo thủ bằng cách drop các cột đã bị contamination do release timing, split leakage, hoặc full-series preprocessing leakage
- Thuộc bước: Bước 4B - Leakage Fix
- Input: `data/processed/dataset_step4_transformed.csv`
- Output chính:
  - `data/processed/dataset_step4_noleak.csv`
  - `data/processed/dataset_final_noleak.csv`
  - `data/processed/dataset_step4_noleak_drop_report.csv`
- Ghi chú:
  - Đây là bước no-leakage theo hướng bảo thủ: loại cột bẩn thay vì cố sửa timestamp ngay trong `step4`
  - Nếu muốn train nghiêm ngặt hơn, nên ưu tiên dùng output của bước này

### `step5_reduction.py`
- Vai trò: bỏ cột trung gian, cột collinearity cao, cột raw không cần thiết để tạo dataset model-ready
- Thuộc bước: Bước 5 - Reduction
- Input: `data/processed/dataset_step4_transformed.csv`
- Output chính:
  - `data/processed/dataset_final_full.csv`
  - `data/processed/dataset_final.csv`
- Ghi chú:
  - Đây là reduction trên dataset gốc sau step4
  - Nếu muốn reduction trên bản no-leakage, dùng `step4b_fix_leakage.py`

### `step5b_processing.py`
- Vai trò: áp dụng processing an toàn sau `step4b` bằng các transform deterministic như cyclical encoding, `log1p`, `signed_log1p`
- Thuộc bước: Bước 5B - Processing
- Input: `data/processed/dataset_final_noleak.csv`
- Output chính:
  - `data/processed/dataset_final_noleak_processed.csv`
  - `data/processed/dataset_final_noleak_processing_report.csv`
- Ghi chú:
  - Bước này cố ý không fit scaler trên full dataset để tránh leakage mới
  - Các scaler như `StandardScaler`, `RobustScaler`, `PowerTransformer` vẫn nên fit trong pipeline train

### `step6_quality_check.py`
- Vai trò: kiểm tra NaN, INF, leakage cơ bản, train/test split, sample data
- Thuộc bước: Bước 6 - Quality Check
- Input: `data/processed/dataset_final.csv`
- Output: báo cáo console
- Ghi chú:
  - Hiện script này đang check `dataset_final.csv`
  - Nếu muốn check bản no-leakage, có thể sửa input sang `dataset_final_noleak.csv`

## 4. Thứ tự chạy khuyến nghị

### Pipeline gốc
1. `step1_load_inspect.py`
2. `step2_cleaning.py`
3. `step3_integration.py`
4. `step4_transformation.py`
5. `step5_reduction.py`
6. `step6_quality_check.py`

### Pipeline có xử lý leakage
1. `step1_load_inspect.py`
2. `step2_cleaning.py`
3. `step3_integration.py`
4. `step4_transformation.py`
5. `step4b_fix_leakage.py`
6. `step5b_processing.py`
7. kiểm tra thêm hoặc train trực tiếp trên:
   - `data/processed/dataset_step4_noleak.csv`
   - `data/processed/dataset_final_noleak.csv`
   - `data/processed/dataset_final_noleak_processed.csv`

## 5. File nào nên dùng cho modeling

### Nếu muốn giữ đúng pipeline cũ
- Dùng `data/processed/dataset_final.csv`

### Nếu muốn hạn chế leakage
- Dùng `data/processed/dataset_step4_noleak.csv` nếu cần full feature set
- Dùng `data/processed/dataset_final_noleak.csv` nếu cần model-ready dataset
- Dùng `data/processed/dataset_final_noleak_processed.csv` nếu muốn thêm processing deterministic trước khi train
