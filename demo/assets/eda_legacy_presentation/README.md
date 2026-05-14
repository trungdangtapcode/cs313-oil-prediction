# Plot Presentation — EDA Oil Price Classification

**Bài toán**: Binary classification dự đoán xu hướng tăng/giảm (Up/Down) của giá dầu ngày tiếp theo.  
**Dataset**: `data/processed/dataset_final.csv` — Train: 2015–2022, Test: 2023–2026  
**Tạo bởi**: `scripts/create_presentation_plots.py`

---

## Thứ tự đề xuất đưa vào slide

| #   | File                           | Slide title                       |
| --- | ------------------------------ | --------------------------------- |
| 1   | `01_oil_returns_over_time.png` | Tổng quan dữ liệu giá dầu         |
| 2   | `02_target_class_balance.png`  | Định nghĩa target & class balance |
| 3   | `03_target_over_time.png`      | Hành vi target theo thời gian     |
| 4   | `04_data_quality.png`          | Chất lượng dữ liệu                |
| 5   | `05_feature_signals.png`       | Features có tín hiệu hữu ích      |
| 6   | `06_timeseries_properties.png` | Đặc tính time-series              |
| 7   | `07_split_leakage_check.png`   | Kiểm tra split & leakage          |

---

## Chi tiết từng plot

---

### 01 — `01_oil_returns_over_time.png`

**Mục đích**: Mở đầu slide bằng "bức tranh toàn cảnh" — cho thấy dữ liệu giá dầu rất volatile, có nhiều chế độ thị trường khác nhau, và train/test được chia theo thời gian.

**Vì sao cần**: Slide đầu tiên bắt buộc phải có để audience hiểu bài toán — oil price là gì, dữ liệu trải dài bao lâu, và tại sao đây là bài toán classification (Up/Down).

**Insight cho thuyết trình**:

- Giá dầu biến động mạnh với nhiều regime: bull market, COVID crash (-24% trong 1 ngày), Ukraine crisis
- Volatility clustering rõ rệt: giai đoạn COVID và Ukraine var biến động gấp 3x bình thường
- Train/Test split theo thời gian (đường xanh lá) — KHÔNG random shuffle

**Hỗ trợ quyết định modeling**:

- Xác nhận bài toán là classification (Up/Down), không phải regression
- Gợi ý TimeSeriesSplit cho cross-validation (không random k-fold)
- Gợi ý cần xử lý regime change giữa train/test

---

### 02 — `02_target_class_balance.png`

**Mục đích**: Định nghĩa chính xác target variable và phân tích class balance.

**Vì sao cần**: Slide core nhất — audience cần hiểu target là gì (`direction = (oil_return > 0).astype(int)`) và class balance có vấn đề không trước khi đi vào modeling.

**Insight cho thuyết trình**:

- Target: `direction` = 1 (Up, ngày tăng) / 0 (Down, ngày giảm)
- Class balance: Down=48.6%, Up=51.4% → **cân bằng chấp nhận được**, không cần SMOTE
- Kurtosis=12 >> Normal (=3) → Fat tails: các sự kiện cực đoan xảy ra thường hơn phân phối chuẩn
- Overlap lớn giữa 2 class → bài toán phân loại khó (không có ranh giới rõ)

**Hỗ trợ quyết định modeling**:

- Không cần resampling đặc biệt (class balance acceptable)
- Nên dùng RobustScaler thay StandardScaler (fat tails)
- Metric: AUC-ROC + F1-weighted, KHÔNG chỉ dùng accuracy

---

### 03 — `03_target_over_time.png`

**Mục đích**: Cho thấy target (Up/Down ratio) thay đổi theo thời gian và theo regime thị trường.

**Vì sao cần**: Trong time-series classification, target không phải i.i.d. — tỷ lệ Up/Down thay đổi theo giai đoạn. Slide này truyền đạt sự phức tạp đó.

**Insight cho thuyết trình**:

- Giai đoạn Post-COVID recovery (2020–2022): Up-ratio ~57% → bull market
- COVID crash: Up-ratio giảm mạnh xuống ~48%
- Rolling 90-day smoothing cho thấy regime change rõ ràng
- Geopolitical stress tăng vọt trùng với giai đoạn biến động cao

**Hỗ trợ quyết định modeling**:

- Gợi ý thêm regime indicator feature (high/low volatility)
- Khuyến nghị TimeSeriesSplit để validate model trên nhiều regime
- Cảnh báo: model train trên 2015–2022 có thể kém trên test 2023+

---

### 04 — `04_data_quality.png`

**Mục đích**: Chứng minh dữ liệu đã sạch và giải thích tại sao outliers là bình thường trong tài chính.

**Vì sao cần**: Reviewer và giảng viên luôn hỏi "dữ liệu có vấn đề gì không?" Slide này trả lời trực tiếp và ngắn gọn.

**Insight cho thuyết trình**:

- 0 missing values, 0 duplicates, 0 INF — pipeline xử lý dữ liệu (step 1-6) hoạt động tốt
- Outliers theo IQR chủ yếu là supply features (net_imports_change_pct 20%, real_rate 19%)
- Đây là **fat-tail outliers tự nhiên** của dữ liệu tài chính, KHÔNG phải lỗi dữ liệu
- Nhiều features bị skewed nặng (net_imports, gdelt_tone) → cân nhắc log-transform

**Hỗ trợ quyết định modeling**:

- Dữ liệu sạch → không cần imputation
- Features skewed → log-transform nếu dùng linear model
- Outliers → dùng RobustScaler, KHÔNG StandardScaler

---

### 05 — `05_feature_signals.png`

**Mục đích**: Trả lời câu hỏi core: feature nào có tín hiệu dự báo tốt nhất cho target Up/Down?

**Vì sao cần**: Đây là slide quan trọng nhất cho modeling decision — giúp audience hiểu signal thực sự tồn tại (dù nhỏ).

**Insight cho thuyết trình**:

- **Top signals** (point-biserial r): `vix_return` (r=−0.21), `sp500_return` (r=+0.21), `usd_return` (r=−0.11)
- Dấu của r có ý nghĩa kinh tế: khi VIX tăng (risk-off) → giá dầu có xu hướng giảm; khi S&P500 tăng (risk-on) → giá dầu tăng
- Mutual Information cho thấy `sp500_return`, `usd_return`, `vix_return` cũng top non-linear signal
- Phân phối 2 class vẫn overlap nhiều (boxplot) → signal yếu nhưng có thật

**Hỗ trợ quyết định modeling**:

- Feature importance: ưu tiên vix_return, sp500_return, usd_return, sau đó lag features
- Chú ý: các features này là same-day → cân nhắc dùng lag1 versions trong strict setup
- Không nên kỳ vọng accuracy cao (>65%) — signal yếu là bình thường với financial data

---

### 06 — `06_timeseries_properties.png`

**Mục đích**: Chứng minh đặc tính time-series chính: random-walk của returns và volatility clustering.

**Vì sao cần**: Bất kỳ bài toán nào có yếu tố time-series đều cần trình bày tính chất này — nó ảnh hưởng trực tiếp đến kỳ vọng performance và lựa chọn model.

**Insight cho thuyết trình**:

- **ACF oil_return ≈ 0** ở tất cả lags → returns gần random walk → rất khó dự đoán (baseline thực)
- **ACF oil_volatility_7d mạnh** ở 20+ lags → volatility clustering → feature này có giá trị dự đoán cao
- Up-ratio có pattern nhẹ theo tháng và ngày trong tuần → seasonal features hữu ích
- Tháng 5, 8 có Up-ratio cao hơn → có thể thêm seasonal dummy features

**Hỗ trợ quyết định modeling**:

- Kỳ vọng realistic: accuracy ≈ 53–58% (do random walk)
- oil_volatility_7d là feature quan trọng — nên giữ hoặc tạo thêm GARCH-like features
- `month` và `day_of_week` có predictive value nhất định

---

### 07 — `07_split_leakage_check.png`

**Mục đích**: Xác nhận data split đúng và không có leakage, đồng thời cảnh báo distribution shift.

**Vì sao cần**: Slide bắt buộc về data integrity — nếu có leakage, toàn bộ kết quả model vô nghĩa. Distribution shift giữa train/test là rủi ro thực sự cần nêu.

**Insight cho thuyết trình**:

- Class ratio giữa train (51.4% Up) và test (49.8% Up) gần bằng nhau → split tốt, không có label drift
- **Distribution shift nghiêm trọng** ở macro features: `cpi_lag`, `real_rate`, `yield_spread` thay đổi hoàn toàn giữa train (2015–2022) và test (2023–2026) → inflation regime change
- **Leakage analysis**: `oil_return`/`direction` đã loại; `vix_return`/`sp500_return`/`usd_return` là same-day (medium risk) — nên dùng lag1 nếu cần strict setup
- 23/31 features có KS shift có ý nghĩa thống kê

**Hỗ trợ quyết định modeling**:

- Distribution shift → monitor model performance theo thời gian (không kỳ vọng stable)
- Strict leakage-free: thay vix_return/sp500_return/usd_return bằng \_lag1 versions
- Walk-forward validation thay vì one-shot train/test split

---

## Plots bị loại và lý do

| Plot bị loại                                     | Lý do                                                                            |
| ------------------------------------------------ | -------------------------------------------------------------------------------- |
| `correlation_heatmap.png`                        | 30×30 matrix — quá dense cho slide; key info đã được đưa vào plot 05             |
| `outlier_boxplots.png`                           | 30-panel grid — không đọc được trên slide; summary chart trong plot 04 đủ        |
| `hist_kde_market/macro/supply/sentiment/lag.png` | 5 plots grid — chi tiết hơn mức cần thiết cho overview; dùng trong phụ lục       |
| `acf_pacf_volatility.png`                        | Trùng thông tin với plot 06 (panel ACF volatility); dùng 1 combined plot tốt hơn |
| `returns_by_class.png`                           | Đã integrated vào plot 02 (target class balance); không cần standalone           |
| `rolling_statistics.png`                         | Thông tin chồng lên với plot 01 và 03; không thêm insight mới                    |
| `rolling_correlation.png`                        | Quá chi tiết cho overview slide; có thể dùng trong phụ lục                       |
| `seasonality_month.png`                          | Đã integrated vào plot 06; standalone version kém compact hơn                    |
| `streak_distribution.png`                        | Insight hữu ích nhưng không critical cho decision-making; phụ lục                |
| `categorical_features.png`                       | Distribution đều → không có insight mạnh; có thể dùng một câu giải thích         |
| `train_test_distribution_shift.png`              | 9-panel KDE grid — đã được summarize trong plot 07 gọn hơn                       |
| `class_ratio_train_vs_test.png`                  | Đã integrated vào plot 07; standalone version thiếu context                      |
| `feature_vs_target_boxplots.png`                 | 12-panel grid — đã integrated gọn hơn vào plot 05                                |

---

## Lưu ý cho thuyết trình

- **Thứ tự logic**: Overview → Target → Behavior-over-time → Quality → Signal → TS-properties → Integrity
- **Kỳ vọng realistic**: Nêu rõ accuracy ceiling ~53–58% vì market gần random walk
- **Key message**: Signal có thật (r≈0.2 cho vix/sp500) nhưng nhỏ → cần feature engineering kỹ
- **Risk transparency**: Distribution shift macro rất lớn → đây là rủi ro thực sự cần acknowledge
