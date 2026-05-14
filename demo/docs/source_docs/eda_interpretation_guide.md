# Hướng dẫn Đọc, Hiểu và Diễn giải Kết quả EDA

## Dự án Dự đoán Xu hướng Tăng/Giảm Giá Dầu

> Tài liệu này hướng dẫn cách đọc và diễn giải các kết quả EDA (Exploratory Data Analysis) trong dự án dự đoán xu hướng giá dầu. Nội dung đi từ cơ bản đến nâng cao, gắn liền với bối cảnh dữ liệu tài chính / hàng hóa.

---

## Mục lục

1. [Tổng quan về EDA](#1-tổng-quan-về-eda)
2. [Thống kê mô tả cơ bản](#2-thống-kê-mô-tả-cơ-bản)
3. [Cách đọc Histogram và KDE](#3-cách-đọc-histogram-và-kde)
4. [Cách đọc Boxplot](#4-cách-đọc-boxplot)
5. [Cách đọc Bar Chart](#5-cách-đọc-bar-chart)
6. [Cách đọc Line Chart (Time Series)](#6-cách-đọc-line-chart-time-series)
7. [Cách đọc Heatmap và Correlation Matrix](#7-cách-đọc-heatmap-và-correlation-matrix)
8. [Cách đọc phân phối Feature](#8-cách-đọc-phân-phối-feature)
9. [Cách phát hiện Missing Values bất thường](#9-cách-phát-hiện-missing-values-bất-thường)
10. [Cách nhìn Outlier trong bối cảnh tài chính](#10-cách-nhìn-outlier-trong-bối-cảnh-tài-chính)
11. [Cách đọc quan hệ Feature–Target](#11-cách-đọc-quan-hệ-feature-target)
12. [Cách đọc biểu đồ theo thời gian](#12-cách-đọc-biểu-đồ-theo-thời-gian)
13. [Cách nhận biết Data Leakage](#13-cách-nhận-biết-data-leakage)
14. [Từ EDA sang Quyết định Modeling](#14-từ-eda-sang-quyết-định-modeling)

---

## 1. Tổng quan về EDA

**EDA là gì?**
EDA (Exploratory Data Analysis) là quá trình khám phá dữ liệu bằng các phương pháp thống kê và visualization nhằm:

- Hiểu cấu trúc và đặc tính dữ liệu.
- Phát hiện patterns, anomalies, và mối quan hệ.
- Đưa ra quyết định cho bước tiếp theo (feature engineering, model selection).

**Trong bài toán của chúng ta:**

- Dữ liệu: daily oil prices + macro/sentiment/geopolitical từ nhiều nguồn.
- Target: dự đoán ngày mai giá dầu **tăng** hay **giảm** (binary classification).
- **Quy tắc số 1**: EDA chỉ được thực hiện trên tập **TRAIN** (trước 2023-01-01) để tránh rò rỉ thông tin từ tương lai.

---

## 2. Thống kê mô tả cơ bản

Khi nhìn vào bảng `describe()`, chú ý:

| Thống kê            | Ý nghĩa            | Ví dụ trong project                                        |
| ------------------- | ------------------ | ---------------------------------------------------------- |
| **count**           | Số quan sát        | 2083 (train set)                                           |
| **mean**            | Giá trị trung bình | `oil_return` mean ≈ 0.0006 → returns trung bình gần 0      |
| **std**             | Độ lệch chuẩn      | `oil_return` std ≈ 0.026 → biến động ~2.6% mỗi ngày        |
| **min / max**       | Giá trị cực đoan   | min = -0.24 → ngày giảm mạnh nhất ~24%                     |
| **25% / 50% / 75%** | Quartiles          | Median (50%) ≈ 0.0006 → gần bằng mean → tương đối đối xứng |

**Cách đọc nhanh:**

- `mean ≈ median` → phân phối khá đối xứng.
- `mean >> median` (hoặc ngược lại) → phân phối bị lệch (skewed).
- `std` lớn so với `mean` → dữ liệu biến động mạnh (typical cho tài chính).

### Skewness (Độ lệch)

- **skewness ≈ 0**: phân phối đối xứng.
- **skewness > 0**: lệch phải (đuôi dài bên phải) — nhiều giá trị nhỏ, ít giá trị lớn.
- **skewness < 0**: lệch trái (đuôi dài bên trái) — ví dụ `oil_return` thường có skewness âm, nghĩa là có nhiều ngày giảm mạnh bất thường hơn ngày tăng mạnh.
- **|skewness| > 1**: skewed đáng kể → cân nhắc transform.

### Kurtosis (Độ nhọn)

- **kurtosis ≈ 3** (excess kurtosis ≈ 0): giống phân phối chuẩn.
- **kurtosis > 3**: leptokurtic (fat tails) — đuôi phân phối dày hơn normal.
  - _Ví dụ_: `oil_return` kurtosis ≈ 12 → rất leptokurtic → những ngày biến động cực đoan xảy ra thường hơn mô hình Normal dự đoán. Đây là **đặc chưng điển hình của dữ liệu tài chính**, không phải lỗi dữ liệu.
- **kurtosis < 3**: platykurtic (thin tails) — ít sự kiện cực đoan.

---

## 3. Cách đọc Histogram và KDE

### Histogram

- Trục X: giá trị feature, chia thành bins.
- Trục Y: tần suất (số observations trong mỗi bin).
- **Hình dạng cho biết phân phối**:
  - **Bell-shaped**: gần normal.
  - **Skewed right**: đuôi dài bên phải → nhiều giá trị nhỏ.
  - **Bimodal** (2 đỉnh): có thể có 2 regime/nhóm khác nhau.

### KDE (Kernel Density Estimation)

- Đường cong mượt **xấp xỉ density** từ histogram.
- Diện tích dưới đường KDE = 1.
- Dễ so sánh phân phối giữa 2 nhóm khi overlay lên nhau.

**Ví dụ minh họa trong project:**

- Histogram `oil_return`: bell-shaped nhưng có fat tails → bình thường cho financial returns.
- KDE chồng `oil_return` theo `direction` (0 vs 1):
  - 2 đường KDE gần trùng nhau → **khó phân tách** 2 class → bài toán prediction khó.
  - 2 đường KDE tách biệt rõ ràng → dễ phân tách → bài toán dễ hơn (hiếm gặp với financial data).

---

## 4. Cách đọc Boxplot

```
Outlier            *
                   |
Upper Whisker  ----+----  Q3 + 1.5 × IQR
                   |
   Q3          --------
                |      |
   Median      |------|   ← Đường ngang giữa hộp
                |      |
   Q1          --------
                   |
Lower Whisker  ----+----  Q1 - 1.5 × IQR
                   |
Outlier            *
```

- **IQR** = Q3 - Q1 (chiều cao hộp).
- **Whiskers**: kéo dài tới giá trị xa nhất trong phạm vi 1.5 × IQR.
- **Dấu chấm ngoài whiskers**: outliers.

**Trong bối cảnh tài chính:**

- Boxplot `oil_return` sẽ có **rất nhiều outliers** (dấu chấm) — đây là chuyện bình thường, KHÔNG nên loại.
- Boxplot theo `direction` (0 vs 1): nếu median của 2 hộp gần nhau → feature kém phân biệt.
- Boxplot theo `month`: nếu có tháng median cao hơn rõ rệt → có seasonality.

---

## 5. Cách đọc Bar Chart

Dùng cho biến **categorical/discrete**:

- `day_of_week` (0=Mon → 4=Fri): bar chart cho biết ngày nào có nhiều dữ liệu hơn (nên đều).
- `direction` (0 = giảm, 1 = tăng): bar chart cho biết **class balance**.
  - 2 bar gần bằng nhau → **balanced** → tốt.
  - 1 bar cao hơn rõ rệt → **imbalanced** → cần strategy đặc biệt.

**Ví dụ**: Nếu class "tăng" chiếm 55%, class "giảm" chiếm 45% → mildly imbalanced → vẫn OK nhưng nên dùng F1 hoặc AUC thay vì chỉ accuracy.

---

## 6. Cách đọc Line Chart (Time Series)

### Line chart đơn giản

- Trục X: thời gian (date).
- Trục Y: giá trị feature.
- **Trend**: feature tăng/giảm dần theo thời gian?
- **Volatility clustering**: có giai đoạn biến động mạnh xen kẽ giai đoạn yên tĩnh?

**Ví dụ trong project:**

- Line chart `oil_return`: thấy rõ COVID crash (3/2020) là spike cực lớn.
- Line chart `oil_volatility_7d`: thấy volatility clustering — giai đoạn 2020 biến động mạnh, sau đó giảm, rồi tăng lại khi chiến sự Ukraine.

### Rolling Mean / Rolling Std

- **Rolling mean** (trung bình trượt): cho thấy trend cục bộ, bỏ qua nhiễu ngắn hạn.
- **Rolling std** (độ lệch chuẩn trượt): cho thấy biến động cục bộ.
- Nếu rolling std thay đổi mạnh theo thời gian → **heteroscedasticity** → model cần account for this.

### ACF (Autocorrelation Function)

- Trục X: lag (1, 2, 3, ..., N).
- Trục Y: correlation giữa time series và bản thân nó dịch đi lag bước.
- **Ý nghĩa**:
  - ACF lag=1 cao → giá trị ngày hôm nay liên quan mạnh đến hôm qua.
  - ACF giảm nhanh về 0 → stationary.
  - ACF giảm chậm → non-stationary hoặc có long memory.
- **Đường blue band**: confidence interval 95%. Bar nằm trong band → không significant.

**Ví dụ**: ACF `oil_return` thường không significant → returns gần random walk → dự đoán khó.

### PACF (Partial Autocorrelation Function)

- Giống ACF nhưng **loại bỏ ảnh hưởng của các lag trung gian**.
- PACF significant ở lag k → có direct relationship giữa t và t-k (không qua t-1, t-2, ...).
- Dùng để xác định bậc AR (AutoRegressive) trong mô hình time series.

---

## 7. Cách đọc Heatmap và Correlation Matrix

### Heatmap Correlation

- Ma trận NxN (N = số features).
- Mỗi ô: correlation giữa 2 features.
- Màu sắc: đỏ = positive correlation, xanh = negative correlation, trắng = không tương quan.

**Cách đọc:**

- Đường chéo luôn = 1 (feature correlate với chính nó).
- **|r| > 0.7**: highly correlated → cặp features này có thể redundant → cân nhắc loại 1.
- **|r| < 0.1**: hầu như không tương quan.
- **Cụm đỏ/xanh**: nhóm features correlate với nhau → multicollinearity.

**Chú ý Spearman vs Pearson:**

- **Pearson**: đo tương quan **tuyến tính**. Nhạy cảm với outliers.
- **Spearman**: đo tương quan **đơn điệu** (monotonic). Robust hơn với outliers → **nên dùng cho financial data**.

**Ví dụ trong project:**

- `gdelt_tone_7d` và `gdelt_tone_30d` có correlation rất cao → redundant → chỉ nên giữ 1.
- `oil_return_lag1` và `oil_return` có correlation thấp → lag1 có ít thông tin dự đoán.

---

## 8. Cách đọc phân phối Feature

Khi phân tích phân phối một feature, kiểm tra:

1. **Shape**: Normal? Skewed? Bimodal?
2. **Center**: Mean và Median ở đâu?
3. **Spread**: Std lớn hay nhỏ? IQR?
4. **Outliers**: Có giá trị cực đoan? Bao nhiêu %?
5. **Bounded**: Feature có giới hạn tự nhiên không? (ví dụ z-score không có giới hạn, nhưng `day_of_week` ∈ [0,4]).

**Red flags:**

- Feature hầu như constant (std ≈ 0) → useless → loại.
- Feature có >90% giá trị giống nhau → ít thông tin.
- Feature có phân phối rất khác giữa train/test → distribution shift.

---

## 9. Cách phát hiện Missing Values bất thường

### Loại missing values

1. **MCAR** (Missing Completely At Random): ngẫu nhiên → ít ảnh hưởng.
2. **MAR** (Missing At Random): phụ thuộc vào biến khác → cần xử lý cẩn thận.
3. **MNAR** (Missing Not At Random): missing có pattern → quan trọng!

**Cách phát hiện:**

- Heatmap missing: nếu nhiều cột missing cùng lúc (cùng hàng) → MAR hoặc MNAR.
- Missing tập trung ở đầu dataset → dữ liệu chưa available ở thời kỳ đầu (bình thường cho rolling features).
- Missing theo ngày nhất định → có thể do ngày lễ, dữ liệu nguồn không cập nhật.

**Trong project này:**

- `dataset_final.csv` không nên có missing (đã xử lý ở step 2-5).
- Nếu phát hiện missing → cần quay lại kiểm tra pipeline.

---

## 10. Cách nhìn Outlier trong bối cảnh tài chính

> **Quy tắc quan trọng**: Trong dữ liệu tài chính, outlier thường là **thông tin, không phải lỗi**.

### Outlier tự nhiên vs Outlier lỗi

- **Outlier tự nhiên**: COVID crash (oil_return ~ -24%), ngày Fed tăng lãi suất bất ngờ → **GIỮ NGUYÊN**.
- **Outlier lỗi**: giá trị vô lý (ví dụ return = -500%), dữ liệu bị nhập sai → **CẦN SỬA**.

**Cách phân biệt:**

1. Kiểm tra **ngày** xảy ra outlier → nếu trùng sự kiện lớn → tự nhiên.
2. Kiểm tra **nhiều features** cùng lúc → nếu nhiều features extreme cùng ngày → sự kiện market, không phải lỗi.
3. Kiểm tra **giá trị có hợp lý** → return -24% khác với return -2400%.

**Cách xử lý:**

- **Winsorize** (cắt ở percentile 1%/99%): giảm extreme values mà không loại bỏ.
- **RobustScaler**: scale dùng median/IQR thay vì mean/std → ít bị ảnh hưởng bởi outliers.
- **KHÔNG nên loại outliers** trong financial data trừ khi chắc chắn là lỗi.

---

## 11. Cách đọc quan hệ Feature–Target

### Cho bài toán Classification (target là binary: 0/1)

**Boxplot feature theo class:**

- 2 boxplot cạnh nhau (class 0, class 1).
- Nếu 2 hộp **chồng lấn nhiều** → feature kém phân biệt.
- Nếu 2 hộp **tách biệt rõ** → feature tốt.

**Point-biserial correlation:**

- Tương quan giữa feature liên tục và target binary.
- r > 0: feature tăng → class 1 (tăng) nhiều hơn.
- r < 0: feature tăng → class 0 (giảm) nhiều hơn.
- |r| > 0.1: có signal (trong tài chính, r > 0.1 là khá tốt!).
- |r| < 0.05: hầu như không có signal.

**Mann-Whitney U test:**

- Non-parametric test so sánh phân phối 2 nhóm.
- p-value < 0.05: 2 nhóm khác biệt có ý nghĩa → feature có thể hữu ích.
- p-value > 0.05: 2 nhóm không khác biệt đáng kể.

**Mutual Information (MI):**

- Đo lượng thông tin feature cung cấp về target.
- MI = 0: feature hoàn toàn independent với target.
- MI càng cao → feature càng hữu ích.
- Ưu điểm: bắt được cả quan hệ **phi tuyến** (non-linear).

**Thực tế trong project:**

- Đừng kỳ vọng correlation cao (>0.3). Trong financial prediction, |r| = 0.05-0.15 đã là signal hữu ích!
- Nhiều features có MI rất thấp → bình thường → market gần efficient.

---

## 12. Cách đọc biểu đồ theo thời gian

### Rolling Correlation

- Thể hiện correlation giữa 2 features **thay đổi theo thời gian**.
- Ý nghĩa: mối quan hệ giữa features không cố định.
- **Ví dụ**: correlation giữa `oil_return` và `sp500_return`:
  - Trong khủng hoảng: correlation tăng (everything sells off together).
  - Trong thời kỳ bình thường: correlation thấp hơn.
  - → Gợi ý: model nên **account for regime** (varying relationships).

### Streak Analysis

- Streak = chuỗi liên tiếp cùng direction (tăng liền hoặc giảm liền).
- Average streak length ≈ 2 → market gần random walk.
- Long streaks (>5) → momentum effect.
- Histogram streak length → nếu skewed → market có bias.

### Regime Detection

- Phân chia thời gian thành các regime (low vol, high vol, Fed hiking, etc.)
- So sánh class ratio trong mỗi regime:
  - Low vol regime: tăng 55% → slight bull bias.
  - High vol regime: tăng 48% → slight bear bias.
  - → Model nên biết "hiện tại đang ở regime nào" để dự đoán tốt hơn.

---

## 13. Cách nhận biết Data Leakage

**Data leakage** xảy ra khi model vô tình sử dụng thông tin mà **tại thời điểm dự đoán, chưa có sẵn**.

### Dấu hiệu leakage:

1. **Feature có correlation quá cao với target** (|r| > 0.5 cho financial data) → nghi ngờ!
2. **Model accuracy quá cao** (>70% cho daily stock/oil direction) → gần như chắc chắn có leakage.
3. **Same-day features**: ví dụ dùng `vix_close` ngày t để predict direction ngày t → nhưng khi predict, chưa biết VIX ngày t!
4. **Normalize/scale trên toàn bộ dataset** (bao gồm test) → information từ test leak vào train.

### Loại leakage thường gặp trong project này:

**a) Same-day information leak:**

- `vix_close` cùng ngày → PHẢI dùng `vix_lag1` (ngày hôm trước).
- `sp500_close` cùng ngày → tương tự.
- **Quy tắc**: feature tại thời điểm t chỉ được dùng thông tin đến t-1.

**b) Look-ahead bias trong rolling features:**

- Rolling mean/std tính cho ngày hôm nay không được bao gồm ngày hôm nay.
- `centered` rolling → leakage! Phải dùng `backward` rolling.

**c) Target leakage:**

- Nếu `oil_return[t]` (return ngày t) nằm trong feature set khi predict `direction[t]` → leakage trực tiếp!
- Feature set chỉ nên chứa `oil_return_lag1`, `oil_return_lag2`.

**d) Test information leak vào preprocessing:**

- Nếu StandardScaler fit trên toàn bộ data (train + test) → mean/std bao gồm thông tin test.
- Phải fit scaler **chỉ trên train**, sau đó transform test.

### Cách kiểm tra:

1. Liệt kê tất cả features → xác định "available when?" cho mỗi feature.
2. Chạy model với/không feature nghi ngờ → nếu accuracy giảm mạnh khi loại → khả năng leakage.
3. Check pipeline: rolling/normalize fit trên dataset nào?

---

## 14. Từ EDA sang Quyết định Modeling

EDA không chỉ để "nhìn cho vui" — mỗi observation nên dẫn đến một quyết định cụ thể:

### Pattern → Quyết định

| Observation EDA                   | Quyết định Modeling                                     |
| --------------------------------- | ------------------------------------------------------- |
| Class imbalanced (60:40)          | Dùng `class_weight='balanced'` hoặc SMOTE               |
| Returns gần random walk (ACF ≈ 0) | Kỳ vọng accuracy realistic (~53-58%)                    |
| Volatility clustering mạnh        | Thêm GARCH features hoặc regime indicator               |
| Seasonality theo tháng            | Giữ feature `month`, hoặc tạo seasonal dummy            |
| 2 features correlation > 0.9      | Loại 1 trong 2 (giữ cái có MI cao hơn)                  |
| VIF > 10                          | Loại feature hoặc dùng PCA                              |
| Distribution shift train/test     | Dùng validation set gần test, hoặc retrain periodically |
| Feature MI ≈ 0 với target         | Loại khỏi model                                         |
| Leakage detected                  | SỬA NGAY — kết quả vô nghĩa nếu có leakage              |
| Fat tails (kurtosis cao)          | Dùng RobustScaler, cân nhắc winsorize                   |
| Feature skewed                    | Log-transform hoặc Box-Cox trước khi vào model          |

### Quy trình chuyển tiếp

```
EDA Findings
    │
    ├─→ Feature Selection
    │       ↳ Giữ features: MI > 0, MW significant, low leakage risk
    │       ↳ Loại features: VIF > 10, MI ≈ 0, leakage
    │
    ├─→ Feature Engineering
    │       ↳ Transform: log-transform skewed features
    │       ↳ Scale: RobustScaler cho fat-tailed data
    │       ↳ New features: interaction, regime indicators
    │
    ├─→ Model Selection
    │       ↳ Start simple: Logistic Regression (baseline)
    │       ↳ Then try: Random Forest, XGBoost, LightGBM
    │       ↳ CV: TimeSeriesSplit (KHÔNG random k-fold!)
    │
    ├─→ Evaluation Strategy
    │       ↳ Metric: F1, AUC-ROC (NOT just accuracy)
    │       ↳ Benchmark: "always predict majority class" → baseline accuracy
    │
    └─→ Risk Management
            ↳ Regime change: monitor performance over time
            ↳ Low signal: set realistic expectations
            ↳ Overfitting: regularize, cross-validate
```

---

## Phụ lục: Chú giải thuật ngữ

| Thuật ngữ      | Ý nghĩa                                                |
| -------------- | ------------------------------------------------------ |
| ACF            | Autocorrelation Function — hàm tự tương quan           |
| ADF test       | Augmented Dickey-Fuller — kiểm tra tính dừng           |
| Classification | Bài toán phân loại — dự đoán nhãn (tăng/giảm)          |
| Fat tails      | Đuôi phân phối dày — nhiều giá trị cực đoan hơn Normal |
| IQR            | Interquartile Range = Q3 - Q1                          |
| KDE            | Kernel Density Estimation — ước lượng mật độ           |
| Leakage        | Rò rỉ dữ liệu — dùng thông tin tương lai để dự đoán    |
| MI             | Mutual Information — thông tin tương hỗ                |
| PACF           | Partial Autocorrelation Function                       |
| Point-biserial | Tương quan giữa biến liên tục và biến nhị phân         |
| Skewness       | Độ lệch của phân phối                                  |
| Kurtosis       | Độ nhọn (fat tails) của phân phối                      |
| Stationary     | Tính dừng — mean/variance không đổi theo thời gian     |
| VIF            | Variance Inflation Factor — đo đa cộng tuyến           |
| Winsorize      | Cắt giá trị cực đoan ở percentile nhất định            |
