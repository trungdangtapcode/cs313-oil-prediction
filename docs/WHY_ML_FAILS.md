# 🔍 Tại sao ML không hoạt động tốt trên bộ dữ liệu này?

## Tóm tắt kết luận

Mô hình đạt **57.4% accuracy** (vs 50.2% baseline random) — thực tế đây là **kết quả hợp lý, không phải thất bại**. Vấn đề không nằm ở model hay code, mà nằm ở **bản chất của dữ liệu và thị trường**. Dưới đây là 7 nguyên nhân gốc rễ, được chứng minh bằng số liệu thực từ dataset.

---

## Nguyên nhân 1: Tỉ lệ Signal-to-Noise cực thấp

> ⚠️ Đây là nguyên nhân **quan trọng nhất** — mọi nguyên nhân khác đều phát sinh từ đây.

```
oil_return_fwd1 (target):
  Mean:  0.000562    (gần 0 — hầu như không có xu hướng rõ)
  Std:   0.024530    (biến động lớn — noise rất cao)
  
  Signal-to-Noise = |Mean| / Std = 0.0229
  ~~> Tín hiệu chỉ chiếm 2.3% so với noise
```

### Ý nghĩa thực tế

| Thống kê | Giá trị | Hậu quả |
|----------|---------|---------|
| % ngày return nằm trong ±0.5% | **26.2%** | 1/4 ngày gần như "zero return" — không thể phân loại |
| % ngày return nằm trong ±1% | **45.5%** | Gần nửa dataset là noise thuần túy |
| Kurtosis | **11.9** | Phân phối heavy-tailed — xuất hiện extreme events bất thường |
| Skew | **-0.37** | Hơi lệch trái — crash days xảy ra nhanh và mạnh hơn rally days |
| t-test (mean ≠ 0) | **p = 0.22** | **Không thể khẳng định** mean khác 0 — target gần như không có xu hướng |

### So sánh với các lĩnh vực ML khác

```
Lĩnh vực              Signal-to-Noise    Accuracy thông thường
──────────────────────────────────────────────────────────────
Image recognition      Rất cao              95-99%
NLP / Sentiment        Cao                  85-95%
Medical diagnosis      Cao                  80-95%
Credit scoring         Trung bình           75-85%
Stock/Oil direction    CỰC THẤP (0.02)     52-60%
```

**Marcos López de Prado** (Cornell, "Advances in Financial ML") nhận định: *"Financial ML khác biệt hoàn toàn với các lĩnh vực ML khác vì tỉ lệ tín hiệu cực thấp. ML algorithm sẽ luôn tìm ra pattern, kể cả khi không có — dẫn đến overfitting."*

---

## Nguyên nhân 2: Giá dầu tuân theo Random Walk

Kiểm định thống kê cho thấy oil returns **không thể phân biệt với chuỗi ngẫu nhiên**:

### Runs Test (Kiểm định tính ngẫu nhiên)

```
z = 1.462, p = 0.1437
Actual runs: 1501, Expected (random): 1461.5
→ KHÔNG THỂ bác bỏ giả thuyết random walk
```

### Variance Ratio Test

```
k= 2: VR = 1.0155    (random walk = 1.0)
k= 5: VR = 1.0261
k=10: VR = 1.0084
k=20: VR = 0.9343
→ Tất cả rất gần 1.0 — hành vi random walk
```

### Autocorrelation (Tự tương quan)

```
Lag  1: +0.0152  (gần 0)
Lag  2: -0.0030
Lag  3: -0.0109
Lag  5: -0.0079
Lag 10: +0.0137
Lag 20: +0.0250
→ KHÔNG CÓ autocorrelation có ý nghĩa
```

### Tỷ lệ đổi dấu hàng ngày

```
Sign change rate: 0.5505  (expected if random: 0.50)
→ Hướng giá hôm nay KHÔNG dự đoán được hướng giá ngày mai
```

> **Kết quả trên nghĩa là**: return ngày T+1 gần như độc lập hoàn toàn với mọi thông tin có tại ngày T. Bất kỳ model nào dự đoán tốt hơn 50% đều đang khai thác một tín hiệu cực kỳ yếu.

---

## Nguyên nhân 3: Features gần như không tương quan với Target

Đây là phát hiện **nghiêm trọng nhất** từ mặt kỹ thuật:

### Correlation với `oil_return_fwd1` (forward target)

```
KHÔNG CÓ feature nào có |Spearman| > 0.05
KHÔNG CÓ feature nào có |Spearman| > 0.10

Feature tốt nhất: gdelt_tone_30d  với |Spearman| = 0.0462
```

| Feature | Spearman | Ý nghĩa |
|---------|----------|---------|
| gdelt_tone_30d | +0.046 | Feature #1 — chỉ giải thích ~0.2% variance |
| gdelt_volume_7d | -0.045 | |
| stress_tone | -0.043 | |
| gdelt_tone_lag1 | +0.041 | |
| usd_close | -0.039 | |
| sp500_return | +0.031 | Feature "mạnh nhất" trong EDA trước — nhưng là contemporaneous! |
| oil_return_lag1 | -0.027 | |

> **Lưu ý quan trọng**: Trong EDA classification, `sp500_return` có **|ρ| = 0.26** với `oil_return` (same-day). Nhưng khi tính correlation với **forward target** (`oil_return_fwd1` — ngày T+1), nó giảm còn **|ρ| = 0.031** — giảm **8 lần**!
>
> Điều này chứng minh: **correlation với same-day return ≠ khả năng dự đoán return ngày mai**.

### Cohen's d (Effect Size giữa UP vs DOWN)

```
|d| > 0.2 (small effect):  Chỉ 3/52 features
|d| > 0.5 (medium effect): 0/52 features
|d| > 0.8 (large effect):  0/52 features

→ KHÔNG CÓ feature nào tạo sự khác biệt có ý nghĩa
  giữa ngày tăng và ngày giảm
```

### Mutual Information

```
Top MI với target:
  sp500_return:    0.045
  vix_return:      0.043
  usd_return:      0.027

  29/52 features có MI = 0.000
```

**Kết luận**: Dù ta có dùng model ML phức tạp đến đâu, thì **đầu vào không chứa đủ thông tin dự đoán đầu ra**. Đây là bài toán "garbage in, garbage out" — nhưng không phải do data quality kém, mà do **thị trường hiệu quả** đã loại bỏ hầu hết tín hiệu dự đoán khỏi dữ liệu công khai.

---

## Nguyên nhân 4: Distribution Shift nghiêm trọng giữa Train/Test

**41/53 features có phân phối thay đổi đáng kể** giữa train (2015-2022) và test (2023-2026):

### Worst offenders

| Feature | KS stat | Mô tả vấn đề |
|---------|---------|---------------|
| fed_funds_rate_lag | **0.989** | Fed rate: 0-2.4% (train) → 4.3-5.5% (test) — regime hoàn toàn khác |
| cpi_lag | **0.989** | CPI tăng mạnh sau COVID |
| crude_inventory_weekly | **0.961** | Mức tồn kho thay đổi structural |
| crude_production_weekly | **0.869** | Sản lượng tăng liên tục |
| real_rate | **0.862** | Lãi suất thực: -7% → +2% |
| sp500_close | **0.807** | S&P500 tăng 40% |
| cpi_yoy | **0.620** | Lạm phát 1-2% → 3-9% |
| gdelt_tone_30d | **0.471** | Tone tin tức thay đổi |

```
Train (2015-2022): COVID crash, zero rates, oil war, recovery
Test  (2023-2026): High rates, post-inflation, AI boom, geopolitical tensions
→ Hai giai đoạn market regime HOÀN TOÀN KHÁC NHAU
```

> ⚠️ Model học patterns từ regime lãi suất 0% (2015-2022) → áp dụng vào regime 5%+ (2023-2026). Các mối quan hệ giữa feature-target đã thay đổi cơ bản.

### Target cũng shift

```
Train: mean=0.000598, std=0.02628  (volatile, COVID era)
Test:  mean=0.000471, std=0.01954  (calmer)
Train UP%: 51.3%
Test  UP%: 49.8%
```

Tuy KS test (p=0.14) cho thấy target không shift đáng kể, nhưng **volatility giảm 26%** — model train trên data volatile hơn sẽ overreact trong test period.

---

## Nguyên nhân 5: Đa cộng tuyến nặng

EDA phát hiện **34 cặp features có |ρ| > 0.80** và **22 features có VIF > 10**:

```
Các cụm cộng tuyến chính:
• gdelt_tone ≈ stress_tone ≈ gdelt_tone_7d ≈ gdelt_tone_lag1    (|ρ|=1.0)
• sp500_close ≈ cpi_lag ≈ net_imports_weekly ≈ GDELT volumes    (|ρ|=1.0)
• wti_fred ≈ crude_inventory ≈ cpi_yoy ≈ MA features ≈ BB       (|ρ|=0.997)
```

### Hậu quả

- **Feature importance bị chia nhỏ**: 5 features cùng cụm sẽ chia nhau importance, che giấu tín hiệu thật
- **Model instability**: Gradient-based models không ổn định khi features collinear
- **Overfitting**: Model dễ fit vào noise khi có nhiều feature redundant
- **Feature selection bị nhiễu**: MI/Spearman ranking đánh giá từng feature riêng lẽ — bỏ sót interactions, không loại được redundancy

---

## Nguyên nhân 6: Dataset quá nhỏ

```
Train samples: 2,083
Test samples:  840
Features:      42-81 (tùy step)

Samples-per-feature ratio: ~25-50
→ Rất thấp cho tree-based models
→ CỰC THẤP cho neural networks
```

### So sánh

| Dataset | Samples | Features | Ratio |
|---------|---------|----------|-------|
| Oil Price (dự án này) | 2,083 | 42-81 | 25-50 |
| Kaggle TabNet benchmarks | 50,000+ | 50-200 | 250-1000 |
| Financial research (typical) | 5,000-50,000 | 20-100 | 100-1000 |

### Tác động thực tế

- **Cross-validation instability**: Với 5-fold TimeSeriesSplit, mỗi fold chỉ ~400 samples → variance rất cao
- **Tuning overfits**: RandomizedSearchCV 50 iterations → CV accuracy 61% nhưng test chỉ 52-55%
- **Permutation importance nhiễu**: Test set chỉ 840 samples → permutation importance không đáng tin cậy
- **Weight decay không hiệu quả**: Dataset đã nhỏ, decay thêm → càng ít effective samples

---

## Nguyên nhân 7: Efficient Market Hypothesis — Giới hạn cơ bản

Thị trường dầu thế giới là một trong những thị trường **liquid nhất** (volume >$100B/ngày):

```
Thông tin mới xuất hiện (geopolitical, macro, supply)
    ↓
Trader/Algo phản ứng trong milliseconds-minutes
    ↓
Giá điều chỉnh ngay lập tức (arbitrage closes gap)
    ↓
Data ta quan sát (giá đã phản ánh thông tin)
    ↓
ML model cố dự đoán từ thông tin đã "cũ"
    ↓
Signal đã bị arbitrage → accuracy ~50%
```

### Tại sao features công khai không có predictive power

| Feature category | Vấn đề |
|-----------------|---------|
| **Macro (FRED)** | CPI, Fed rate → thị trường đã price in TRƯỚC khi data công bố. Fed futures cho biết kỳ vọng lãi suất months ahead |
| **Supply (EIA)** | Inventory report → trader phản ứng **trong phút đầu** khi data release, trước khi row data xuất hiện trong dataset |
| **Geopolitical (ACLED)** | Conflict events → market đã phản ứng qua tin tức real-time, trước khi ACLED ghi nhận |
| **Sentiment (GDELT)** | Media tone → phản ánh **consequence** của giá, không phải **cause**. Giá giảm → tin xấu tăng (reverse causality) |
| **Technical (lagged)** | MA, RSI, MACD (shifted) → đã bị stripped khỏi predictive content do shifting |

> **Vicious cycle**: Feature công khai → trader đã khai thác → signal bị arbitrage → feature mất khả năng dự đoán

---

## Bức tranh tổng thể

```
DATA QUALITY      ✅ Tốt (0 missing, 0 duplicates)
PIPELINE          ✅ Đúng (no leakage sau fix)
FEATURE ENG       ✅ Hợp lý (returns, rolling, lag, stress)
MODEL CHOICE      ✅ Đa dạng (8+ models, ensemble)
TUNING            ✅ Cẩn thận (TimeSeriesSplit, no leakage)

╔═══════════════════════════════════════════╗
║  VẤN ĐỀ GỐC: THỊ TRƯỜNG HIỆU QUẢ       ║
║  → Signal-to-noise = 0.023               ║
║  → Random walk (p=0.14)                  ║
║  → Feature-target corr < 0.05            ║
║  → 78% features shift giữa train/test    ║
║  → Dataset chỉ 2,083 samples             ║
╚═══════════════════════════════════════════╝

→ 57.4% accuracy (+7.2% vs random) là kết quả HỢP LÝ
```

---

## So sánh với Academic Literature

Các nghiên cứu academic về dự đoán giá dầu hàng ngày:

| Paper/Approach | Accuracy | Ghi chú |
|----------------|----------|---------|
| Random baseline | 50% | — |
| **Dự án này (GBM + TOP_50)** | **57.4%** | **Honest evaluation, no leakage** |
| Typical daily oil ML (academic, shifted) | 52-58% | Tùy period và feature set |
| "High accuracy" papers (80%+) | 80-90% | Thường có leakage hoặc same-day features |
| Monthly/Quarterly prediction | 60-70% | Horizon dài hơn → dễ hơn daily |

> Khi đọc paper claim >70% accuracy cho daily oil prediction, hãy kiểm tra:
> 1. Same-day features? (leakage)
> 2. Random split thay vì time split? (leakage)
> 3. In-sample accuracy thay vì out-of-sample?
> 4. Period nào? (Trend period dễ hơn)

---

## Hướng cải thiện tiềm năng (nếu muốn tiếp tục)

### Có thể thử

| Hướng | Reasoning | Khó khăn |
|-------|-----------|----------|
| **Alternative data** | Satellite imagery, tanker tracking, options flow, Twitter sentiment real-time | Cost, accessibility |
| **Triple-Barrier Labeling** (López de Prado) | Thay binary UP/DOWN bằng target có stop-loss/take-profit → label gần thực tế hơn | Thêm hyperparams |
| **Fractional differentiation** | Giữ memory của chuỗi giá mà vẫn stationary | Complexity |
| **Meta-labeling** | Model 1 dự đoán direction, Model 2 dự đoán **bet size/confidence** | Literature mới |
| **Multi-scale features** | Kết hợp features ở weekly + monthly horizon thay vì chỉ daily | Feature engineering |
| **Regime detection** | Tự động detect market regime (bull/bear/range) → train model riêng cho mỗi regime | Clustering + small samples |
| **Longer horizons** | Weekly/monthly direction thay vì daily | Ít samples hơn |
| **Probabilistic output** | Thay vì predict direction, predict **confidence interval** hoặc **volatility** | Calibration |

### Không nên thử thêm

| Hướng | Tại sao |
|-------|---------|
| ❌ Thêm nhiều model phức tạp hơn | Dataset quá nhỏ → overfitting |
| ❌ Deeper hyperparameter tuning | Đã chứng minh: extensive tuning → overfit (step 9: CV 61% → test 52%) |
| ❌ Neural networks / Transformers | Quá ít data, signal quá yếu |
| ❌ Feature engineering phức tạp hơn | Feature-target correlation quá thấp — vấn đề ở source data, không phải engineering |

---

## Kết luận cuối cùng

**Mô hình ML KHÔNG thất bại — thị trường dầu quá hiệu quả.**

57.4% accuracy với:
- Honest time-split evaluation
- No data leakage
- Public data only
- 2,083 training samples

...là kết quả **tốt** theo chuẩn academic. Bất kỳ ai claim >70% trên daily oil direction mà không dùng same-day features hoặc alternative data đều cần được verify rất cẩn thận.

Bài toán "dự đoán hướng giá dầu hàng ngày từ dữ liệu công khai" về bản chất gần với việc **tung đồng xu hơi lệch** (biased coin flip) — ML chỉ có thể phát hiện và khai thác phần "hơi lệch" đó, không thể biến nó thành dự đoán chính xác.

---

*Generated: 2026-04-27*
