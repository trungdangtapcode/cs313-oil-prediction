# DATA LEAKAGE ANALYSIS

## 1. Data Leakage là gì?

Data leakage xảy ra khi model sử dụng thông tin mà **lẽ ra không có tại thời điểm dự đoán**. Có 2 loại chính:

| Loại | Mô tả | Ví dụ |
|---|---|---|
| **Target leakage** | Feature chứa thông tin trực tiếp hoặc gián tiếp từ target | Dùng giá đóng cửa ngày T để dự đoán return ngày T |
| **Contemporaneous leakage** | Feature là thông tin xảy ra **đồng thời** với target, không biết trước | Dùng return S&P500 ngày T để dự đoán return dầu ngày T (cả 2 chỉ biết sau khi đóng cửa) |

---

## 2. Phân tích từng nhóm features

### 2.1 Dataset gốc (54 cột) — `dataset_step4_transformed.csv`

#### KHÔNG bị leakage (an toàn)

| Feature | Lý do |
|---|---|
| `oil_return_lag1`, `oil_return_lag2` | Shift sẵn — chỉ dùng T-1, T-2 |
| `vix_lag1` | Shift sẵn — VIX ngày T-1 |
| `gdelt_tone_lag1`, `gdelt_volume_lag1` | Shift sẵn — GDELT ngày T-1 |
| `fed_funds_rate_lag`, `cpi_lag`, `unemployment_lag` | Trễ 1 tháng do publication lag |
| `oil_volatility_7d` | Rolling std trên `oil_return`, nhưng bao gồm return ngày T → **nên shift(1)** |
| `gdelt_tone_7d`, `gdelt_tone_30d`, `gdelt_goldstein_7d`, `gdelt_volume_7d` | Rolling trên GDELT — bao gồm ngày T nhưng GDELT data thường available trong ngày → chấp nhận được |
| `yield_spread`, `cpi_yoy`, `real_rate`, `fed_rate_change`, `fed_rate_regime`, `recession_signal` | Macro data — cập nhật theo tháng/quý, biết trước ngày T |
| `inventory_zscore`, `inventory_change_pct`, `production_change_pct`, `net_imports_change_pct` | EIA weekly data — công bố trước ngày T |
| `conflict_event_count`, `fatalities`, `conflict_intensity_7d`, `fatalities_7d` | ACLED data — delay vài ngày |
| `geopolitical_stress_index`, `stress_tone`, `stress_volume`, `stress_goldstein` | Derived từ GDELT — tương tự GDELT |
| `gdelt_tone_spike`, `media_attention_spike`, `gdelt_data_imputed` | Binary flags — OK |
| `day_of_week`, `month` | Calendar features — biết trước |

#### Contemporaneous (cùng ngày, cần cẩn thận)

| Feature | Vấn đề | Mức độ |
|---|---|---|
| `sp500_return` | Return S&P500 **cùng ngày T** — chỉ biết sau khi thị trường Mỹ đóng cửa | Trung bình |
| `usd_return` | Return USD Index **cùng ngày T** — tương tự | Trung bình |
| `vix_return` | Return VIX **cùng ngày T** — tương tự | Trung bình |
| `oil_close` | Giá đóng cửa dầu ngày T — **chứa trực tiếp target** | Cao |
| `usd_close`, `sp500_close`, `vix_close` | Giá đóng cửa cùng ngày | Cao |
| `wti_fred` | Giá WTI cùng ngày (gần giống oil_close) | Cao |

**Lưu ý:** `oil_close`, `usd_close`, `sp500_close`, `vix_close`, `wti_fred` đã bị drop trong `config.py` (DROP_COLS). Nhưng `sp500_return`, `usd_return`, `vix_return` vẫn được dùng.

**Trong thực tế:** Nếu dự đoán return dầu **trước khi thị trường đóng cửa**, thì `sp500_return`, `usd_return`, `vix_return` cùng ngày là leakage. Nếu dự đoán return **ngày mai** (T+1), thì chúng không phải leakage.

---

### 2.2 Technical Indicators (thêm ở step4_improve.py) — BỊ LEAKAGE

Tất cả technical indicators sau đây sử dụng `oil_close` ngày T trong phép tính:

| Feature | Công thức | Dùng `close_T`? | Leakage? |
|---|---|---|---|
| `rsi_14` | RSI từ `price.diff()` 14 ngày | Có — `diff_T = close_T - close_{T-1}` chính là return ngày T | **CÓ** |
| `macd` | EMA12 - EMA26 | Có — EMA bao gồm `close_T` | **CÓ** |
| `macd_signal` | EMA9 của MACD | Có — MACD đã bao gồm `close_T` | **CÓ** |
| `macd_hist` | MACD - Signal | Có | **CÓ** |
| `macd_cross` | MACD > Signal | Có | **CÓ** |
| `ma_5/10/20/50` | Rolling mean giá | Có — mean bao gồm `close_T` | **CÓ** |
| `ma_5_10_cross` | MA5 > MA10 | Có | **CÓ** |
| `ma_10_20_cross` | MA10 > MA20 | Có | **CÓ** |
| `ma_20_50_cross` | MA20 > MA50 | Có | **CÓ** |
| `price_vs_ma20` | `(close_T - MA20) / MA20` | Có — trực tiếp | **CÓ** |
| `price_vs_ma50` | `(close_T - MA50) / MA50` | Có — trực tiếp | **CÓ** |
| `bb_upper/lower` | MA20 ± 2×std20 | Có | **CÓ** |
| `bb_width` | `(upper - lower) / MA20` | Có | **CÓ** |
| `bb_position` | `(close_T - lower) / (upper - lower)` | Có — trực tiếp | **CÓ** |
| `momentum_5/10/20` | `price.pct_change(N)` | Có — `(close_T - close_{T-N}) / close_{T-N}` | **CÓ** |
| `ret_std_5/20` | `oil_return.rolling(N).std()` | Có — rolling bao gồm return ngày T | **CÓ** |
| `ret_mean_5/20` | `oil_return.rolling(N).mean()` | Có — rolling bao gồm return ngày T | **CÓ** |
| `gdelt_vol_momentum` | `gdelt_volume_7d.pct_change(5)` | Có — bao gồm ngày T | **CÓ** |

#### Features KHÔNG bị leakage (đã shift)

| Feature | Lý do |
|---|---|
| `oil_return_lag3` | `oil_return.shift(3)` — chỉ dùng T-3 |
| `oil_return_lag5` | `oil_return.shift(5)` — chỉ dùng T-5 |
| `sp500_return_lag1` | `sp500_return.shift(1)` — chỉ dùng T-1 |
| `vix_return_lag1` | `vix_return.shift(1)` — chỉ dùng T-1 |

---

## 3. Cách sửa: Shift(1)

### Nguyên tắc

Mọi feature tính từ giá hoặc return **phải shift(1)** để chỉ dùng thông tin đến ngày T-1:

```python
# SAI (leakage)
df['rsi_14'] = compute_rsi(price, 14)

# ĐÚNG (không leakage)
df['rsi_14'] = compute_rsi(price, 14).shift(1)
```

### Tại sao shift(1) là đủ?

**Target ngày T:**
```
oil_return_T = (close_T - close_{T-1}) / close_{T-1}
```

**Feature sau shift(1):**
```
rsi_14_shifted = RSI tính từ close_{T-1}, close_{T-2}, ..., close_{T-14}
```

Sau shift(1), feature chỉ dùng `close_{T-1}` trở về trước → **không chứa `close_T`** → không leak.

### Ý nghĩa thực tế

Shift(1) tương đương với: "vào cuối ngày T-1, bạn tính RSI/MACD/... từ dữ liệu đã có, rồi dùng nó để dự đoán return ngày T."

Đây chính xác là cách một trader thực tế sẽ sử dụng technical indicators.

---

## 4. Tác động lên kết quả

| Phiên bản | Accuracy | Leakage? |
|---|---|---|
| Baseline (54 features gốc, không technicals) | ~53% | Không (trừ contemporaneous returns) |
| Step4 technicals **KHÔNG shift** | 89.2% | **CÓ — kết quả giả** |
| Step4 technicals **CÓ shift(1)** | ~53-55% (dự kiến) | Không |

**Kết luận:** Accuracy 89.2% là giả tạo do leakage. Sau khi shift(1), accuracy sẽ rơi về gần baseline (~53-55%), đây là kết quả thật.

---

## 5. Contemporaneous features — đánh giá rủi ro

| Feature | Cần shift? | Giải thích |
|---|---|---|
| `sp500_return` | **Tùy use case** | Nếu predict trước khi thị trường đóng cửa → cần shift. Nếu predict cho ngày mai → không cần |
| `usd_return` | **Tùy use case** | Tương tự |
| `vix_return` | **Tùy use case** | Tương tự |
| `oil_volatility_7d` | **Nên shift(1)** | Rolling std bao gồm return ngày T |
| `gdelt_tone/volume/goldstein` | Không cần | GDELT data available trong ngày (không phụ thuộc giá dầu) |
| Macro features (CPI, Fed, EIA...) | Không cần | Public data, biết trước |

### Khuyến nghị

- **Nếu predict return ngày T (intraday):** Shift tất cả market returns + technicals
- **Nếu predict return ngày T+1 (overnight):** Có thể dùng returns ngày T, chỉ cần shift technicals dùng `close_T`

---

*Generated: 2026-04-12*
