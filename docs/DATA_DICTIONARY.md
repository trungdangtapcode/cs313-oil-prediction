# DATA DICTIONARY - Oil Price Prediction Project

**Mô tả chi tiết 54 cột trong `dataset_step4_transformed.csv`**

> File này là bộ dữ liệu đầy đủ nhất (54 cột, 2923 dòng), chưa bị bỏ bất kỳ feature nào.
> File `dataset_final.csv` (33 cột) là phiên bản đã bị step5 loại bớt 21 cột.

---

## Tổng quan cấu trúc dữ liệu

| Pipeline Step | File output | Số cột | Mô tả |
|---|---|---|---|
| Raw | `data/raw/*.csv` | 5 file riêng | Dữ liệu thô từ 5 nguồn |
| Step 2 - Cleaning | `dataset_step2_cleaned.csv` | 26 | Merge 5 nguồn, làm sạch, ffill |
| Step 3 - Integration | `dataset_step3_integrated.csv` | 26 | Re-index full business days |
| **Step 4 - Transformation** | **`dataset_step4_transformed.csv`** | **54** | **Đầy đủ nhất - 26 gốc + 28 engineered** |
| Step 5 - Reduction | `dataset_final.csv` | 33 | Bỏ 21 cột cho model-ready |

---

## Nguồn dữ liệu

| Nguồn | File raw | Số cột gốc | Nội dung chính |
|---|---|---|---|
| Yahoo Finance | `market_data.csv` | 5 | Giá dầu Brent, USD Index, S&P500, VIX |
| FRED | `fred_data.csv` | 6 | Lãi suất Fed, CPI, thất nghiệp, yield spread, WTI |
| EIA | `eia_data.csv` | 5 | Tồn kho, sản lượng, nhập khẩu dầu thô Mỹ |
| GDELT | `gdelt_data.csv` | 10 | Tone truyền thông, Goldstein, volume tin Trung Đông |
| ACLED | `ACLED Data_2026-03-26.csv` | N/A | Sự kiện xung đột, thương vong (**FILE BỊ THIẾU**) |

---

## PHẦN A: 26 CỘT GỐC (Base Features - từ Step 2 Cleaning)

Các cột này được tạo bằng cách merge 5 nguồn raw data lại với nhau trên trục thời gian của market (chỉ ngày giao dịch Mon-Fri), sau đó forward-fill (limit=3) để lấp khoảng trống.

### Nhóm 1: Market Data (Yahoo Finance)

| # | Tên cột | Kiểu | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 1 | `date` | datetime | YYYY-MM-DD | Ngày giao dịch (chỉ business days Mon-Fri, từ 2015-01-01 đến 2026-03-20) | Trục thời gian chính của toàn bộ dataset |
| 2 | `oil_close` | float | USD/barrel | Giá đóng cửa dầu Brent hàng ngày | **Biến mục tiêu gốc** - giá dầu thô quốc tế. Brent là benchmark cho ~2/3 lượng dầu giao dịch toàn cầu |
| 3 | `usd_close` | float | Index points | Chỉ số USD Index (DXY) - đo sức mạnh đồng USD so với rổ 6 đồng tiền chủ chốt (EUR, JPY, GBP, CAD, SEK, CHF) | USD tăng → dầu thường giảm vì dầu định giá bằng USD. Quan hệ nghịch chiều kinh điển |
| 4 | `sp500_close` | float | Index points | Chỉ số S&P 500 - đại diện 500 công ty lớn nhất Mỹ | Phản ánh sức khỏe kinh tế, nhu cầu năng lượng. Kinh tế tốt → nhu cầu dầu tăng → giá tăng |
| 5 | `vix_close` | float | % (annualized) | Chỉ số biến động VIX ("Fear Index") - đo kỳ vọng biến động của S&P 500 trong 30 ngày tới | VIX cao = thị trường lo sợ/bất ổn. Thường tương quan nghịch với giá dầu vì investor bán tài sản rủi ro |

### Nhóm 2: FRED - Kinh tế vĩ mô (Federal Reserve Economic Data)

Lưu ý: Các biến tháng (`fed_funds_rate`, `cpi`, `unemployment`) được dịch chuyển +1 tháng khi crawl để phản ánh độ trễ công bố (publication lag), sau đó đổi tên thành `_lag`.

| # | Tên cột | Kiểu | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 6 | `yield_spread` | float | % points | Chênh lệch lợi suất trái phiếu chính phủ Mỹ (10-Year minus 2-Year Treasury) | Âm = đường cong lợi suất đảo ngược → tín hiệu suy thoái kinh điển (đã dự báo đúng mọi lần suy thoái từ 1970s) |
| 7 | `wti_fred` | float | USD/barrel | Giá dầu WTI (West Texas Intermediate) từ FRED | Benchmark dầu Mỹ, tương quan rất cao với Brent (>0.98). Redundant với `oil_close` - bị drop ở step5 |
| 8 | `fed_funds_rate_lag` | float | % | Lãi suất Fed Funds (trễ 1 tháng do publication lag) - lãi suất qua đêm giữa các ngân hàng | Công cụ chính sách tiền tệ chính của Fed. Tăng lãi suất → USD mạnh → dầu có xu hướng giảm. Giảm lãi suất → kích thích kinh tế → nhu cầu dầu tăng |
| 9 | `cpi_lag` | float | Index (1982-84=100) | Chỉ số giá tiêu dùng CPI (trễ 1 tháng) | Đo lạm phát - lạm phát cao có thể đẩy giá hàng hóa (bao gồm dầu) lên. Đồng thời là yếu tố khiến Fed tăng lãi suất |
| 10 | `unemployment_lag` | float | % | Tỷ lệ thất nghiệp (trễ 1 tháng) | Thất nghiệp cao → kinh tế yếu → nhu cầu dầu giảm → giá giảm. Là chỉ báo tham chiếu của chu kỳ kinh tế |

### Nhóm 3: EIA - Cung cầu dầu thô (Energy Information Administration)

Dữ liệu gốc là weekly, được forward-fill thành daily khi crawl.

| # | Tên cột | Kiểu | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 11 | `crude_inventory_weekly` | float | 1000 barrels | Tồn kho dầu thô thương mại Mỹ (báo cáo hàng tuần) | Tồn kho tăng = cung dư thừa → áp lực giảm giá. Tồn kho giảm = cung thiếu → đẩy giá lên. Là chỉ báo cung-cầu trực tiếp nhất |
| 12 | `crude_production_weekly` | float | 1000 barrels/day | Sản lượng khai thác dầu thô Mỹ (báo cáo hàng tuần) | Sản lượng tăng → cung tăng → áp lực giảm giá. Mỹ là nước sản xuất dầu lớn nhất thế giới (>12M barrels/day) |
| 13 | `net_imports_weekly` | float | 1000 barrels/day | Nhập khẩu ròng dầu thô (nhập khẩu - xuất khẩu) | Phản ánh mức phụ thuộc vào dầu nhập khẩu. Net imports giảm = Mỹ tự chủ hơn về năng lượng |
| 14 | `inventory_change_pct` | float | % | Phần trăm thay đổi tồn kho so với tuần trước (tính sẵn khi crawl từ EIA) | Biến động ngắn hạn của tồn kho - thay đổi đột ngột có thể gây shock giá |

### Nhóm 4: GDELT - Truyền thông & Địa chính trị (Global Database of Events, Language, and Tone)

Dữ liệu được lọc chỉ giữ sự kiện liên quan Trung Đông - khu vực ảnh hưởng trực tiếp nhất đến giá dầu.

| # | Tên cột | Kiểu | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 15 | `gdelt_tone` | float | Score (-100 đến +100) | Tone truyền thông trung bình ngày từ GDELT. Âm = tin tiêu cực, dương = tích cực | Tone âm (tin xấu về Trung Đông) → thị trường lo ngại gián đoạn cung → giá dầu tăng |
| 16 | `gdelt_goldstein` | float | Scale (-10 đến +10) | Thang Goldstein - đo mức độ xung đột/hợp tác của sự kiện | -10 = xung đột cực độ (chiến tranh), +10 = hợp tác cực độ (hiệp ước hòa bình). Goldstein âm → rủi ro cung dầu |
| 17 | `gdelt_volume` | float | Count | Số lượng bài báo/mention liên quan sự kiện Trung Đông trong ngày | Volume cao = sự kiện lớn đang được truyền thông chú ý. Thường tăng đột ngột khi có khủng hoảng |
| 18 | `gdelt_events` | float | Count | Số sự kiện GDELT ghi nhận trong ngày ở Trung Đông | Khác volume (số bài báo), đây là số sự kiện thực tế. Nhiều sự kiện = khu vực bất ổn |
| 19 | `gdelt_tone_7d` | float | Score | Tone trung bình 7 ngày (rolling mean) | Xu hướng ngắn hạn của tâm lý truyền thông. Làm mịn nhiễu ngẫu nhiên hàng ngày |
| 20 | `gdelt_tone_30d` | float | Score | Tone trung bình 30 ngày (rolling mean) | Xu hướng dài hạn - dùng làm baseline để phát hiện spike bất thường |
| 21 | `gdelt_tone_spike` | int | 0/1 | Cờ spike tone: `1` nếu `gdelt_tone < gdelt_tone_30d - 1` | Tin xấu đột ngột so với xu hướng 30 ngày → thị trường có thể phản ứng mạnh |
| 22 | `gdelt_volume_7d` | float | Count | Volume trung bình 7 ngày (rolling mean) | Xu hướng chú ý truyền thông ngắn hạn. Tăng đều = sự kiện đang kéo dài |
| 23 | `gdelt_goldstein_7d` | float | Scale | Goldstein trung bình 7 ngày (rolling mean) | Xu hướng xung đột/hợp tác ngắn hạn. Âm liên tục = căng thẳng leo thang |

### Nhóm 5: Dữ liệu phụ trợ & ACLED

| # | Tên cột | Kiểu | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 24 | `gdelt_data_imputed` | int | 0/1 | Cờ imputation: `1` nếu dữ liệu GDELT ngày đó bị thiếu và được forward-fill | Giúp model biết data nào là "thật" vs "ước tính". Model có thể học cách giảm trọng số cho ngày imputed |
| 25 | `conflict_event_count` | float | Count | Số sự kiện xung đột trong ngày (từ ACLED - Armed Conflict Location & Event Data) | Nhiều sự kiện = khu vực bất ổn → rủi ro gián đoạn cung dầu. **Lưu ý: có thể toàn 0 vì file ACLED bị thiếu** |
| 26 | `fatalities` | float | Count | Số người chết do xung đột trong ngày (từ ACLED) | Đo mức nghiêm trọng của xung đột. Thương vong cao → khủng hoảng lớn → giá dầu tăng mạnh. **Lưu ý: có thể toàn 0** |

---

## PHẦN B: 28 CỘT ENGINEERED (Feature Engineering - từ Step 4 Transformation)

Các cột này được tạo từ 26 cột gốc bằng các phép biến đổi: pct_change, rolling, lag, log, z-score, MinMaxScale, và các chỉ báo phái sinh.

### Nhóm 6: Market Returns - % thay đổi hàng ngày

Mục đích: Chuyển từ giá tuyệt đối (non-stationary) sang % thay đổi (stationary) để model học tốt hơn.

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 27 | `oil_return` | `oil_close.pct_change()` | % (decimal) | Phần trăm thay đổi giá dầu Brent so với ngày trước | **ĐÂY LÀ TARGET (biến mục tiêu) CHÍNH cho model dự đoán.** Vd: 0.02 = tăng 2%, -0.01 = giảm 1% |
| 28 | `usd_return` | `usd_close.pct_change()` | % (decimal) | Phần trăm thay đổi USD Index so với ngày trước | USD tăng 1% → dầu thường giảm ~0.5-1%. Stationary version của usd_close |
| 29 | `sp500_return` | `sp500_close.pct_change()` | % (decimal) | Phần trăm thay đổi S&P 500 so với ngày trước | Tín hiệu risk-on/risk-off của thị trường. Tăng = risk-on → dầu có thể tăng theo |
| 30 | `vix_return` | `vix_close.pct_change()` | % (decimal) | Phần trăm thay đổi VIX so với ngày trước | VIX tăng mạnh = panic → nhà đầu tư bán tài sản rủi ro → dầu giảm |

### Nhóm 7: Biến động (Volatility)

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 31 | `oil_volatility_7d` | `oil_return.rolling(7).std(ddof=0)`, winsorize tại percentile 99 | Decimal | Độ lệch chuẩn của oil_return trong 7 ngày gần nhất. Bị cắt ở percentile 99 để loại outlier | Biến động cao = thị trường bất ổn, rủi ro lớn. Biến động thường tăng trước/trong khủng hoảng. Winsorize giúp model không bị ảnh hưởng bởi giá trị cực đoan |

### Nhóm 8: FRED Derived Features - Chỉ báo chính sách tiền tệ & kinh tế

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 32 | `fed_rate_change` | `fed_funds_rate_lag.diff()`, NaN → 0 | % points | Thay đổi lãi suất Fed giữa 2 kỳ liên tiếp | Dương = Fed tăng lãi suất (thắt chặt tiền tệ), âm = Fed cắt lãi suất (nới lỏng). Mỗi lần Fed thay đổi lãi suất đều tác động mạnh lên thị trường |
| 33 | `recession_signal` | `1 if yield_spread < 0, else 0` | 0/1 | Tín hiệu suy thoái: 1 khi yield curve đảo ngược | Yield curve đảo ngược là chỉ báo suy thoái kinh điển (độ chính xác ~90% từ 1970s). Suy thoái → nhu cầu dầu giảm mạnh |
| 34 | `cpi_yoy` | `CPI_monthly.pct_change(12) * 100` (từ raw FRED monthly, reindex với ffill) | % | Lạm phát năm-qua-năm (Year-over-Year) | Lạm phát cao → Fed tăng lãi suất → USD mạnh → dầu giảm. Nhưng lạm phát cũng có thể phản ánh giá hàng hóa (bao gồm dầu) đang tăng |
| 35 | `fed_rate_regime` | Phân loại: 0 nếu rate<0.5, 1 nếu diff>0.1, 2 nếu rate>3.0 & abs(diff)<0.1, 3 = khác | Category (0-3) | Chế độ lãi suất Fed | `0` = gần 0% (QE/khủng hoảng), `1` = đang tăng (thắt chặt), `2` = cao & ổn định (>3%, bình ổn), `3` = khác. Mỗi regime có tác động khác nhau lên giá dầu |
| 36 | `real_rate` | `fed_funds_rate_lag - cpi_yoy` | % | Lãi suất thực (lãi suất danh nghĩa trừ lạm phát) | Real rate dương = chính sách tiền tệ thắt chặt thực sự → áp lực giảm giá dầu. Real rate âm = tiền "rẻ" → kích thích đầu tư hàng hóa |

### Nhóm 9: EIA Derived Features - Cung cầu chuẩn hóa

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 37 | `inventory_zscore` | `(inventory - rolling_mean_252d) / rolling_std_252d`, fallback expanding z-score cho giai đoạn đầu, NaN → 0 | Z-score (standard deviations) | Tồn kho chuẩn hóa theo trung bình 1 năm (252 ngày giao dịch) | Z > 0: tồn kho trên trung bình → cung dư → áp lực giảm giá. Z < 0: tồn kho dưới trung bình → cung thiếu → đẩy giá lên. Chuẩn hóa giúp so sánh cross-time |
| 38 | `production_change_pct` | `crude_production.pct_change(5)`, NaN → 0 | % (decimal) | Phần trăm thay đổi sản lượng trong 5 ngày (xấp xỉ 1 tuần) | Sản lượng tăng nhanh → cung dư thừa. Sản lượng giảm đột ngột (vd: bão, lệnh trừng phạt) → giá tăng |
| 39 | `net_imports_change_pct` | `net_imports.pct_change(5)`, NaN → 0 | % (decimal) | Phần trăm thay đổi nhập khẩu ròng trong 5 ngày | Nhập khẩu tăng = nhu cầu nội địa vượt sản xuất. Nhập khẩu giảm = Mỹ tự chủ hơn |

### Nhóm 10: GDELT Derived Features - Truyền thông nâng cao

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 40 | `gdelt_volume_log` | `log(1 + gdelt_volume)` | Log scale | Log-transform của volume truyền thông | Giảm skew của volume gốc (rất lệch phải - vài ngày volume cực lớn). Log giúp model xử lý tốt hơn các giá trị extreme |
| 41 | `media_attention_spike` | `1 if gdelt_volume_log > percentile_95 của 90 ngày gần nhất` | 0/1 | Spike chú ý truyền thông: 1 khi volume vượt percentile 95 trong 90 ngày | Sự kiện bất thường đang được truyền thông tập trung (chiến tranh, khủng hoảng, đàm phán OPEC...). Thường đi kèm biến động giá dầu lớn |

### Nhóm 11: ACLED Derived Features - Xung đột rolling

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 42 | `conflict_intensity_7d` | `conflict_event_count.rolling(7).sum()` | Count | Tổng số sự kiện xung đột trong 7 ngày gần nhất | Xung đột leo thang (tăng đều) → rủi ro gián đoạn cung dầu từ Trung Đông. **Lưu ý: có thể toàn 0 vì file ACLED bị thiếu** |
| 43 | `fatalities_7d` | `fatalities.rolling(7).sum()` | Count | Tổng số tử vong do xung đột trong 7 ngày | Đo mức nghiêm trọng của xung đột. Thương vong tăng = tình hình xấu đi → giá dầu có thể tăng do lo ngại. **Lưu ý: có thể toàn 0** |

### Nhóm 12: Geopolitical Stress Index - Chỉ số căng thẳng tổng hợp

MinMaxScaler được fit chỉ trên dữ liệu train (trước 2023-01-01) để tránh data leakage.

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 44 | `stress_tone` | `MinMaxScale(-gdelt_tone)`, fit trên train (<2023) | 0 - 1 | Stress từ tone truyền thông. Đảo dấu tone (âm → dương = stress cao), chuẩn hóa 0-1 | Tone càng âm (tin xấu) → stress càng cao → rủi ro giá dầu biến động |
| 45 | `stress_volume` | `MinMaxScale(gdelt_volume_log)`, fit trên train (<2023) | 0 - 1 | Stress từ volume truyền thông. Volume cao = nhiều tin = stress | Truyền thông tập trung vào Trung Đông = đang có sự kiện lớn → bất ổn |
| 46 | `stress_goldstein` | `MinMaxScale(-gdelt_goldstein)`, fit trên train (<2023) | 0 - 1 | Stress từ thang Goldstein. Đảo dấu (xung đột cao → stress cao) | Goldstein âm (xung đột) được chuyển thành stress dương |
| 47 | `geopolitical_stress_index` | `0.40 * stress_tone + 0.35 * stress_volume + 0.25 * stress_goldstein` | 0 - 1 | **Chỉ số căng thẳng địa chính trị tổng hợp** - kết hợp 3 thành phần với trọng số | Cao (gần 1) = khu vực Trung Đông rất bất ổn → giá dầu có thể tăng mạnh do lo ngại gián đoạn cung. Thấp (gần 0) = ổn định |

**Trọng số geopolitical_stress_index:**
- Tone (40%): Yếu tố quan trọng nhất vì phản ánh trực tiếp cảm xúc truyền thông
- Volume (35%): Lượng tin phản ánh mức độ chú ý
- Goldstein (25%): Thang xung đột/hợp tác bổ sung thêm ngữ cảnh

### Nhóm 13: Lag Features - Giá trị trễ (chống data leakage)

Mục đích: Dùng giá trị của ngày hôm trước để dự đoán hôm nay, tránh sử dụng thông tin cùng ngày (data leakage).

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 48 | `oil_return_lag1` | `oil_return.shift(1)` | % (decimal) | Return dầu ngày hôm qua (t-1) | Momentum ngắn hạn: dầu tăng hôm qua có tiếp tục tăng hôm nay? Hiệu ứng momentum/mean-reversion |
| 49 | `oil_return_lag2` | `oil_return.shift(2)` | % (decimal) | Return dầu 2 ngày trước (t-2) | Momentum xa hơn. Kết hợp với lag1 để nhận biết pattern ngắn hạn |
| 50 | `vix_lag1` | `vix_close.shift(1)` | % (annualized) | VIX ngày hôm qua (t-1) | Tránh dùng VIX cùng ngày (leakage). VIX hôm qua ảnh hưởng tâm lý giao dịch hôm nay |
| 51 | `gdelt_tone_lag1` | `gdelt_tone.shift(1)` | Score | Tone GDELT ngày hôm qua (t-1) | Tin tức hôm qua ảnh hưởng giá hôm nay. Tin xấu hôm qua → trader phản ứng hôm nay |
| 52 | `gdelt_volume_lag1` | `gdelt_volume_log.shift(1)` | Log scale | Log volume GDELT ngày hôm qua (t-1) | Lượng tin hôm qua → mức độ chú ý carry-over sang hôm nay |

### Nhóm 14: Time Features - Hiệu ứng thời gian

| # | Tên cột | Công thức | Đơn vị | Mô tả | Ý nghĩa kinh tế |
|---|---|---|---|---|---|
| 53 | `day_of_week` | `date.dt.dayofweek` | 0-4 | Ngày trong tuần: 0 = Thứ 2, 1 = Thứ 3, ..., 4 = Thứ 6 | Giá dầu có pattern theo ngày: Thứ 2 thường biến động mạnh hơn (tích lũy tin cuối tuần), Thứ 6 có thể có hiệu ứng chốt lời |
| 54 | `month` | `date.dt.month` | 1-12 | Tháng trong năm | Seasonality: mùa đông (tháng 11-2) nhu cầu sưởi ấm tăng → giá tăng. Mùa hè (tháng 6-8) mùa lái xe/vận tải → nhu cầu xăng tăng |

---

## TÓM TẮT THEO NHÓM CHỨC NĂNG

| Nhóm | Số cột | Các cột | Vai trò |
|---|---|---|---|
| Market raw prices | 4 | `oil_close`, `usd_close`, `sp500_close`, `vix_close` | Giá gốc các thị trường tài chính |
| FRED macro | 5 | `yield_spread`, `wti_fred`, `fed_funds_rate_lag`, `cpi_lag`, `unemployment_lag` | Chính sách tiền tệ & kinh tế vĩ mô |
| EIA supply | 4 | `crude_inventory_weekly`, `crude_production_weekly`, `net_imports_weekly`, `inventory_change_pct` | Cung cầu dầu thực tế |
| GDELT sentiment | 9 | `gdelt_tone/goldstein/volume/events`, `*_7d`, `*_30d`, `*_spike` | Tâm lý truyền thông & địa chính trị |
| ACLED conflict | 2 | `conflict_event_count`, `fatalities` | Xung đột vũ trang |
| Auxiliary flags | 1 | `gdelt_data_imputed` | Đánh dấu data imputed |
| Date | 1 | `date` | Trục thời gian |
| **Market returns** | **4** | `oil_return` **(TARGET)**, `usd_return`, `sp500_return`, `vix_return` | % thay đổi hàng ngày |
| Volatility | 1 | `oil_volatility_7d` | Đo rủi ro/biến động |
| FRED engineered | 5 | `fed_rate_change`, `recession_signal`, `cpi_yoy`, `fed_rate_regime`, `real_rate` | Chỉ báo chính sách tiền tệ nâng cao |
| EIA engineered | 3 | `inventory_zscore`, `production_change_pct`, `net_imports_change_pct` | Cung cầu chuẩn hóa |
| GDELT engineered | 2 | `gdelt_volume_log`, `media_attention_spike` | Phát hiện sự kiện bất thường |
| ACLED engineered | 2 | `conflict_intensity_7d`, `fatalities_7d` | Xung đột rolling |
| Stress index | 4 | `stress_tone`, `stress_volume`, `stress_goldstein`, `geopolitical_stress_index` | Chỉ số căng thẳng tổng hợp |
| Lag features | 5 | `oil_return_lag1/lag2`, `vix_lag1`, `gdelt_tone_lag1`, `gdelt_volume_lag1` | Giá trị trễ (chống leakage) |
| Time features | 2 | `day_of_week`, `month` | Hiệu ứng mùa/ngày |

---

## 21 CỘT BỊ DROP Ở STEP 5 (Reduction)

Các cột sau bị loại khi tạo `dataset_final.csv` (33 cột) từ `dataset_step4_transformed.csv` (54 cột):

| # | Cột bị drop | Lý do |
|---|---|---|
| 1 | `stress_tone` | Thành phần trung gian của `geopolitical_stress_index` - đã được tổng hợp |
| 2 | `stress_volume` | Thành phần trung gian của `geopolitical_stress_index` - đã được tổng hợp |
| 3 | `stress_goldstein` | Thành phần trung gian của `geopolitical_stress_index` - đã được tổng hợp |
| 4 | `wti_fred` | Redundant với `oil_close` (correlation >0.98) |
| 5 | `gdelt_tone_spike` | Redundant - đã có `gdelt_tone_7d` và `gdelt_tone_30d` |
| 6 | `media_attention_spike` | Binary quá thô - đã có `gdelt_volume_log` chi tiết hơn |
| 7 | `recession_signal` | Binary quá đơn giản - đã có `yield_spread` liên tục |
| 8 | `gdelt_data_imputed` | Cờ kỹ thuật, không phải feature kinh tế |
| 9 | `cpi_yoy` | Đã được bao hàm trong `real_rate` |
| 10 | `fed_funds_rate_lag` | Đã được bao hàm trong `fed_rate_change` và `real_rate` |
| 11 | `crude_inventory_weekly` | Đã được chuẩn hóa thành `inventory_zscore` |
| 12 | `crude_production_weekly` | Đã được chuẩn hóa thành `production_change_pct` |
| 13 | `net_imports_weekly` | Đã được chuẩn hóa thành `net_imports_change_pct` |
| 14 | `gdelt_tone` | Đã có `gdelt_tone_7d` (smoothed) và `gdelt_tone_lag1` |
| 15 | `gdelt_volume` | Đã có `gdelt_volume_log` (transformed) |
| 16 | `gdelt_volume_log` | Đã có `gdelt_volume_lag1` (lag version) |
| 17 | `gdelt_volume_7d` | Redundant với `gdelt_volume_lag1` |
| 18 | `vix_close` | Same-day leakage risk - đã có `vix_lag1` và `vix_return` |
| 19 | `usd_close` | Same-day leakage risk - đã có `usd_return` |
| 20 | `sp500_close` | Same-day leakage risk - đã có `sp500_return` |
| 21 | `oil_close` | Giá tuyệt đối - model dự đoán `oil_return` (% thay đổi), không cần giá gốc |

---

## LƯU Ý QUAN TRỌNG

1. **Target variable**: `oil_return` (cột 27) - dự án dự đoán biến động % giá dầu, không phải giá tuyệt đối
2. **File ACLED bị thiếu**: `data/raw/ACLED Data_2026-03-26.csv` không tồn tại trên disk → các cột `conflict_event_count`, `fatalities`, `conflict_intensity_7d`, `fatalities_7d` có thể toàn 0/NaN
3. **Data leakage**: Các cột same-day (`vix_close`, `usd_close`, `sp500_close`, `oil_close`) có nguy cơ leakage khi dự đoán → nên dùng phiên bản lag hoặc return
4. **Train/Test split**: Train < 2023-01-01 (2083 rows), Test >= 2023-01-01 (840 rows). MinMaxScaler cho stress index chỉ fit trên train
5. **Forward-fill limit**: Tất cả forward-fill đều có `limit=3` để tránh lan truyền dữ liệu quá xa
6. **Publication lag**: Các biến FRED tháng (`fed_funds_rate`, `cpi`, `unemployment`) đã được dịch +1 tháng để phản ánh độ trễ công bố thực tế

---

*Generated: 2026-04-12*
*Source: `dataset_step4_transformed.csv` (54 columns, 2923 rows)*
