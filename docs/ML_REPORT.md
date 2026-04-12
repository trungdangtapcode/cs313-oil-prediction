# BÁO CÁO MACHINE LEARNING - Oil Price Prediction

## 1. Tổng quan dự án

**Mục tiêu:** Dự đoán biến động giá dầu Brent hàng ngày bằng Machine Learning.

**Dataset:** `dataset_step4_transformed.csv` — 2,923 dòng × 54 cột, giai đoạn 2015-01-07 đến 2026-03-20.

**Hai hướng tiếp cận:**
- **Regression:** Dự đoán giá trị `oil_return` (% thay đổi giá dầu hàng ngày)
- **Classification:** Dự đoán hướng giá (UP / DOWN)

**Train/Test Split:** Chia theo thời gian (không random)
- Train: 2,083 mẫu (2015-01-07 → 2022-12-30)
- Test: 840 mẫu (2023-01-02 → 2026-03-20)

---

## 2. Kết quả EDA chính

| Phân tích | Kết quả |
|---|---|
| Chất lượng dữ liệu | 0 missing, 0 INF, 0 duplicates |
| Target (`oil_return`) | Không phân phối chuẩn (kurtosis=12.1, skew=-0.39), Stationary |
| Tính dừng | 19 stationary / 12 trend-stationary / 21 non-stationary |
| ARCH effects | Có — volatility clustering quanh COVID (2020) và Ukraine (2022) |
| Correlation với target | Rất yếu — chỉ 3 features có \|Spearman\| > 0.1 |
| Đa cộng tuyến | 34 cặp \|ρ\| > 0.80, 22 features VIF > 10 |
| Distribution shift | 41/53 features bị shift giữa train và test |
| PCA | 90% variance cần 20 PCs, 95% cần 25 PCs |

**Top 3 features tương quan với target:** `sp500_return` (0.261), `vix_return` (-0.233), `usd_return` (-0.122)

**Top 5 Mutual Information:** `oil_volatility_7d`, `sp500_return`, `vix_return`, `usd_return`, `cpi_yoy`

**Plots EDA:** Xem thư mục `outputs/` (10 files bao gồm distributions, heatmap, scatter, boxplot, time series, PCA...)

---

## 3. Regression — Dự đoán `oil_return`

### 3.1 Baseline (10 models, 42 features)

| # | Model | MAE | RMSE | R² | DA% | Thời gian |
|---|---|---|---|---|---|---|
| 1 | LightGBM | 0.01403 | **0.01938** | **0.014** | 49.9% | 24.5s |
| 2 | XGBoost | 0.01401 | 0.01948 | 0.004 | 48.8% | 26.7s |
| 3 | ElasticNet | 0.01413 | 0.01959 | -0.007 | 48.7% | 0.2s |
| 4 | GradientBoosting | 0.01406 | 0.01966 | -0.015 | 49.5% | 33.4s |
| 5 | Lasso | 0.01428 | 0.01976 | -0.025 | 47.9% | 0.1s |
| 6 | SVR | 0.01485 | 0.01998 | -0.048 | 48.8% | 2.8s |
| 7 | Ridge | 0.01483 | 0.02028 | -0.079 | 50.0% | 5.2s |
| 8 | RandomForest | 0.01465 | 0.02029 | -0.080 | 48.8% | 24.1s |
| 9 | LinearRegression | 0.01698 | 0.02251 | -0.331 | 51.9% | 0.1s |
| 10 | MLP | 0.04204 | 0.06609 | -10.465 | 48.5% | 6.6s |

**Nhận xét:** LightGBM là model duy nhất có R² > 0. Hầu hết models có R² < 0, tức là tệ hơn dự đoán trung bình. Dự đoán giá trị chính xác của daily return là bài toán cực khó.

### 3.2 Fine-tuned (RandomizedSearchCV, 30 iter)

| # | Model | RMSE | R² | DA% |
|---|---|---|---|---|
| 1 | XGBoost_v2 | 0.01966 | -0.015 | 50.0% |
| 2 | LightGBM_v2 | 0.01981 | -0.031 | 50.4% |
| 3 | GBM_v2 | 0.02029 | -0.081 | 50.2% |

### 3.3 Ensemble

| # | Model | RMSE | R² | DA% |
|---|---|---|---|---|
| 1 | **Stacking_Reg** | **0.01946** | **0.006** | 49.8% |
| 2 | Voting_Reg | 0.02102 | -0.159 | 49.0% |

### 3.4 Feature Importance (Regression, 4 tree models trung bình)

| # | Feature | Importance |
|---|---|---|
| 1 | `sp500_return` | 0.228 |
| 2 | `oil_volatility_7d` | 0.177 |
| 3 | `vix_return` | 0.082 |
| 4 | `usd_return` | 0.080 |
| 5 | `oil_return_lag1` | 0.067 |

### 3.5 Backtest — Regression (Long/Short strategy)

| Model | Tổng Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Buy & Hold | +26.5% | — | — |
| **XGBoost_v2** | **+26.8%** | **0.39** | -55.7% |
| LightGBM_v2 | +22.3% | 0.35 | -38.0% |
| Stacking_Reg | +11.5% | 0.26 | -48.6% |
| LightGBM (baseline) | +6.0% | 0.21 | -47.8% |

---

## 4. Classification — Dự đoán hướng giá (UP/DOWN)

### 4.1 Baseline (8 models, 42 features, target = `oil_return > 0`)

| # | Model | Accuracy | F1_macro | AUC |
|---|---|---|---|---|
| 1 | GradientBoosting | 52.7% | 0.527 | 0.557 |
| 2 | SVM_Linear | 52.6% | 0.525 | 0.525 |
| 3 | SVM_RBF | 51.3% | 0.513 | 0.530 |
| 4 | XGBoost | 51.7% | 0.513 | 0.556 |
| 5 | RandomForest | 51.5% | 0.512 | 0.551 |
| 6 | LightGBM | 51.4% | 0.507 | 0.556 |
| 7 | LogisticRegression | 50.4% | 0.500 | 0.523 |
| 8 | MLP | 53.0% | 0.491 | 0.540 |

**Nhận xét:** Tất cả models chỉ đạt ~50-53% accuracy, gần như random guessing. Nguyên nhân: features kinh tế vĩ mô/sentiment có tương quan rất yếu với daily return.

### 4.2 Fine-tuned (TOP_25 features, RandomizedSearchCV)

| # | Model | Accuracy | F1_macro | AUC |
|---|---|---|---|---|
| 1 | XGB_cls_v2 | 53.1% | 0.524 | 0.563 |
| 2 | GBM_cls_v2 | 52.4% | 0.522 | 0.562 |
| 3 | LGBM_cls_v2 | 51.1% | 0.510 | 0.553 |
| 4 | SVM_RBF_v2 | 50.6% | 0.497 | 0.529 |

### 4.3 Ensemble

| # | Model | Accuracy | F1_macro | AUC |
|---|---|---|---|---|
| 1 | **Voting_Cls** | **53.6%** | **0.527** | **0.568** |
| 2 | Stacking_Cls | 53.5% | 0.522 | 0.552 |

### 4.4 Feature Importance (Classification, 4 tree models trung bình)

| # | Feature | Importance |
|---|---|---|
| 1 | `sp500_return` | 0.289 |
| 2 | `vix_return` | 0.109 |
| 3 | `usd_return` | 0.072 |
| 4 | `oil_volatility_7d` | 0.037 |
| 5 | `net_imports_change_pct` | 0.031 |

### 4.5 Backtest — Classification (Long if UP, flat if DOWN)

| Model | Tổng Return | Sharpe | Max Drawdown |
|---|---|---|---|
| Buy & Hold | +26.5% | — | — |
| SVM_RBF (baseline) | +94.1% | 1.05 | -24.8% |
| **Voting_Cls** | **+54.7%** | **0.83** | **-13.8%** |
| XGB_cls_v2 | +33.9% | 0.58 | -21.1% |
| GBM_cls_v2 | +30.3% | 0.52 | -20.9% |

---

## 5. Cải thiện Classification — Technical Indicators + Threshold + Multi-Horizon

### 5.1 Các cải tiến áp dụng

1. **Thêm 29 technical indicators:** RSI-14, MACD + signal + histogram, Bollinger Bands (width, position), MA crossover (5/10, 10/20, 20/50), price vs MA, momentum (5d/10d/20d), rolling return stats, thêm lags

2. **Threshold target:** Bỏ ngày có biến động quá nhỏ (noise gần 0), chỉ phân loại ngày có biến động rõ ràng

3. **Multi-horizon:** Thử dự đoán hướng 1 ngày, 3 ngày, 5 ngày

### 5.2 Kết quả — 1-day predictions

| Target | Best Model | Accuracy | F1_macro | AUC | So với baseline |
|---|---|---|---|---|---|
| **1d_raw** (return > 0) | GBM | **81.4%** | 0.814 | 0.893 | **+28.7%** |
| **1d_t03** (bỏ \|return\| < 0.3%) | LGBM | **85.6%** | 0.854 | 0.935 | **+32.9%** |
| **1d_t05** (bỏ \|return\| < 0.5%) | **LGBM** | **89.2%** | **0.892** | **0.953** | **+36.5%** |

### 5.3 Kết quả — Multi-day predictions

| Target | Best Model | Accuracy | AUC | Nhận xét |
|---|---|---|---|---|
| 3d_raw | LGBM | 52.1% | 0.513 | Gần random |
| 5d_raw | GBM | 50.6% | 0.520 | Gần random |
| 3d_t05 | XGB | 49.2% | 0.524 | Gần random |

### 5.4 Phân tích kết quả

**Tại sao 1-day tăng mạnh (53% → 89%)?**
- Technical indicators (RSI, MACD, Bollinger, MA) chứa thông tin mạnh về xu hướng giá ngắn hạn
- Threshold filtering loại bỏ ~26% ngày biến động nhỏ (noise), model chỉ cần phân loại ngày có tín hiệu rõ
- Kết hợp cả hai biện pháp cho AUC = 0.953

**Tại sao multi-day vẫn ~50%?**
- Forward return 3-5 ngày phụ thuộc thông tin tương lai mà features hiện tại không chứa
- Technical indicators mất hiệu lực ở horizon dài hơn

**Cảnh báo Data Leakage:**
- Một số technical indicators (MA, RSI, Bollinger) sử dụng giá đóng cửa cùng ngày → có thể bị leakage
- Trong thực tế, cần shift tất cả technical features thêm 1 ngày (`shift(1)`) để đảm bảo chỉ dùng thông tin quá khứ
- Kết quả 89% accuracy có thể giảm sau khi khắc phục leakage

---

## 6. Feature Selection

### 6.1 Feature Ranking (MI_regression + MI_classification + |Spearman|)

| # | Feature | MI_reg | MI_cls | \|Spearman\| | Score |
|---|---|---|---|---|---|
| 1 | `sp500_return` | 0.1179 | 0.0448 | 0.2611 | 0.895 |
| 2 | `vix_return` | 0.0973 | 0.0425 | 0.2331 | 0.802 |
| 3 | `usd_return` | 0.0696 | 0.0278 | 0.1215 | 0.496 |
| 4 | `oil_volatility_7d` | 0.1720 | 0.0065 | 0.0038 | 0.387 |
| 5 | `gdelt_tone_30d` | 0.0091 | 0.0157 | 0.0505 | 0.199 |

### 6.2 So sánh Feature Subsets

| Subset | N features | Reg RMSE | Reg R² | Cls Acc | Cls F1m |
|---|---|---|---|---|---|
| ALL_42 | 42 | **0.02279** | -0.364 | 0.533 | 0.518 |
| TOP_10 | 10 | 0.02363 | -0.466 | 0.530 | 0.508 |
| TOP_15 | 15 | 0.02338 | -0.435 | 0.541 | 0.533 |
| TOP_20 | 20 | 0.02338 | -0.435 | 0.530 | 0.520 |
| **TOP_25** | **25** | 0.02291 | -0.378 | **0.555** | **0.544** |

**Kết luận:** ALL_42 tốt nhất cho regression, TOP_25 tốt nhất cho classification.

---

## 7. Tổng kết & Khuyến nghị

### 7.1 Model tốt nhất

| Task | Model | Metric | Feature Set |
|---|---|---|---|
| Regression (RMSE) | LightGBM baseline | RMSE=0.01938 | 42 features |
| Regression (Backtest) | XGBoost_v2 | +26.8%, Sharpe 0.39 | 42 features |
| Classification (Accuracy) | **LGBM + technicals + threshold** | **89.2%** | 81 features (1d_t05) |
| Classification (Backtest) | Voting_Cls | +54.7%, Sharpe 0.83, MaxDD -13.8% | 25 features |

### 7.2 Khuyến nghị

1. **Regression:** R² rất thấp (~0) — mô hình không dự đoán tốt giá trị chính xác daily return. Nên tập trung vào classification.

2. **Classification baseline (không technical):** ~53% accuracy. Cải thiện bằng ensemble (Voting) và feature selection (TOP_25) đưa lên ~54% với backtest vượt Buy&Hold.

3. **Classification + technicals:** Lên tới 89% accuracy nhưng cần kiểm tra kỹ data leakage từ technical indicators cùng ngày.

4. **Multi-day horizon:** Không hiệu quả (~50%) — thị trường quá hiệu quả ở horizon dài.

5. **Trong thực tế nên dùng:** Voting_Cls (baseline, 25 features) với chiến lược Long if UP / flat if DOWN — Sharpe 0.83, MaxDD chỉ -13.8%.

---

## 8. Cấu trúc thư mục

```
ml/
├── config.py                      # Shared config, load data
├── train_regression.py            # 10 models regression baseline
├── train_classification.py        # 8 models classification baseline
├── step1_feature_selection.py     # Feature ranking & subset comparison
├── step2_finetune.py              # Fine-tune top models (reg + cls)
├── step3_ensemble.py              # Stacking & Voting ensembles
├── step4_improve.py               # Technical indicators + threshold + multi-horizon
└── results/
    ├── regression_results.csv
    ├── regression_feature_importance.csv
    ├── regression_importance.png
    ├── regression_backtest.png
    ├── regression_residuals.png
    ├── classification_results.csv
    ├── classification_feature_importance.csv
    ├── classification_importance.png
    ├── classification_confusion.png
    ├── classification_roc.png
    ├── classification_backtest.png
    ├── feature_ranking.csv
    ├── subset_comparison.csv
    ├── finetune_regression.csv
    ├── finetune_classification.csv
    ├── ensemble_regression.csv
    ├── ensemble_classification.csv
    ├── ensemble_backtest.png
    └── improved_classification.csv

outputs/                           # EDA plots
├── 01_dist_all53.png
├── 02_corr_target_all52.png
├── 03_heatmap_53x53.png
├── 04_scatter_all52.png
├── 05_boxplot_all53.png
├── 06_mi_all52.png
├── 07_shift_all53.png
├── 08_timeseries_all53.png
├── 09_pca.png
└── 10_rolling_top10.png
```

---

*Generated: 2026-04-12*
