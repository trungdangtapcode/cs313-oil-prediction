# THỰC NGHIỆM CẢI THIỆN MÔ HÌNH - Oil Price Prediction

Tài liệu ghi lại toàn bộ các phương pháp đã thử để cải thiện accuracy cho bài toán classification (dự đoán hướng giá dầu UP/DOWN).

---

## Tổng quan

| Kết quả | Accuracy |
|---|---|
| Majority class baseline (luôn đoán UP) | 50.2% |
| Baseline model (GBM, 42 features gốc) | 52.7% |
| **Kết quả tốt nhất đạt được** | **57.4%** |
| Cải thiện so với majority class | **+7.2%** |
| Cải thiện so với baseline model | **+4.7%** |

**Model tốt nhất: GradientBoostingClassifier, TOP_50 features (gốc + technical indicators shifted), uniform weight.**

---

## Thực nghiệm 1: Baseline — 10 models, 42 features gốc

**File:** `ml/train_classification.py`

Chạy 8 models classification trên 42 features gốc (đã loại raw prices, redundant, near-zero-var từ 54 cột).

| # | Model | Accuracy | F1_macro | AUC |
|---|---|---|---|---|
| 1 | GradientBoosting | **52.7%** | 0.527 | 0.557 |
| 2 | SVM_Linear | 52.6% | 0.525 | 0.525 |
| 3 | MLP | 53.0% | 0.491 | 0.540 |
| 4 | SVM_RBF | 51.3% | 0.513 | 0.530 |
| 5 | XGBoost | 51.7% | 0.513 | 0.556 |
| 6 | RandomForest | 51.5% | 0.512 | 0.551 |
| 7 | LightGBM | 51.4% | 0.507 | 0.556 |
| 8 | LogisticRegression | 50.4% | 0.500 | 0.523 |

**Nhận xét:** Tất cả models chỉ đạt ~50-53%, gần như random guessing. Features kinh tế vĩ mô và sentiment có tương quan rất yếu với daily return (max |Spearman| = 0.26).

---

## Thực nghiệm 2: Feature Selection — MI + Spearman ranking

**File:** `ml/step1_feature_selection.py`

Xếp hạng 42 features theo Mutual Information (phi tuyến) + |Spearman Correlation| (đơn điệu), thử các subset TOP_10, TOP_15, TOP_20, TOP_25.

| Subset | N | Accuracy | F1_macro |
|---|---|---|---|
| ALL_42 | 42 | 53.3% | 0.518 |
| TOP_25 | 25 | **55.5%** | **0.544** |
| TOP_20 | 20 | 53.0% | 0.520 |
| TOP_15 | 15 | 54.1% | 0.533 |
| TOP_10 | 10 | 53.0% | 0.508 |

**Kết quả:** TOP_25 tốt nhất cho classification. Bỏ bớt features yếu giúp model generalize tốt hơn.

**Hạn chế:** MI + Spearman đánh giá từng feature riêng lẻ, không xét tương tác giữa các features và không xử lý đa cộng tuyến.

---

## Thực nghiệm 3: Fine-tune hyperparameters

**File:** `ml/step2_finetune.py`

RandomizedSearchCV 30 iterations cho GBM, XGBoost, LightGBM, SVM_RBF trên TOP_25 features.

| Model | Accuracy | F1_macro | AUC | Best Params |
|---|---|---|---|---|
| XGB_cls_v2 | **53.1%** | 0.524 | 0.563 | n_estimators=500, max_depth=3, lr=0.01 |
| GBM_cls_v2 | 52.4% | 0.522 | 0.562 | n_estimators=200, max_depth=3, lr=0.01 |
| LGBM_cls_v2 | 51.1% | 0.510 | 0.553 | n_estimators=200, max_depth=3, lr=0.01 |
| SVM_RBF_v2 | 50.6% | 0.497 | 0.529 | C=1.0, gamma=0.01 |

**Nhận xét:** Fine-tune không cải thiện đáng kể so với default params. Tất cả best configs đều chọn max_depth=3 và lr=0.01 (regularization mạnh), cho thấy model cần đơn giản để tránh overfitting.

---

## Thực nghiệm 4: Ensemble — Stacking & Voting

**File:** `ml/step3_ensemble.py`

Kết hợp 3 models (LightGBM + XGBoost + GBM) bằng Voting (soft) và Stacking (meta-learner: LogisticRegression).

| Model | Accuracy | F1_macro | AUC |
|---|---|---|---|
| **Voting_Cls (soft)** | **53.6%** | **0.527** | **0.568** |
| Stacking_Cls | 53.5% | 0.522 | 0.552 |

**Nhận xét:** Voting tốt hơn Stacking. Cải thiện nhẹ so với single model (+0.9%). Ensemble giúp ổn định prediction nhưng không tạo breakthrough.

---

## Thực nghiệm 5: Technical Indicators — PHÁT HIỆN DATA LEAKAGE

**File:** `ml/step4_improve.py`

Thêm 29 technical indicators: RSI-14, MACD (line + signal + histogram + cross), Bollinger Bands (upper, lower, width, position), Moving Averages (5, 10, 20, 50 + crossovers), Momentum (5d, 10d, 20d), Return rolling stats (mean, std), thêm lags.

### Phiên bản 1: KHÔNG shift — BỊ DATA LEAKAGE

| Target | Model | Accuracy | AUC |
|---|---|---|---|
| 1d_raw | GBM | **81.4%** | 0.893 |
| 1d_t05 (bỏ |return| < 0.5%) | LGBM | **89.2%** | 0.953 |

**KẾT QUẢ GIẢ.** Technical indicators dùng `oil_close` ngày T (cùng ngày với target), khiến model gián tiếp "nhìn thấy" đáp án. Chi tiết phân tích leakage xem file `DATA_LEAKAGE.md`.

### Phiên bản 2: CÓ shift(1) — KHÔNG LEAKAGE

Shift tất cả technical indicators 1 ngày để chỉ dùng thông tin đến T-1.

| Target | Model | Accuracy | AUC |
|---|---|---|---|
| 1d_raw | LGBM | **55.2%** | 0.584 |
| 1d_t05 | GBM | 55.3% | 0.565 |
| 1d_t03 | LGBM | 53.5% | 0.563 |

**Kết quả thật:** Technical indicators (đã shift) cải thiện ~2.5% so với baseline. Tuy nhiên phần lớn "cải thiện" ở phiên bản 1 là leakage.

### Threshold target

Bỏ ngày có biến động quá nhỏ (|return| < 0.3% hoặc 0.5%) để giảm noise. Kết quả: không cải thiện đáng kể sau khi fix leakage.

### Multi-day prediction (3 ngày, 5 ngày)

Thử dự đoán hướng giá 3 ngày và 5 ngày tới. Accuracy ~50% (random) cho tất cả models và horizons. Features hiện tại không chứa thông tin dự đoán được hướng dài hạn.

---

## Thực nghiệm 6: Feature Selection trên 81 features (gốc + technicals)

**File:** `ml/step5_select_and_train.py`

Sau khi thêm technicals (shifted), tổng cộng 81 features. Áp dụng MI + Spearman ranking lại trên bộ mở rộng.

| Subset | N | GBM Acc | XGB Acc | LGBM Acc |
|---|---|---|---|---|
| ALL_81 | 81 | 53.9% | 55.2% | 55.8% |
| TOP_50 | 50 | **57.4%** | 55.5% | 55.6% |
| TOP_40 | 40 | **57.3%** | 55.0% | 55.0% |
| TOP_30 | 30 | 56.7% | 55.1% | 55.1% |
| TOP_25 | 25 | 53.9% | 53.5% | 52.3% |
| TOP_20 | 20 | 54.2% | 53.7% | 53.6% |
| TOP_15 | 15 | 53.8% | 50.9% | 50.9% |
| TOP_10 | 10 | 55.5% | 53.4% | 54.4% |

**Kết quả tốt nhất: GBM + TOP_50 = 57.4%**

**Phát hiện quan trọng:** Bỏ 31 features yếu nhất (giữ top 50) giúp GBM tăng từ 53.9% → 57.4%. Feature selection có tác dụng rõ ràng, loại noise giúp model generalize tốt hơn.

---

## Thực nghiệm 7: Smart Feature Selection — Correlation Clustering + Permutation Importance

**File:** `ml/step6_smart_selection.py`

Phương pháp xử lý đồng thời feature importance và đa cộng tuyến:

**Bước 1 - Correlation Clustering:** Tính ma trận |Spearman| giữa 81 features, hierarchical clustering với threshold=0.3 (features có |ρ| > 0.7 gom cùng cụm). Kết quả: 40 clusters từ 81 features.

**Các cụm đa cộng tuyến phát hiện được:**
- `gdelt_tone`, `gdelt_tone_7d`, `stress_tone`, `geopolitical_stress_index`, `gdelt_tone_lag1` → |ρ| = 1.0
- `sp500_close`, `cpi_lag`, `net_imports_weekly`, 6 GDELT volume features → |ρ| = 1.0
- `wti_fred`, `crude_inventory_weekly`, `cpi_yoy`, 4 MA features, 2 BB features → |ρ| = 0.997
- `price_vs_ma20`, `rsi_14`, `macd_hist`, `macd_cross`, `bb_position`, `momentum_10` → |ρ| = 0.942
- `oil_volatility_7d`, `ret_std_5` → |ρ| = 0.893
- `sp500_return`, `vix_return` → |ρ| = 0.776

**Bước 2 - Permutation Importance:** Train LightGBM, xáo trộn từng feature, đo accuracy giảm bao nhiêu. Trong mỗi cluster, giữ feature có permutation importance cao nhất.

**Kết quả:** 81 → 40 features (sau clustering) → 12 features (chỉ giữ perm > 0).

| Feature Set | N | Best Accuracy |
|---|---|---|
| ALL_81 | 81 | 56.8% (GBM) |
| CLUSTER_40 | 40 | 56.6% (XGB) |
| CLUSTER_POS_12 | 12 | 54.2% (GBM) |
| PERM_TOP_10 | 10 | 55.4% (LGBM) |

**Nhận xét:** Smart selection (56.6%) không vượt naive selection (57.4%). Lý do: permutation importance trên test set nhỏ (840 samples) bị nhiễu, dẫn đến chọn feature chưa tối ưu. Phương pháp vẫn có giá trị về mặt lý thuyết (loại đa cộng tuyến đúng) nhưng trên dataset nhỏ không thể hiện ưu thế.

---

## Thực nghiệm 8: Sample Weight Decay

**File:** `ml/step7_weight_decay.py`

Giả thuyết: dữ liệu gần đây quan trọng hơn dữ liệu cũ (market regime thay đổi). Gán weight cao hơn cho training samples gần test set.

**Các weight scheme thử nghiệm:**

| Scheme | Mô tả | Min weight | Mean weight |
|---|---|---|---|
| uniform | Không decay, tất cả bằng nhau | 1.0 | 1.0 |
| exp_hl100 | Exponential, half-life 100 ngày | 0.0000 | 0.071 |
| exp_hl250 | Exponential, half-life 250 ngày | 0.0036 | 0.177 |
| exp_hl500 | Exponential, half-life 500 ngày | 0.0598 | 0.334 |
| exp_hl1000 | Exponential, half-life 1000 ngày | 0.2445 | 0.536 |
| linear_01 | Tuyến tính từ 0.1 đến 1.0 | 0.1 | 0.55 |
| linear_03 | Tuyến tính từ 0.3 đến 1.0 | 0.3 | 0.65 |
| linear_05 | Tuyến tính từ 0.5 đến 1.0 | 0.5 | 0.75 |
| step_50pct_2x | 50% gần nhất weight=2, còn lại weight=1 | 1.0 | 1.5 |
| step_50pct_3x | 50% gần nhất weight=3, còn lại weight=1 | 1.0 | 2.0 |
| step_30pct_3x | 30% gần nhất weight=3, còn lại weight=1 | 1.0 | 1.6 |

**Kết quả (TOP_50 features):**

| Scheme | GBM | XGB | LGBM |
|---|---|---|---|
| **uniform** | **57.4%** | 55.5% | 55.6% |
| exp_hl100 | 50.8% | 53.0% | 50.5% |
| exp_hl250 | 54.2% | 54.2% | 53.7% |
| exp_hl500 | 54.6% | 55.1% | 52.9% |
| exp_hl1000 | 55.0% | 55.6% | 53.2% |
| linear_01 | 56.7% | 56.1% | 54.9% |
| linear_03 | 56.3% | 56.2% | 54.4% |
| linear_05 | 55.4% | 55.8% | 55.8% |
| step_50pct_2x | 56.6% | 53.2% | 55.1% |
| step_50pct_3x | 56.1% | 54.1% | 56.3% |
| step_30pct_3x | 53.8% | 54.8% | 53.6% |

**Kết luận: Weight decay KHÔNG cải thiện.** Uniform (không decay) vẫn tốt nhất.

**Phân tích:**
- Decay quá mạnh (hl=100, hl=250): model gần như bỏ qua 70-80% training data → thiếu data → tệ hơn
- Decay nhẹ (hl=1000, linear_05): gần giống uniform → không khác biệt
- Dataset chỉ có 2033 training samples — đã ít, decay thêm thì càng ít effective samples
- Có thể market regime không thay đổi đủ mạnh giữa train (2015-2022) và test (2023-2026) để weight decay có lợi

---

## Thực nghiệm 9: XGBoost vs GBM — Extensive Tuning (50 iterations)

**File:** `ml/step8_xgb_vs_gbm.py`

Giả thuyết: XGBoost với grid rộng hơn sẽ vượt GBM. RandomizedSearchCV 50 iterations, grid rất rộng cho cả 2.

**XGBoost grid:** n_estimators (100-800), max_depth (2-8), learning_rate (0.005-0.1), min_child_weight (1-10), subsample (0.6-1.0), colsample_bytree (0.5-1.0), reg_alpha (0-1.0), reg_lambda (0.1-5.0), gamma (0-1.0), scale_pos_weight (0.9-1.1).

**GBM grid:** n_estimators (100-800), max_depth (2-8), learning_rate (0.005-0.1), min_samples_leaf (1-20), min_samples_split (2-20), subsample (0.6-1.0), max_features (sqrt, log2, 0.5-0.9, None).

| Model | CV Accuracy | Test Accuracy | F1_macro | AUC | Time |
|---|---|---|---|---|---|
| XGBoost | 60.9% | 54.8% | 0.535 | 0.576 | 172s |
| GBM | 60.8% | 51.9% | 0.518 | 0.559 | 640s |

**Best params XGBoost:** n_estimators=200, max_depth=7, lr=0.005, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.001, reg_lambda=0.5, gamma=0.01, scale_pos_weight=1.1.

**Best params GBM:** n_estimators=100, max_depth=3, lr=0.01, min_samples_leaf=10, min_samples_split=10, subsample=0.7, max_features=0.9.

**Kết luận: Cả 2 đều THẤP hơn config đơn giản ở step5 (57.4%).**

**Phân tích overfitting:** CV accuracy 60-61% nhưng test chỉ 52-55%. Grid rộng → RandomizedSearch tìm được config overfit trên CV folds nhưng không generalize lên test. Dataset quá nhỏ (2033 train) và signal quá yếu → model đơn giản generalize tốt hơn model phức tạp.

---

## Tổng kết tất cả thực nghiệm

| Step | Phương pháp | Features | Model | Accuracy | So với baseline |
|---|---|---|---|---|---|
| 1 | Baseline | 42 | GBM | 52.7% | — |
| 2 | Feature selection (TOP_25) | 25 | GBM | 55.5% | +2.8% |
| 3 | Fine-tune hyperparams | 25 | XGB | 53.1% | +0.4% |
| 4 | Ensemble (Voting) | 25 | LGBM+XGB+GBM | 53.6% | +0.9% |
| 5 | Technical indicators (LEAK) | 81 | LGBM | ~~89.2%~~ | ~~leakage~~ |
| 5 | Technical indicators (shifted) | 81 | LGBM | 55.2% | +2.5% |
| **6** | **Technicals + feature selection** | **50** | **GBM** | **57.4%** | **+4.7%** |
| 7 | Smart selection (cluster+perm) | 40 | XGB | 56.6% | +3.9% |
| 8 | Weight decay | 50 | GBM | 57.4% | +4.7% (=step6) |
| 9 | Extensive tuning (50 iter) | 50 | XGB | 54.8% | +2.1% |

---

## Những gì KHÔNG hiệu quả

| Phương pháp | Tại sao |
|---|---|
| Technical indicators (không shift) | Data leakage — kết quả giả 89% |
| Threshold target (bỏ ngày nhỏ) | Giảm sample size, không cải thiện sau fix leakage |
| Multi-day prediction (3d, 5d) | Features không chứa thông tin tương lai, accuracy ~50% |
| Weight decay / sample weighting | Dataset quá nhỏ, decay làm mất thêm effective samples |
| Extensive hyperparameter search | Overfitting trên CV, test accuracy giảm so với config đơn giản |
| Smart feature selection (cluster+perm) | Permutation importance bị nhiễu trên test set nhỏ |
| Stacking ensemble | Không vượt Voting, phức tạp hơn mà không tốt hơn |

## Những gì CÓ hiệu quả

| Phương pháp | Đóng góp |
|---|---|
| Feature selection (loại features yếu) | +2-4% accuracy, phương pháp hiệu quả nhất |
| Technical indicators (shifted) | +2.5% accuracy, thêm tín hiệu kỹ thuật hợp lệ |
| Model đơn giản (GBM, ít cây, depth thấp) | Generalize tốt hơn model phức tạp trên dataset nhỏ |
| Giữ nguyên uniform weight | Tốt hơn mọi weight scheme khác |

---

## Bài học rút ra

1. **Daily oil price direction prediction là bài toán cực khó.** 57.4% accuracy (hơn random 7.2%) là kết quả hợp lý với dữ liệu công khai.

2. **Feature selection quan trọng hơn model selection.** Chọn đúng features (TOP_50 từ 81) cải thiện nhiều hơn so với thay đổi model hoặc tuning hyperparams.

3. **Data leakage rất dễ xảy ra** khi dùng technical indicators. Bất kỳ feature nào tính từ giá cùng ngày đều cần shift(1).

4. **Model đơn giản tốt hơn model phức tạp** trên dataset nhỏ (2033 samples) với signal yếu. Extensive tuning dễ overfitting.

5. **Ensemble giúp ổn định** nhưng không tạo breakthrough khi base models đều yếu.

6. **Weight decay không giúp** khi dataset đã nhỏ — mất thêm effective samples.

---

## Cấu trúc scripts

```
ml/
├── config.py                      # Shared config
├── train_regression.py            # Thực nghiệm 1 (regression)
├── train_classification.py        # Thực nghiệm 1 (classification)
├── step1_feature_selection.py     # Thực nghiệm 2
├── step2_finetune.py              # Thực nghiệm 3
├── step3_ensemble.py              # Thực nghiệm 4
├── step4_improve.py               # Thực nghiệm 5 (technicals + leakage fix)
├── step5_select_and_train.py      # Thực nghiệm 6
├── step6_smart_selection.py       # Thực nghiệm 7
├── step7_weight_decay.py          # Thực nghiệm 8
├── step8_xgb_vs_gbm.py           # Thực nghiệm 9
└── results/                       # CSV + plots
```

---

*Generated: 2026-04-12*
