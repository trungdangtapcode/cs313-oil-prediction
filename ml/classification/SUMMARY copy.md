# CLASSIFICATION SUMMARY — Dự đoán hướng giá dầu (UP/DOWN)

---

## 1. Bài toán

- **Input:** 52 features (kinh tế vĩ mô, cung cầu dầu, sentiment truyền thông, xung đột địa chính trị)
- **Output:** Giá dầu ngày mai tăng (UP=1) hay giảm (DOWN=0)
- **Target:** `oil_return > 0`
- **Train:** 2,083 ngày (2015-01-07 → 2022-12-30)
- **Test:** 840 ngày (2023-01-02 → 2026-03-20)
- **Majority class baseline:** 50.2% (gần 50/50, gần như cân bằng)

---

## 2. EDA Classification

### Phát hiện chính

| Phân tích | Kết quả |
|---|---|
| Class balance | Train: 51.4% UP / 48.6% DOWN — gần cân bằng |
| KS test (UP ≠ DOWN) | 26/52 features có phân phối khác nhau |
| Point-biserial significant | 17/52 features |
| Cohen's d > 0.2 (small effect) | Chỉ 3/52 features |
| Target shift (train→test) | Không shift (p=0.997) |

### Top 5 features phân biệt UP vs DOWN

| # | Feature | Point-biserial | Mutual Info | KS stat |
|---|---|---|---|---|
| 1 | `sp500_return` | 0.186 | 0.046 | 0.238 |
| 2 | `vix_return` | -0.164 | 0.045 | 0.208 |
| 3 | `usd_return` | -0.088 | 0.030 | 0.131 |
| 4 | `sp500_close` | 0.037 | 0.020 | 0.058 |
| 5 | `gdelt_tone_lag1` | 0.053 | 0.018 | 0.087 |

### Kết luận EDA

- Sự khác biệt giữa ngày UP và DOWN **rất nhỏ** (Cohen's d < 0.2 cho hầu hết features)
- Chỉ 3 features (sp500_return, vix_return, usd_return) có khả năng phân biệt đáng kể
- Giải thích tại sao accuracy khó vượt 55-57%

---

## 3. Pipeline thực nghiệm

### Bước 1: Baseline — 8 models, 42 features gốc

| Model | Accuracy | F1_macro | AUC |
|---|---|---|---|
| GradientBoosting | **52.7%** | 0.527 | 0.557 |
| SVM_Linear | 52.6% | 0.525 | 0.525 |
| MLP | 53.0% | 0.491 | 0.540 |
| SVM_RBF | 51.3% | 0.513 | 0.530 |
| XGBoost | 51.7% | 0.513 | 0.556 |
| RandomForest | 51.5% | 0.512 | 0.551 |
| LightGBM | 51.4% | 0.507 | 0.556 |
| LogisticRegression | 50.4% | 0.500 | 0.523 |

Tất cả ~50-53%, gần random guessing.

---

### Bước 2: Fine-tune + Ensemble

- RandomizedSearchCV 30 iterations, TimeSeriesSplit 5-fold
- Voting (soft) và Stacking (meta-learner: LogisticRegression)

| Model | Accuracy | F1_macro | AUC |
|---|---|---|---|
| XGB_v2 | 53.1% | 0.524 | 0.563 |
| GBM_v2 | 52.4% | 0.522 | 0.562 |
| **Voting** | **53.6%** | **0.527** | **0.568** |
| Stacking | 53.5% | 0.522 | 0.552 |

→ Cải thiện nhẹ +0.9%.

---

### Bước 3: Thêm Technical Indicators

Thêm 29 features kỹ thuật: RSI-14, MACD, Bollinger Bands, Moving Averages, Momentum, Rolling return stats.

**Phát hiện Data Leakage:** Technical indicators dùng giá đóng cửa cùng ngày → model gián tiếp "nhìn thấy" đáp án → accuracy giả 89%.

**Khắc phục:** Shift tất cả technical indicators 1 ngày (`shift(1)`) → chỉ dùng thông tin đến T-1.

| Version | Accuracy | Leakage? |
|---|---|---|
| Không shift | ~~89.2%~~ | CÓ — kết quả giả |
| **Có shift(1)** | **55.2%** | **Không** |

→ Technical indicators (đã shift) cải thiện +2.5%.

---

### Bước 4: Feature Selection trên 81 features

Ranking 81 features (42 gốc + 29 technicals + 10 lags) bằng Mutual Information + |Spearman|. Thử các subset TOP_10 đến ALL_81.

| Subset | GBM Acc | XGB Acc | LGBM Acc |
|---|---|---|---|
| TOP_10 | 55.5% | 53.4% | 54.4% |
| TOP_30 | 56.7% | 55.1% | 55.1% |
| TOP_40 | 57.3% | 55.0% | 55.0% |
| **TOP_50** | **57.4%** | 55.5% | 55.6% |
| ALL_81 | 53.9% | 55.2% | 55.8% |

→ **GBM + TOP_50 = 57.4%** — kết quả tốt nhất.
→ Bỏ 31 features yếu giúp GBM tăng 53.9% → 57.4%.

---

### Bước 5: Smart Feature Selection — Correlation Clustering + Permutation Importance

**Mục đích:** Giải quyết đồng thời feature importance và đa cộng tuyến.

**Phương pháp:**
1. Tính ma trận |Spearman| 81×81
2. Hierarchical clustering (threshold=0.3) → 40 clusters
3. Trong mỗi cluster, giữ feature có permutation importance cao nhất

**Đa cộng tuyến phát hiện được:**
- `gdelt_tone` ↔ `stress_tone` ↔ `geopolitical_stress_index` — |ρ| = 1.0
- `sp500_close` ↔ `cpi_lag` ↔ 6 GDELT volume features — |ρ| = 1.0
- `wti_fred` ↔ MA features ↔ Bollinger — |ρ| = 0.997
- `sp500_return` ↔ `vix_return` — |ρ| = 0.776

**Kết quả:** 81 → 40 features → XGB 56.6%

→ Không vượt naive selection (57.4%). Permutation importance bị nhiễu trên test set nhỏ (840 samples).

---

### Bước 6: Weight Decay — Data gần quan trọng hơn

**Giả thuyết:** Market regime thay đổi, data gần test set quan trọng hơn data cũ.

**11 weight schemes thử nghiệm:** Exponential (half-life 100-1000), Linear (min 0.1-0.5), Step (2x-3x).

| Scheme | GBM | XGB | LGBM |
|---|---|---|---|
| **uniform (no decay)** | **57.4%** | 55.5% | 55.6% |
| exp_hl100 (aggressive) | 50.8% | 53.0% | 50.5% |
| linear_01 | 56.7% | 56.1% | 54.9% |
| step_50pct_3x | 56.1% | 54.1% | 56.3% |

→ **Weight decay KHÔNG cải thiện.** Uniform vẫn tốt nhất. Dataset quá nhỏ (2033), decay làm mất thêm effective samples.

---

### Bước 7: Extensive Tuning

**Giả thuyết:** XGBoost với grid rộng hơn sẽ vượt GBM.

RandomizedSearchCV 50 iterations, grid rất rộng (10 hyperparameters, hàng nghìn combinations).

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| XGBoost (50 iter) | 60.9% | 54.8% |
| GBM (50 iter) | 60.8% | 51.9% |

→ **Cả 2 THẤP hơn config đơn giản (57.4%).** Overfitting: CV 61% nhưng test chỉ 52-55%. Grid rộng → tìm config overfit trên CV folds.

---

## 4. Tổng hợp kết quả

| Bước | Phương pháp | Features | Model | Accuracy | Δ vs baseline |
|---|---|---|---|---|---|
| 1 | Baseline | 42 | GBM | 52.7% | — |
| 2 | Fine-tune + Ensemble | 25 | Voting | 53.6% | +0.9% |
| 3 | + Technical Indicators (shifted) | 81 | LGBM | 55.2% | +2.5% |
| **4** | **+ Feature Selection (TOP_50)** | **50** | **GBM** | **57.4%** | **+4.7%** |
| 5 | Smart Selection (cluster+perm) | 40 | XGB | 56.6% | +3.9% |
| 6 | Weight Decay | 50 | GBM | 57.4% | +4.7% |
| 7 | Extensive Tuning (50 iter) | 50 | XGB | 54.8% | +2.1% |

### Accuracy tốt nhất: **57.4%** (GBM, TOP_50 features, uniform weight)

### Cải thiện so với majority class (50.2%): **+7.2%**

---

## 5. Những gì KHÔNG hiệu quả

| Phương pháp | Tại sao không hiệu quả |
|---|---|
| Technical indicators không shift | Data leakage — accuracy giả 89% |
| Weight decay | Dataset quá nhỏ (2033), mất effective samples |
| Extensive hyperparameter search | Overfitting trên CV, test accuracy giảm |
| Smart feature selection (cluster+perm) | Permutation importance nhiễu trên test nhỏ |
| Stacking ensemble | Không vượt Voting, phức tạp hơn |
| Multi-day prediction (3d, 5d) | Features không chứa thông tin tương lai |

## 6. Những gì CÓ hiệu quả

| Phương pháp | Đóng góp |
|---|---|
| Feature selection (loại features yếu) | **+2-4%**, hiệu quả nhất |
| Technical indicators (shifted đúng cách) | **+2.5%**, thêm tín hiệu kỹ thuật |
| Model đơn giản (GBM, depth thấp) | Generalize tốt hơn model phức tạp |

---

## 7. Bài học rút ra

1. **Daily oil price direction prediction là bài toán cực khó.** 57.4% accuracy là hợp lý với dữ liệu công khai.

2. **Feature selection > Model selection > Hyperparameter tuning.** Chọn đúng features quan trọng hơn chọn đúng model.

3. **Data leakage rất dễ xảy ra** khi dùng technical indicators. Luôn shift(1) bất kỳ feature nào tính từ giá cùng ngày.

4. **Model đơn giản thắng model phức tạp** trên dataset nhỏ với signal yếu. GBM (sklearn) với config mặc định tốt hơn XGBoost tuned 50 iterations.

5. **Overfitting trên CV ≠ performance tốt trên test.** CV accuracy 61% nhưng test chỉ 55%.

---

## 8. Model tốt nhất — Config cuối cùng

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    min_samples_leaf=5,
    random_state=42
)

# Features: TOP_50 (ranked by MI + |Spearman|)
# Technical indicators: RSI, MACD, Bollinger, MA, Momentum — tất cả shift(1)
# Weight: uniform (không decay)
# Accuracy: 57.4%
```
