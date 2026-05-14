# EDA Summary Report
## Oil Price Direction Classification

**Date**: 2026-04-11
**Dataset**: `data/processed/dataset_final.csv` — 2923 rows × 33 cols
**Train**: 2015-01-07 → 2022-12-30 (2083 rows)
**Test**: 2023-01-02 → 2026-03-20  (840 rows)

---

## 1. Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | **Class balance** | Up=51.4% / Down=48.6%. ✅ Acceptable balance |
| 2 | **Returns distribution** | Skewness≈negative, Kurtosis≈12 — strong fat tails (financial norm) |
| 3 | **ACF oil_return** | Near-zero → market close to random walk → prediction is hard |
| 4 | **Volatility clustering** | oil_volatility_7d shows strong ACF → GARCH-like features may help |
| 5 | **Feature signal** | 14/31 features significant by Mann-Whitney (p<0.05) |
| 6 | **Multicollinearity** | 12 features with VIF>10: ['gdelt_volume_lag1', 'cpi_lag', 'geopolitical_stress_index', 'gdelt_tone_7d', 'gdelt_tone_lag1', 'gdelt_tone_30d', 'gdelt_events', 'conflict_intensity_7d', 'unemployment_lag', 'vix_lag1', 'fatalities_7d', 'gdelt_goldstein'] |
| 7 | **Distribution shift** | 23/31 features shifted train→test (KS p<0.05) |
| 8 | **Seasonality** | Visible monthly pattern; high vol in COVID/Ukraine regimes |
| 9 | **Leakage** | oil_return/direction excluded from features; same-day market returns (medium risk — use lag versions for strict setup) |
| 10 | **Top predictors (MI)** | ['sp500_return', 'usd_return', 'vix_return', 'day_of_week', 'fatalities_7d'] |

---

## 2. Feature Ranking (Top 15)

| feature                   |   abs_r |   mutual_info |    VIF | recommendation    |
|:--------------------------|--------:|--------------:|-------:|:------------------|
| vix_return                |  0.2118 |    0.0332252  |   2.13 | KEEP              |
| sp500_return              |  0.2076 |    0.0492708  |   2.1  | KEEP              |
| usd_return                |  0.1113 |    0.047256   |   1.03 | KEEP              |
| fed_rate_change           |  0.0571 |    0.00392143 |   1.04 | KEEP              |
| inventory_change_pct      |  0.057  |    0          |   1.44 | CONSIDER REMOVING |
| day_of_week               |  0.0526 |    0.0286814  |   5.34 | KEEP              |
| conflict_intensity_7d     |  0.0508 |    0.00367033 |  29.1  | REMOVE            |
| gdelt_goldstein_7d        |  0.0501 |    0.0126911  |   4.91 | KEEP              |
| gdelt_tone_lag1           |  0.0463 |    0.0116254  |  85.06 | KEEP              |
| gdelt_events              |  0.0462 |    0.00550805 |  45.68 | KEEP              |
| oil_return_lag1           |  0.0413 |    0          |   1.04 | CONSIDER REMOVING |
| fatalities                |  0.0403 |    0          |   7.68 | CONSIDER REMOVING |
| geopolitical_stress_index |  0.0387 |    0.00546659 | 831.79 | KEEP              |
| gdelt_tone_7d             |  0.0368 |    0          | 182.25 | REMOVE            |
| inventory_zscore          |  0.0346 |    0.0137016  |   2.7  | KEEP              |

---

## 3. Feature Engineering Suggestions

- **Lag extension**: Add lag2/lag3 for `vix_lag`, `sp500_return`, `gdelt_tone` if ACF significant.
- **Strict leakage-free path**: Replace `usd_return`, `sp500_return`, `vix_return` with their `_lag1` versions.
- **Interaction**: `real_rate × geopolitical_stress_index` (macro × geo).
- **Regime flag**: Binary: `high_vol_regime = (oil_volatility_7d > Q75)`.
- **Transform**: log-transform highly skewed supply features if using linear models.

---

## 4. Modeling Strategy

| Decision | Recommendation |
|----------|---------------|
| Validation | `TimeSeriesSplit(n_splits=5)` — NO random k-fold |
| Baseline | "Always predict majority class" → ~51.4% accuracy |
| Metric | AUC-ROC + F1 (macro or weighted) |
| Class imbalance | `class_weight='balanced'` first; SMOTE if needed |
| Algorithm order | Logistic Regression → Random Forest → XGBoost |
| Feature scaling | RobustScaler (handles fat-tails better than StandardScaler) |

---

## 5. Risks & Concerns

- **Low signal**: Market near-random-walk → realistic ceiling ~55-60% accuracy.
- **Distribution shift**: 23 features drifted train→test — monitor model performance over time.
- **Regime mismatch**: Train (2015-2022) includes COVID & oil crash; Test (2023+) is post-war inflation era.
- **Same-day features**: usd_return/sp500_return/vix_return are same-day — use `_lag1` in strict setting.
- **Overfitting**: 31 features vs 2083 train rows — regularize aggressively.

---

## 6. Conclusion

EDA confirms this is a **hard classification task** (near-random-walk target). Signals exist but are weak.
Key features: lag returns, volatility, geopolitical stress, macro regime.
Proceed with strict train-only preprocessing, TimeSeriesSplit CV, and baseline comparison first.
