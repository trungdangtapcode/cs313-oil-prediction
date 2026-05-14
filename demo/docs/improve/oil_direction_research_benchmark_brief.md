# Research & Benchmark Brief: Daily Next-Day Oil Price Direction

**Generated:** 2026-05-14  
**Format:** Markdown, coding-agent friendly  
**Use case:** Internal team context before implementing models for daily next-day crude oil direction prediction.

---

## 0. Executive summary

The target problem is:

```text
Input at date t: market / macro / sentiment / supply features known as-of date t
Target: direction of crude oil price from t to t+1
Frequency: daily
Label: UP/DOWN, usually y_t = 1[log(close_{t+1}/close_t) > 0]
```

The most important conclusion from the research discussion is:

```text
Reported literature results are not the same as a realistic engineering KPI.
```

Some papers report `0.70-0.85` directional accuracy, but many of those results are not directly comparable to the repository's current setup because they may use regression-to-direction, smoothed labels, graph/deep-learning architectures, weekly horizons, special preprocessing, or older validation standards. For the repository's current daily next-day classification setup, a realistic target is closer to:

```text
Accuracy 0.53-0.56: acceptable baseline
Accuracy 0.56-0.60: good
Accuracy 0.60-0.63: very good
Accuracy >0.63: audit carefully before trusting
Accuracy 0.70+: not a default KPI for this setup

AUC 0.56-0.60: acceptable
AUC 0.60-0.64: strong
AUC >0.64: audit carefully
```

Current repo result, according to the uploaded benchmark report:

```text
Best Accuracy = 0.5452  (LGBM, CLUSTER_POS_10)
Best AUC      = 0.5586  (XGBoost)
```

Interpretation:

```text
The repo is above no-skill and has some signal.
It is not weak for daily oil direction.
It is not yet in the strong 0.58-0.61 accuracy / 0.60+ AUC region.
```

---

## 1. Scope and target definition

### 1.1 Base target

For each trading day `t`:

```text
ret_next_t = log(close_{t+1} / close_t)
y_dir_t    = 1 if ret_next_t > 0 else 0
```

Use this as the main benchmark label.

### 1.2 Optional threshold label

Daily oil returns are noisy. A secondary label can remove small moves:

```text
y_dir_t = 1 if ret_next_t > +threshold
y_dir_t = 0 if ret_next_t < -threshold
ignore / neutral if abs(ret_next_t) <= threshold
```

Candidate thresholds:

```text
threshold = transaction_cost
threshold = 0.1% / 0.2% / 0.3%
threshold = rolling volatility quantile
```

Do not compare threshold-label accuracy directly with raw UP/DOWN accuracy unless the class coverage is also reported.

### 1.3 Forecast timing

The intended prediction is:

```text
At or after close on date t, forecast direction for date t+1.
```

Every feature must be known by the prediction cutoff. This is the most important leakage control.

---

## 2. Corrected interpretation of the benchmark debate

### 2.1 What was initially confusing

The earlier survey mixed two things:

```text
A. Results reported by papers
B. Results that should be used as KPIs for this repo
```

These are different.

A paper can report high accuracy, but if it uses a different target, different horizon, regression-to-direction, special architecture, non-daily data, or opaque validation, then its number should be treated as `reported claim`, not as `repo benchmark`.

### 2.2 Practical rule

Use the literature as follows:

```text
Use high-claim papers to get feature/model ideas.
Use conservative daily-direction papers to set KPI expectations.
Use leakage/data-matter papers to define audit rules.
```

---

## 3. Current repo status and KPI bands

### 3.1 Current repo result from uploaded benchmark report

```text
Best saved Accuracy = 0.5452
File: ml/classification/results_step5b_v2/step5_results.csv
Model: LGBM
Feature set: CLUSTER_POS_10

Best saved AUC = 0.5586
File: ml/classification/results_step5b_v2/step7_test_results.csv
Model: XGBoost
```

### 3.2 Interpretation

```text
0.5452 accuracy is not random guessing.
0.5586 AUC indicates weak but plausible signal.
The result is in the acceptable baseline band for daily oil direction.
```

### 3.3 Recommended KPI bands

#### Accuracy

| Band | Interpretation | Action |
|---:|---|---|
| `<0.50` | worse than random | check label/model direction |
| `0.50-0.53` | very weak | improve features/evaluation |
| `0.53-0.56` | acceptable baseline | current repo is here |
| `0.56-0.60` | good | meaningful progress |
| `0.60-0.63` | very good | strong if walk-forward clean |
| `0.63-0.66` | unusually strong | audit leakage/selection bias |
| `>0.70` | not default target | requires independent replication |

#### AUC

| Band | Interpretation | Action |
|---:|---|---|
| `<0.56` | weak / baseline | improve features |
| `0.56-0.60` | acceptable | current repo is near here |
| `0.60-0.64` | strong | good target |
| `>0.64` | very strong | audit carefully |

### 3.4 Suggested KPI for next phase

```text
Near-term target:
  Accuracy >= 0.56
  AUC      >= 0.58

Good target:
  Accuracy 0.58-0.60
  AUC      0.60-0.62

Stretch target:
  Accuracy 0.61-0.63
  AUC      0.63-0.65

Any result above this:
  audit first, celebrate second.
```

---

## 4. Literature map: data, features, methods, results, and comparability

### 4.1 Summary table

| ID | Paper / source | Problem and data | Feature groups | Method | Reported result | How to use for this repo |
|---:|---|---|---|---|---|---|
| 1 | Luo et al. (2019), *Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks* | Daily crude oil futures, short-term forecasting, reports directional accuracy | Price/market-derived features | NF, NN, CNN | Uploaded benchmark report states: NF DA ~0.495, NN best ~0.575, CNN full-sample ~0.595, some subperiods ~0.525 | Most useful conservative benchmark for daily direction. Treat 0.57-0.60 as good. |
| 2 | Kulkarni & Haidar (2009), *Forecasting Model for Crude Oil Price Using Artificial Neural Networks and Commodity Futures Prices* | Short-term spot price direction, 1-3 days ahead | Lagged spot price + futures prices 1-4 months | Multilayer feedforward ANN | 78%, 66%, 53% for 1/2/3 days | High-claim historical reference. Use feature idea: futures curve. Do not use as KPI. |
| 3 | Pan, Haidar & Kulkarni (2009), *Daily prediction of short-term trends of crude oil prices using neural networks exploiting multimarket dynamics* | Daily WTI spot/futures, 1-3 day trend | Spot, futures, S&P 500, Dollar Index, Gold, Heating Oil | ANN with multimarket dynamics | Earlier review cited ~79.95%, 69.74%, 60.64% for t+1/t+2/t+3 | Same family of high-claim old ANN results. Use as feature inspiration, not default KPI. |
| 4 | Cohen (2025), *A Comprehensive Study on Short-Term Oil Price Forecasting Using Econometric and Machine Learning Techniques* | Daily short-term oil price forecasting, 2015-2024, 21 series | VIX, OVX, MOVE, lagged oil returns, Gold, S&P 500, etc. | Econometric + ML, meta-learner / stacking | Reported RMSE 2.04, MAE 1.65, test R² 0.532; directional accuracy ~71.4%-79.7% in paper | Very useful for feature ideas: OVX/MOVE/Gold/stacking. Do not compare the high DA directly with repo classification. |
| 5 | Foroutan & Lahmiri (2024), *Deep Learning-Based Spatial-Temporal Graph Neural Networks for Price Movement Classification in Crude Oil and Precious Metal Markets* | Price movement classification for WTI/Brent/gold/silver | Multivariate market + technical indicators | ST-GNN variants, attention, graph models | Earlier review cited WTI accuracy in mid-80% range | Strong but graph-specific. Use only as advanced research track, not baseline KPI. |
| 6 | Li, Shang & Wang (2019), *Text-based crude oil price forecasting* | Crude oil price forecasting with online news and financial data | News text, sentiment, topics, financial variables | CNN text extraction, sentiment, LDA/topic, VAR/RF/SVR/linear models | Reported text+financial improves forecasting error for SVR/RF in earlier review | Use for sentiment feature pipeline. Not direct direction benchmark. |
| 7 | Bai et al. (2022), *Crude oil price forecasting incorporating news text* | Forecasting oil price using short/sparse headlines | News headline topic and dynamic sentiment indicators | SeaNMF/topic, dynamic sentiment, AdaBoost.RT | AdaBoost.RT + textual indicators outperforms benchmarks; earlier review cited RMSE/MAE/MAPE values | Use for short headline sentiment/topic engineering. Not direct direction benchmark. |
| 8 | Gao et al. (2022), explainable ML during COVID-19 | Crude oil forecasting in COVID period | Market, macro, COVID, web-search, inventories | RF/XGB/LGBM/CatBoost + SHAP | LGBM generalized best in earlier review | Use for SHAP/explainable feature-rich setup. COVID-specific, not universal KPI. |
| 9 | Gupta & Yoon (2018), OPEC news predictability | Predictability of oil futures returns/volatility | OPEC-related variables, meeting dates, production announcements | Nonlinear quantile causality | Significant nonlinear predictability, especially Brent in lower quantiles | Justifies OPEC event features. No accuracy benchmark. |
| 10 | Derbali, Wu & Jamel (2020), OPEC news and energy futures | Energy futures returns/volatility | OPEC meeting dates and production announcements | Conditional quantile regression / quantile causality | OPEC news statistically significant for several energy futures | Justifies OPEC/supply-event features. No accuracy benchmark. |
| 11 | Li et al. (2024), OPEC+ policy index | Oil price forecasting using news-based OPEC+ policy index | News-mined OPEC+ production-decision index | Text mining + forecasting models | Weekly OPEC+ policy index has predictive effect; daily effect also tested | Use as OPEC+ policy/supply sentiment feature. Not daily classifier KPI. |
| 12 | Li et al. (2025), improved sentiment lexicon | Oil price fluctuation forecasting with news sentiment | Oil-specific news sentiment categories, supply/demand/finance | Improved lexicon + GRU/LSTM/BP + SHAP | Earlier review cited GRU directional accuracy 0.875, but weekly data | Excellent sentiment-design reference. Not directly comparable to daily next-day setup. |
| 13 | Mohsin & Jamaani (2023), green finance and socio-politico-economic factors | Oil price prediction with 26 factors | Commodity, geopolitical, supply, demand, financial, green finance | LASSO vs OLS/GARCH/EIA/ANN | LASSO beats benchmarks across multiple horizons in paper summary | Use for macro/demand/geopolitical feature ideas. Not daily direction KPI. |
| 14 | Bu (2014), inventory announcements | Effect of EIA inventory reports on returns/volatility | Inventory information shocks, not just raw inventory changes | GARCH / event-style analysis | Inventory information shocks affect crude oil returns and price level; not daily volatility | Strong justification for EIA inventory surprise features. No classifier metric. |
| 15 | Conlon et al. (2022), *The Illusion of Oil Return Predictability: The Choice of Data Matters* | Methodological critique of oil return predictability | Data definition, return construction, benchmark choices | Empirical re-examination | Predictability can be overstated by data choices such as averaging | Use as audit philosophy: high results need careful scrutiny. |

---

## 5. Important correction: Luo et al. vs. Huang and Wang

The uploaded benchmark report refers to `Huang and Wang (2019)` for the MDPI paper at:

```text
https://www.mdpi.com/1911-8074/12/1/9
```

The correct article metadata is:

```text
Luo et al. (2019)
Title: Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks
```

Team should cite the corrected author/title when writing reports or papers.

---

## 6. Recommended dataset design

### 6.1 One-row-per-trading-day table

Every row should represent one tradable date `t`.

Minimal schema:

```text
date
asset_id                 # WTI or Brent
close_t
ret_1d_t
ret_next_t
y_dir_next_t
feature_market_*
feature_curve_*
feature_macro_*
feature_supply_*
feature_sentiment_*
split_id
fold_id
```

### 6.2 Target columns

```text
ret_next = log(close_{t+1} / close_t)
y_raw    = 1 if ret_next > 0 else 0
```

Optional:

```text
y_threshold_10bp = 1 / 0 / neutral based on +/-10bp threshold
y_triple_barrier = label based on upper/lower barrier hit before horizon H
```

### 6.3 Market and technical features

Examples:

```text
wti_ret_1d
wti_ret_5d
wti_ret_20d
brent_ret_1d
brent_wti_spread
realized_vol_5
realized_vol_20
atr_14
rsi_14
macd
ma_ratio_5_20
ma_ratio_20_60
volume_change
open_interest_change
```

### 6.4 Futures curve features

If futures data is available, this is one of the most important upgrades.

Examples:

```text
cl1_close
cl2_close
cl3_close
cl6_close
cl12_close
cl1_cl2_spread
cl1_cl3_spread
cl1_cl6_spread
term_structure_slope_1_6
backwardation_flag
roll_yield_proxy
curve_pc1
curve_pc2
curve_pc3
```

Rationale:

```text
Oil futures term structure reflects storage, supply/demand tightness, and market expectations.
Old ANN papers also found futures prices useful for short-term spot direction.
```

### 6.5 Intermarket features

Examples:

```text
spx_ret_1d
dxy_ret_1d
gold_ret_1d
silver_ret_1d
heating_oil_ret_1d
gasoline_ret_1d
natgas_ret_1d
copper_ret_1d
```

Rationale:

```text
Cohen (2025) highlights Gold and S&P 500.
Pan et al. used S&P 500, Dollar Index, Gold, Heating Oil.
Heating Oil can be relevant because it is an energy-product lead/related market.
```

### 6.6 Volatility and risk features

Examples:

```text
vix_level
vix_change_1d
ovx_level
ovx_change_1d
move_level
move_change_1d
credit_spread
```

Rationale:

```text
Cohen (2025) emphasizes VIX, OVX, MOVE as important predictors.
OVX is specifically the Cboe Crude Oil ETF Volatility Index, a proxy for expected 30-day volatility in oil-linked ETF options.
```

### 6.7 Macro features

Examples:

```text
us10y_yield_change
us2y_yield_change
yield_curve_10y2y
usd_index_change
cpi_surprise
pmi_surprise
ism_surprise
nfp_surprise
fed_event_day
```

Handling rule:

```text
Only use macro data after official release timestamp.
For daily rows, store both latest known value and days_since_release.
```

### 6.8 Supply and inventory features

Examples:

```text
eia_crude_inventory_change
eia_crude_inventory_surprise
eia_cushing_inventory_change
eia_gasoline_inventory_change
eia_distillate_inventory_change
eia_production
refinery_utilization
crude_imports
crude_exports
api_inventory_surprise
baker_hughes_oil_rig_count
baker_hughes_oil_rig_change
days_since_eia_release
days_since_rig_release
```

Very important:

```text
Do not forward-fill an EIA value before it is released.
The feature becomes available only after the release time.
```

### 6.9 OPEC / supply-event features

Examples:

```text
opec_meeting_day
opec_plus_meeting_day
opec_announcement_day
opec_cut_flag
opec_hike_flag
opec_hold_flag
opec_cut_size_bpd
opec_surprise_vs_expectation
days_since_opec_event
opec_news_sentiment
opec_policy_index
```

Use these as event/sentiment variables rather than expecting a daily continuous signal.

### 6.10 Sentiment and text features

Minimum viable sentiment features:

```text
news_count
headline_count
sent_mean
sent_std
sent_pos_ratio
sent_neg_ratio
topic_opec_sent
topic_inventory_sent
topic_demand_sent
topic_macro_sent
topic_geopolitical_sent
sent_decay_3d
sent_decay_5d
```

Better text pipeline:

```text
1. Collect timestamped oil-related headlines/articles.
2. Filter for relevant categories: supply, OPEC, inventory, demand, macro, finance, geopolitics.
3. Compute sentiment using oil-specific lexicon or finetuned financial model.
4. Aggregate by market cutoff time, not calendar day blindly.
5. Add decayed sentiment because news impact can persist.
```

Recommended hierarchy:

```text
Level 1: headline count + simple lexicon sentiment
Level 2: topic-level sentiment with LDA/SeaNMF/clustering
Level 3: FinBERT / oil-domain sentiment classifier
Level 4: LLM-assisted event extraction for OPEC/supply shocks
```

---

## 7. As-of joining rules

### 7.1 Daily market data

For close-to-close prediction:

```text
Features computed from close_t are allowed if prediction is made after close_t.
Features from close_{t+1} are not allowed.
```

### 7.2 Weekly EIA inventory

Correct structure:

```text
date        close_t   eia_inv_chg_asof   eia_surprise_asof   days_since_eia
2024-01-08  ...       last_known_value    0                   5
2024-01-09  ...       last_known_value    0                   6
2024-01-10  ...       new_release_value   surprise_value      0
2024-01-11  ...       same_new_value      0                   1
```

If EIA is released after the prediction cutoff, use it only starting next eligible row.

### 7.3 Monthly macro

Correct structure:

```text
latest_value_asof
release_surprise_on_release_day
days_since_release
revision_flag
```

Do not use revised historical values unless the same vintage would have been available historically.

### 7.4 News/sentiment

For each news item:

```text
published_timestamp_utc
market_timestamp_local
asset_relevance
category
sentiment_score
```

Aggregate only news published before the prediction cutoff.

---

## 8. Recommended modeling stack

### 8.1 Baselines

Always include:

```text
majority_class
random
sign_persistence: predict sign(ret_t)
logistic_regression_l1_l2
elastic_net_logistic
simple_arimax_or_garch_sign_proxy
```

### 8.2 Main tabular models

Recommended first wave:

```text
LightGBM
XGBoost
CatBoost
Random Forest
ExtraTrees
SVM-RBF, only if feature count/sample size manageable
```

Notes:

```text
Tree boosting is usually strong for heterogeneous tabular features.
CatBoost is useful with categorical event variables.
LightGBM/XGBoost are good for ablation and feature importance.
```

### 8.3 Sequence models

Only after tabular baseline is clean:

```text
LSTM
GRU
TCN
Transformer / iTransformer
```

Use sequence models when you can represent a rolling window like:

```text
X_t = [features_{t-L+1}, ..., features_t]
y_t = direction_{t+1}
```

### 8.4 Graph / multi-asset models

Advanced research track:

```text
nodes = assets / markets / indicators
edges = correlation, Granger, learned adjacency, or economic relation
models = ST-GNN, MTGNN, ASTGCN, attention-based graph models
```

This is relevant to Foroutan & Lahmiri-style results, but not required for the first engineering sprint.

### 8.5 Text/sentiment models

Recommended evolution:

```text
Sprint 1: lexicon/topic aggregate features
Sprint 2: FinBERT or finance-domain transformer sentiment
Sprint 3: oil-domain weak supervision / LLM-labeled training set
Sprint 4: event extraction for OPEC, inventory, sanctions, geopolitics
```

---

## 9. Validation design

### 9.1 Never use random split

Use time-series evaluation only.

Recommended:

```text
expanding window
rolling window
purged walk-forward split
```

Example:

```text
Fold 1 train: 2015-2018, valid: 2019, test: 2020
Fold 2 train: 2015-2019, valid: 2020, test: 2021
Fold 3 train: 2015-2020, valid: 2021, test: 2022
...
```

### 9.2 Fit preprocessing only on train

For every fold:

```text
fit imputer on train only
fit scaler on train only
fit feature selector on train only
fit PCA/UMAP/clustering on train only
fit model on train only
tune on validation only
evaluate once on test
```

### 9.3 Metrics

Classification metrics:

```text
Accuracy
Balanced Accuracy
F1
MCC
ROC-AUC
PR-AUC
Log-loss
Brier score
```

Trading sanity metrics:

```text
hit_ratio
average_return_per_trade
turnover
transaction_cost_adjusted_pnl
Sharpe
max_drawdown
calibration_by_probability_bucket
```

### 9.4 Ablation design

Run in this order:

```text
A0: baseline only
A1: market + technical
A2: A1 + futures curve
A3: A2 + volatility/risk (VIX/OVX/MOVE)
A4: A3 + macro
A5: A4 + supply/inventory
A6: A5 + OPEC event features
A7: A6 + sentiment/news
A8: A7 + ensemble/stacking
```

Report all rows, not just the best one.

---

## 10. What results are plausible?

### 10.1 Main judgment

For clean daily next-day oil direction:

```text
0.545 accuracy / 0.559 AUC is plausible and not bad.
0.58-0.60 accuracy would be strong progress.
0.60+ accuracy is already very good.
0.70+ should not be expected without a different setup or very strong new signal.
```

### 10.2 When 0.70+ could happen

Possible scenarios:

```text
label is thresholded or neutral days are removed
horizon is weekly rather than daily
model predicts event-day direction only
features include high-quality futures/options/sentiment data
there is a special graph/sequence architecture with a strong multivariate signal
there is leakage or loose validation
```

The last explanation must be ruled out first.

---

## 11. Recommended next research tracks

### Track A: Clean daily benchmark upgrade

Goal:

```text
Improve current daily classifier from Accuracy 0.545 / AUC 0.559 toward Accuracy 0.58 / AUC 0.60.
```

Add features in this order:

```text
OVX
MOVE
Gold
S&P 500
DXY
Treasury yields
futures curve spreads
EIA inventory surprise
Baker Hughes rig count
opec event flags
better text sentiment
```

### Track B: Threshold / triple-barrier labels

Goal:

```text
Reduce label noise and make predictions more trade-relevant.
```

Do not compare raw accuracy with thresholded accuracy unless coverage is reported.

### Track C: Sentiment/supply-event extraction

Goal:

```text
Build oil-specific text features rather than generic aggregate GDELT tone.
```

Prioritize:

```text
OPEC/OPEC+
inventory
production cuts/hikes
sanctions/geopolitics
demand slowdown/growth
refinery outages
shipping disruptions
```

### Track D: Weekly side experiment

Goal:

```text
Test whether macro/supply/sentiment features become more stable at lower frequency.
```

Caution:

```text
weekly sample size is smaller
variance across folds will be larger
use simpler models and stricter validation
```

---

## 12. Source list for team

### Daily / direction / deep-learning benchmark

- Luo et al. (2019), *Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks*  
  https://www.mdpi.com/1911-8074/12/1/9

### Old ANN high-claim references

- Kulkarni & Haidar (2009), *Forecasting Model for Crude Oil Price Using Artificial Neural Networks and Commodity Futures Prices*  
  https://arxiv.org/abs/0906.4838

- Pan, Haidar & Kulkarni (2009), *Daily prediction of short-term trends of crude oil prices using neural networks exploiting multimarket dynamics*  
  https://link.springer.com/article/10.1007/s11704-009-0025-3

### Short-term ensemble / feature-rich study

- Cohen (2025), *A Comprehensive Study on Short-Term Oil Price Forecasting Using Econometric and Machine Learning Techniques*  
  https://www.mdpi.com/2504-4990/7/4/127

### Graph / sequence advanced track

- Foroutan & Lahmiri (2024), *Deep Learning-Based Spatial-Temporal Graph Neural Networks for Price Movement Classification in Crude Oil and Precious Metal Markets*  
  https://www.sciencedirect.com/science/article/pii/S2666827024000288

### Text / sentiment

- Li, Shang & Wang (2019), *Text-based crude oil price forecasting: A deep learning approach*  
  https://www.sciencedirect.com/science/article/pii/S0169207018301110

- Bai et al. (2022), *Crude oil price forecasting incorporating news text*  
  https://www.sciencedirect.com/science/article/pii/S0169207021001060

- Li et al. (2025), *Crude oil price fluctuation forecasting incorporating news sentiment based on improved sentiment lexicon*  
  https://link.springer.com/article/10.1007/s44443-025-00289-8

### OPEC / supply policy / inventory

- Bu (2014), *Effect of inventory announcements on crude oil price volatility*  
  https://www.sciencedirect.com/science/article/pii/S0140988314001236

- Li et al. (2024), *Do OPEC+ policies help predict the oil price: A novel news-based predictor*  
  https://www.sciencedirect.com/science/article/pii/S2405844024104689

### Macro / broad feature set

- Mohsin & Jamaani (2023), *Green finance and the socio-politico-economic factors' impact on the future oil prices*  
  https://www.sciencedirect.com/science/article/pii/S0301420723004919

### Methodological caution

- Conlon et al. (2022), *The Illusion of Oil Return Predictability: The Choice of Data Matters*  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3841507

### Data sources

- FRED WTI daily price, DCOILWTICO  
  https://fred.stlouisfed.org/series/DCOILWTICO

- EIA Weekly Petroleum Status Report  
  https://www.eia.gov/petroleum/supply/weekly/

- EIA weekly release schedule  
  https://www.eia.gov/petroleum/supply/weekly/schedule.php

- Cboe OVX dashboard  
  https://www.cboe.com/us/indices/dashboard/ovx/

- FRED OVX daily close  
  https://fred.stlouisfed.org/series/OVXCLS

- Baker Hughes rig count  
  https://rigcount.bakerhughes.com/

---

## 13. Final decision for team

Default path:

```text
Keep daily next-day direction as the main benchmark.
Treat current result as a valid baseline, not a failure.
Do not set 0.70-0.80 as KPI.
Aim for Accuracy 0.58-0.61 and AUC 0.60-0.64.
Add features with strict as-of timing.
Run ablations and leakage audit before claiming improvement.
```

Side path:

```text
Open weekly experiment separately.
Do not mix weekly results with daily benchmark.
```
