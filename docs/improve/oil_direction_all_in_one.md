# All-in-One Team Document: Daily Next-Day Oil Price Direction

**Generated:** 2026-05-14  
**Format:** Markdown, coding-agent friendly  
**Purpose:** Single consolidated document containing the research context, benchmark interpretation, implementation plan, and leakage checklist for the team.

---

## Part A. Core context

The team is working on:

```text
Task: predict next-day crude oil price direction
Frequency: daily
Horizon: T -> T+1
Target: UP/DOWN
Feature groups: market, macro, sentiment, supply, volatility/risk, futures curve
```

Base label:

```text
ret_next_t = log(close_{t+1} / close_t)
y_t        = 1 if ret_next_t > 0 else 0
```

The main lesson from the research discussion:

```text
Do not confuse paper-reported results with realistic repo KPIs.
```

Reported literature may show `0.70-0.85`, but the clean daily next-day setup should use more conservative benchmarks.

Current repo status from the uploaded report:

```text
Best Accuracy = 0.5452
Best AUC      = 0.5586
```

This is not bad. It is a plausible weak edge for daily oil direction.

---

## Part B. Realistic KPI bands

### Accuracy

```text
0.50-0.53: very weak / near no-skill
0.53-0.56: acceptable baseline
0.56-0.60: good
0.60-0.63: very good
0.63-0.66: unusually strong, audit carefully
0.70+    : not default target, replicate independently
```

### AUC

```text
<0.56    : weak
0.56-0.60: acceptable
0.60-0.64: strong
>0.64    : audit carefully
```

### Recommended next goals

```text
Near-term:
  Accuracy >= 0.56
  AUC      >= 0.58

Good:
  Accuracy 0.58-0.60
  AUC      0.60-0.62

Stretch:
  Accuracy 0.61-0.63
  AUC      0.63-0.65
```

---

## Part C. Literature summary

| Paper | Setup | Data/features | Method | Reported result | Benchmark use |
|---|---|---|---|---|---|
| Luo et al. 2019, CNN daily oil futures | Daily oil futures, short-term | Price/market features | NF, NN, CNN | Uploaded report: NF ~0.495 DA, NN ~0.575, CNN ~0.595 | Best conservative daily benchmark |
| Kulkarni & Haidar 2009 | Spot direction 1-3 days | Spot + futures 1-4 months | ANN | 78%, 66%, 53% | High-claim old reference, not KPI |
| Pan, Haidar & Kulkarni 2009 | Daily WTI trend 1-3 days | Spot, futures, SPX, DXY, Gold, Heating Oil | ANN | Earlier review: ~79.95%, 69.74%, 60.64% | Feature inspiration, not KPI |
| Cohen 2025 | Daily short-term price forecasting | 21 series, VIX/OVX/MOVE/Gold/SPX | Econometric + ML + stacking | R² 0.532, DA ~71.4%-79.7% | Feature ideas; not direct classifier benchmark |
| Foroutan & Lahmiri 2024 | Price movement classification | Multi-market + technical | ST-GNN / graph models | Earlier review: WTI mid-80% accuracy | Advanced graph track only |
| Li, Shang & Wang 2019 | Price forecasting with news | News text + financial data | CNN text + topic/sentiment + ML | Text+financial improves error | Sentiment feature design |
| Bai et al. 2022 | Forecasting with headlines | Topic + dynamic sentiment | SeaNMF + AdaBoost.RT | Text indicators outperform benchmarks | Short headline pipeline |
| Gao et al. 2022 | Forecasting during COVID | Market/macro/COVID/search/inventory | LGBM, XGB, RF, CatBoost + SHAP | LGBM best in review | SHAP and feature-rich design |
| Gupta & Yoon 2018 | OPEC predictability | OPEC meeting/production variables | Quantile causality | Nonlinear effects, especially Brent | OPEC event feature justification |
| Derbali et al. 2020 | OPEC news vs energy futures | OPEC news/events | Conditional quantile regression | Significant predictive relation | Supply-event justification |
| Li et al. 2024 | OPEC+ policy index | News-mined OPEC+ production policy | Text mining + ML/econometric | Predictive weekly/daily effect | OPEC policy feature |
| Li et al. 2025 | News sentiment fluctuation forecasting | Oil-specific sentiment categories | Lexicon + GRU/LSTM/BP + SHAP | Earlier review: DA 0.875 weekly | Good sentiment design, not daily KPI |
| Mohsin & Jamaani 2023 | Broad factors and green finance | 26 socio-politico-economic factors | LASSO | Beats benchmark models | Macro/demand/geopolitical feature ideas |
| Bu 2014 | EIA inventory announcements | Inventory information shocks | GARCH/event analysis | Shocks affect returns/price level | Inventory surprise feature justification |
| Conlon et al. 2022 | Data-choice critique | Oil return data construction | Re-examination | Predictability can be overstated | Audit philosophy |

---

## Part D. Corrected citation note

The uploaded benchmark report labels the MDPI daily/CNN paper as `Huang and Wang (2019)`. The link points to Luo et al. (2019):

```text
Luo et al. (2019)
Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks
https://www.mdpi.com/1911-8074/12/1/9
```

Use this corrected citation in future team docs.

---

## Part E. Dataset specification

Each row is one trading date:

```text
date
asset_id
close_t
ret_next_t
y_dir_next_t
market features
curve features
volatility/risk features
macro features
supply/inventory features
OPEC/event features
sentiment features
fold_id
split_id
```

### Feature groups

#### Market/technical

```text
wti_ret_1d, wti_ret_5d, wti_ret_20d
realized_vol_5, realized_vol_20
rsi_14, macd, atr_14
volume_change, open_interest_change
```

#### Futures curve

```text
cl1_close, cl2_close, cl3_close, cl6_close
cl1_cl2_spread, cl1_cl3_spread, cl1_cl6_spread
backwardation_flag, contango_flag, roll_yield_proxy
curve_pc1, curve_pc2, curve_pc3
```

#### Intermarket

```text
spx_ret_1d, dxy_ret_1d, gold_ret_1d
heating_oil_ret_1d, gasoline_ret_1d, natgas_ret_1d
```

#### Volatility/risk

```text
vix_level, vix_change_1d
ovx_level, ovx_change_1d
move_level, move_change_1d
```

#### Macro

```text
us2y_change, us10y_change, yield_curve_10y2y
cpi_surprise, pmi_surprise, ism_surprise, nfp_surprise
fed_event_day
```

#### Supply/inventory

```text
eia_crude_inventory_change
eia_crude_inventory_surprise
eia_cushing_inventory_change
eia_production
refinery_utilization
crude_imports, crude_exports
api_inventory_surprise
baker_hughes_oil_rig_count
rig_count_change
days_since_eia_release
days_since_rig_release
```

#### OPEC/supply event

```text
opec_meeting_day
opec_plus_meeting_day
opec_cut_flag
opec_hike_flag
opec_hold_flag
opec_cut_size_bpd
opec_surprise_vs_expectation
days_since_opec_event
opec_news_sentiment
opec_policy_index
```

#### Sentiment/news

```text
news_count
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

---

## Part F. As-of rules

Core rule:

```text
A feature can be used only if it is known before the prediction cutoff.
```

### EIA weekly inventory

Wrong:

```text
Forward-fill the week-ending inventory value before EIA release.
```

Correct:

```text
Use actual EIA release timestamp.
Feature becomes available only after release.
Use days_since_eia_release.
```

### Macro releases

```text
Use release timestamp.
Use actual - consensus surprise.
Do not use revised history unless vintage data is available.
```

### News

```text
Use publication timestamp.
Aggregate only news before cutoff.
Do not aggregate the whole calendar day blindly.
```

---

## Part G. Modeling plan

### Baselines

```text
majority class
random
sign persistence
logistic regression
elastic-net logistic
```

### Main tabular models

```text
LightGBM
XGBoost
CatBoost
Random Forest
ExtraTrees
```

### Advanced models

```text
GRU / LSTM / TCN / Transformer
ST-GNN / MTGNN / ASTGCN
```

Do advanced models after the tabular pipeline is clean.

---

## Part H. Evaluation design

Use walk-forward only:

```text
Fold 1: train 2015-2018, valid 2019, test 2020
Fold 2: train 2015-2019, valid 2020, test 2021
Fold 3: train 2015-2020, valid 2021, test 2022
...
```

Never use:

```text
random split
shuffled cross-validation
full-sample scaler/imputer/feature selector
```

Metrics:

```text
Accuracy
Balanced Accuracy
F1
MCC
ROC-AUC
PR-AUC
Log-loss
Brier score
PnL after costs
Sharpe
Max drawdown
```

---

## Part I. Ablation roadmap

```text
A0: baseline only
A1: market + technical
A2: A1 + futures curve
A3: A2 + VIX/OVX/MOVE
A4: A3 + intermarket
A5: A4 + macro
A6: A5 + supply/inventory
A7: A6 + OPEC event features
A8: A7 + sentiment/news
A9: A8 + stacking
```

Acceptance rule:

```text
A feature group is useful only if it improves average walk-forward metrics and is stable across folds.
```

---

## Part J. Leakage checklist

Run this before accepting any result:

```text
[ ] Target uses close_{t+1}/close_t exactly.
[ ] No feature uses future price/return.
[ ] Features are available before cutoff.
[ ] EIA/macro data joined by release timestamp.
[ ] News aggregated only before cutoff.
[ ] Chronological walk-forward split used.
[ ] Scaler/imputer/feature selector fit only on train.
[ ] Hyperparameters tuned on validation only.
[ ] Test set not used repeatedly for model selection.
[ ] SHAP/top features are economically plausible.
[ ] Regime-wise metrics are stable.
[ ] Falsification tests pass.
```

Mandatory if result is high:

```text
Accuracy > 0.63 or AUC > 0.64: full audit
Accuracy > 0.70: classify as unverified high-claim until independently replicated
```

---

## Part K. First sprint proposal

Goal:

```text
Move from Accuracy 0.5452 / AUC 0.5586
Toward Accuracy >= 0.56 / AUC >= 0.58
```

Tasks:

```text
1. Reproduce current result.
2. Build feature registry.
3. Add OVX, MOVE, Gold, S&P 500, DXY.
4. Implement clean walk-forward evaluation.
5. Run A0-A4 ablations.
6. Produce leakage audit report.
```

Non-goals:

```text
No graph neural network yet.
No complex LLM sentiment yet.
No weekly result mixed with daily benchmark.
No 0.70+ KPI.
```

---

## Part L. Weekly side experiment

Weekly is worth testing because macro/supply/sentiment may be less noisy at lower frequency.

But:

```text
fewer samples
higher overfit risk
larger fold variance
```

Rules:

```text
Keep weekly as separate research track.
Use simpler models.
Do not compare weekly accuracy directly with daily accuracy.
```

---

## Part M. References

- Luo et al. (2019), *Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks*  
  https://www.mdpi.com/1911-8074/12/1/9

- Kulkarni & Haidar (2009), *Forecasting Model for Crude Oil Price Using Artificial Neural Networks and Commodity Futures Prices*  
  https://arxiv.org/abs/0906.4838

- Pan, Haidar & Kulkarni (2009), *Daily prediction of short-term trends of crude oil prices using neural networks exploiting multimarket dynamics*  
  https://link.springer.com/article/10.1007/s11704-009-0025-3

- Cohen (2025), *A Comprehensive Study on Short-Term Oil Price Forecasting Using Econometric and Machine Learning Techniques*  
  https://www.mdpi.com/2504-4990/7/4/127

- Foroutan & Lahmiri (2024), *Deep Learning-Based Spatial-Temporal Graph Neural Networks for Price Movement Classification in Crude Oil and Precious Metal Markets*  
  https://www.sciencedirect.com/science/article/pii/S2666827024000288

- Li, Shang & Wang (2019), *Text-based crude oil price forecasting: A deep learning approach*  
  https://www.sciencedirect.com/science/article/pii/S0169207018301110

- Bai et al. (2022), *Crude oil price forecasting incorporating news text*  
  https://www.sciencedirect.com/science/article/pii/S0169207021001060

- Bu (2014), *Effect of inventory announcements on crude oil price volatility*  
  https://www.sciencedirect.com/science/article/pii/S0140988314001236

- Li et al. (2024), *Do OPEC+ policies help predict the oil price: A novel news-based predictor*  
  https://www.sciencedirect.com/science/article/pii/S2405844024104689

- Li et al. (2025), *Crude oil price fluctuation forecasting incorporating news sentiment based on improved sentiment lexicon*  
  https://link.springer.com/article/10.1007/s44443-025-00289-8

- Mohsin & Jamaani (2023), *Green finance and the socio-politico-economic factors' impact on the future oil prices*  
  https://www.sciencedirect.com/science/article/pii/S0301420723004919

- Conlon et al. (2022), *The Illusion of Oil Return Predictability: The Choice of Data Matters*  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3841507

- FRED WTI daily price  
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

## Final instruction to team

```text
Treat current repo as a reasonable baseline.
Improve features and validation before adding complex models.
Aim for clean 0.58-0.61 accuracy or 0.60-0.64 AUC.
Do not chase 0.70+ without audit.
```
