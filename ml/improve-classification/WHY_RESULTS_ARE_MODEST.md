# Why The Current Oil Direction Result Is Modest

Generated: 2026-05-14

This note explains why the current unified pipeline reaches only about **54-55% accuracy** on the strict final test set, while some papers report **70-85%+ directional accuracy**.

The short answer is:

> The current result is modest because the experiment is a strict, leakage-conscious, full-coverage daily next-day UP/DOWN benchmark using the features currently available in this repo. Many high paper numbers are produced under different targets, richer feature sets, different horizons, graph/text architectures, smoothing/thresholding, or validation designs that are not directly comparable.

This does **not** mean the current model is good enough. It means the current result is closer to a clean engineering baseline than to a paper-level system.

---

## 1. Current Result

Current strict final test result from the unified pipeline:

| Item | Value |
|---|---:|
| Best strict test model | `ENS_FINAL3` |
| Accuracy | `0.5476` |
| F1_macro | `0.5406` |
| AUC | `0.5351` |
| Final test setup | `target date >= 2023-01-01` |
| Coverage | `1.0` |
| Threshold | `fixed_0.5` |
| Final comparable test configs | `33` |

The setup is:

```text
Target: oil_return_fwd1 > 0
Train for validation: target date < 2022-01-01
Validation: 2022-01-01 <= target date < 2023-01-01
Final refit: target date < 2023-01-01
Final test: target date >= 2023-01-01
Main threshold: fixed_0.5
Main comparison: full coverage only
```

So the headline number is not the best row after threshold tricks or coverage filtering. It is the strict full-coverage result.

---

## 2. Why This Looks Low Compared With Papers

The paper numbers are useful for ideas, but not all of them are benchmark numbers for this exact task.

| Paper/result type | Reported result | Why it is not directly comparable |
|---|---:|---|
| Pan, Haidar & Kulkarni 2009 | About `79.95%` t+1 hit rate | Daily short-term trend paper, but uses its own preprocessing/modeling setup and multimarket dynamics. Needs replication before using as KPI. |
| Foroutan & Lahmiri 2024 | WTI accuracy around `85%` with MTGNN-style models | Uses spatial-temporal graph neural networks and richer multivariate/technical-indicator feature design. This is not the current repo feature set or architecture. |
| Cohen 2025 style forecasting reports | Often reports RMSE/R2 and directional checks | Not always a pure fixed-threshold classifier benchmark. Regression-to-direction can produce a different comparison. |
| Bai et al. 2022 | Forecasting error improves with news text | Main target is price forecasting with text indicators, not strict daily UP/DOWN classification. |
| Li et al. 2025 sentiment/GRU | High directional accuracy, but weekly | Weekly horizon is materially easier/different than raw daily t+1 direction. |

The correct reading is:

```text
High paper numbers are feature/model ideas.
They are not automatically the target KPI for this repo.
```

The local benchmark brief says a more realistic clean daily target is:

| Accuracy band | Interpretation |
|---:|---|
| `0.50-0.53` | very weak |
| `0.53-0.56` | acceptable baseline |
| `0.56-0.60` | good |
| `0.60-0.63` | very good |
| `>0.63` | audit carefully |
| `0.70+` | not a default KPI for this strict setup |

By that standard, the current `0.5476` is **not strong**, but it is also **not surprising**.

---

## 3. Root Cause 1: The Target Is Extremely Noisy

The task is:

```text
Given information known at date t, predict whether crude oil return from t to t+1 is positive.
```

Daily one-day oil direction has very low signal-to-noise. Local analysis in `docs/WHY_ML_FAILS.md` shows the forward oil return target has:

| Diagnostic | Meaning |
|---|---|
| Mean near zero | There is no large unconditional drift to exploit. |
| High volatility | Noise dominates the one-day signal. |
| Many near-zero days | Small moves are almost label noise for a binary classifier. |
| Weak autocorrelation | Yesterday's direction gives little information about tomorrow. |
| Very weak feature-target correlations | Current features do not separate UP and DOWN days strongly. |

This matters because ML does not create information. If the feature set contains only weak predictive content, stronger learners mostly create more ways to overfit.

The current model grid confirms this:

| Family | Best reading |
|---|---|
| Baselines | Some weak signal, but no dominant simple classifier. |
| Feature selection | Did not unlock a large improvement. |
| Weight decay | Helped in places, but not enough to change the problem. |
| Ensembles | Best hard classifier, but only modestly better. |
| Deep learning | Did not beat tabular ML ensemble. |

That pattern usually means the bottleneck is **information/features**, not just model class.

---

## 4. Root Cause 2: The Current Feature Set Is Not The Feature Set Used By High-Accuracy Papers

The current strict pipeline is running on:

```text
data/processed/dataset_final_noleak_step5c_scaler.csv
```

with a compact set of leakage-audited features. This is very different from the feature stack used by high-performing papers.

| Feature family | Current status | Why it matters |
|---|---|---|
| Lead energy markets | Limited | Pan et al. emphasize multimarket dynamics such as related energy markets. |
| Technical indicators | Restricted by leakage controls | Many high results use technical indicators; in this repo unshifted technicals previously created fake high accuracy. |
| Futures curve / term structure | Not a central feature block yet | Spreads, backwardation/contango, and curve structure can reveal inventory/supply-demand pressure. |
| EIA/API inventory surprises | Not fully built as timestamp-safe surprise features | Raw weekly inventory is weaker than release-time surprise versus consensus. |
| OPEC/supply events | Not fully event-engineered | OPEC meetings, production announcements, and policy changes can matter, but need exact availability dates. |
| News/sentiment | Basic or incomplete | Text papers improve forecasts by constructing domain-specific news/topic/sentiment signals. |
| Graph relationships | Not implemented | Foroutan & Lahmiri's strong results come from spatial-temporal graph models, not plain tabular classifiers. |

So we should not expect a plain tabular/GRU sweep over the current features to reproduce an 80% graph/text/technical-indicator paper.

---

## 5. Root Cause 3: Leakage Controls Remove Fake Performance

This repo already has a clear warning sign: earlier technical indicators could produce very high accuracy when they were not shifted correctly.

From `docs/DATA_LEAKAGE.md`:

| Version | Accuracy | Interpretation |
|---|---:|---|
| Baseline clean-ish features | about `53%` | Plausible |
| Unshifted technical indicators | about `89.2%` | Leakage-driven fake result |
| Shifted technical indicators | expected around `53-55%` | More realistic |

This is the central lesson:

> If a daily oil direction model jumps from 55% to 80-90%, the first explanation to audit is leakage, not genius.

The current pipeline intentionally avoids the most obvious leakage traps:

- chronological split, not random split;
- validation before test;
- fixed final test threshold for the primary leaderboard;
- full-coverage comparison;
- no threshold-label accuracy mixed with raw UP/DOWN accuracy;
- no confidence-filtered selective rows used as the headline result.

That strictness lowers the headline number, but makes it more believable.

---

## 6. Root Cause 4: The 2023+ Test Period Is A Different Regime

The final test period is:

```text
target date >= 2023-01-01
```

This is a difficult period to generalize into. The local docs note large train/test distribution shifts in macro and market variables:

| Shift | Why it hurts |
|---|---|
| zero-rate/COVID/oil-war training history versus high-rate post-inflation test period | relationships learned in one regime may not transfer |
| volatility changed | models trained in volatile regimes can overreact in calmer periods |
| macro variables shifted | Fed rate, CPI, real rates, inventory/production regimes changed |
| geopolitical/news environment changed | historical text/supply relationships may not remain stable |

In time series, a chronological test is harder than a random split because the model must survive future regime change. That is the point of the test.

---

## 7. Root Cause 5: Deep Learning Was Tried, But The Setup Does Not Favor It Yet

The pipeline did include DL:

```text
MLP sequence models: lookback 5, 10, 20, 40
GRU sequence models: lookback 5, 10, 20, 40
```

Best DL strict test row:

| Model | Accuracy | F1_macro | AUC |
|---|---:|---:|---:|
| `DL_GRU_L40` | `0.5167` | `0.5138` | `0.5362` |

This is not surprising. DL often needs:

- much more data;
- richer sequential features;
- strong cross-market structure;
- text/event features;
- careful architecture matched to the data structure.

The Foroutan & Lahmiri result is not just "use deep learning"; it is "use spatial-temporal graph neural networks with attention and graph-structured multivariate relationships." The current DL branch is a fair first pass, but not a replication of that architecture.

---

## 8. Root Cause 6: We Are Comparing A Clean Baseline To Paper-Level Systems

The current pipeline is best understood as:

```text
clean unified benchmark
+ baseline ML
+ feature selection
+ recency weighting
+ ensembles
+ first-pass DL
```

It is not yet:

```text
paper-level feature engineering
+ timestamped news/event system
+ EIA/API surprise system
+ OPEC policy index
+ futures curve features
+ graph neural architecture
+ walk-forward model selection protocol
+ external replication of Pan/Foroutan-style setups
```

Therefore, the honest conclusion is:

> We did not fail to reproduce paper-level results with the same ingredients. We have not yet built the same ingredients.

---

## 9. What The Current Result Actually Means

The result means:

| Observation | Interpretation |
|---|---|
| Accuracy is above 50% | There is weak signal. |
| Accuracy is below 56% | The signal is not strong enough yet. |
| Ensemble is best | Combining weak learners is more stable than trusting one model. |
| AUC and F1 disagree | Probability ranking and hard UP/DOWN decisions are not aligned. |
| DL does not win | Current sequential representation is not rich enough. |
| Feature selection does not win | The issue is not simply too many noisy features. |
| High paper numbers are much larger | Need richer features/architecture, or audit comparability carefully. |

My direct reading:

> The current model found a small, plausible edge. It did not find a strong predictive law. The next improvement will probably come more from better timestamped features than from adding another classifier.

---

## 10. What Would Be Needed To Improve Toward Paper-Level Results

The next research plan should be feature-first:

| Priority | Work item | Why |
|---:|---|---|
| 1 | Build a feature registry with availability timestamps | Prevent leakage while enabling richer features. |
| 2 | Add futures curve / spread features | Captures supply-demand balance better than flat price alone. |
| 3 | Add EIA/API inventory surprise features | Surprise matters more than raw inventory level. |
| 4 | Add OPEC/supply event features | Supply policy is central to crude oil. |
| 5 | Add news/sentiment features | Text papers show news features can help, but must be timestamp-safe. |
| 6 | Add lead-market features | Pan-style multimarket dynamics require related energy/commodity markets. |
| 7 | Run ablation blocks | Prove which feature family adds signal. |
| 8 | Add walk-forward evaluation | Reduces dependence on one 2023+ test period. |
| 9 | Replicate one high-accuracy paper setup separately | Avoid mixing paper replication with the clean production benchmark. |

Suggested ablation order:

```text
market only
market + technicals, shifted safely
market + futures curve
market + EIA/API inventory surprise
market + OPEC/supply events
market + news/sentiment
all features
all features + graph/attention model
```

The goal is not just to increase accuracy. The goal is to know **why** accuracy increases.

---

## 11. Blog-Style Explanation

### Why does my oil direction model only get 55% when papers report 80%?

Because those two numbers often describe different problems.

A clean daily next-day crude oil direction benchmark asks a very harsh question: using only information known today, can we predict whether tomorrow's close-to-close return is positive? Most days are noisy. Many moves are tiny. The market reacts to inventory, macro data, geopolitics, risk sentiment, OPEC signals, and random shocks. A simple UP/DOWN label compresses all of that into one bit.

In this repo, the model is evaluated chronologically. It trains on the past, validates on 2022, and tests on 2023 onward. The test set is not used for training. The final leaderboard uses full coverage and a fixed 0.5 threshold. That is intentionally strict.

Some papers report much higher numbers, but many of them use richer feature systems: lead markets, technical indicators, graph neural networks, news text, OPEC policy signals, or weekly horizons. Some report price-forecasting errors rather than direct classification accuracy. Some use target construction or preprocessing that must be replicated carefully before the number can become a KPI.

The current result, around 54.8% accuracy, is not impressive. But it is believable. It says there is a weak edge, not a solved forecasting problem. The ensemble helps because weak learners disagree in useful ways, but deep learning does not rescue the problem because the current sequence input is not rich enough.

The next step is not to train 200 more classifiers. The next step is to add better information with timestamp discipline: futures curve, inventory surprises, OPEC events, news sentiment, and lead-market structure. Then run ablations to prove which feature family actually moves the needle.

The uncomfortable answer is also the useful answer: the model is not doing 80% because the current benchmark is cleaner and the current feature set does not contain enough predictive information yet.

---

## 12. Slide Outline

### Slide 1: The Question

Why is the current model around 55% when some papers report 80%+?

### Slide 2: Current Result

| Metric | Value |
|---|---:|
| Best model | `ENS_FINAL3` |
| Accuracy | `0.5476` |
| F1_macro | `0.5406` |
| AUC | `0.5351` |

Strict full-coverage test, fixed threshold, chronological split.

### Slide 3: Apples-To-Apples Problem

Paper numbers are not all the same task:

- daily direction;
- trend classification;
- regression-to-direction;
- weekly direction;
- price forecasting;
- graph/text feature systems.

### Slide 4: Current Setup Is Strict

- Train before 2022.
- Validate on 2022.
- Test on 2023+.
- Full coverage.
- Fixed `0.5` threshold.
- Test is not used for training.

### Slide 5: Daily Direction Is Noisy

- One-day oil returns have low signal-to-noise.
- Many days are small moves.
- Feature-target correlations are weak.
- Regimes change.

### Slide 6: Feature Gap

Current pipeline is missing or underusing:

- futures curve;
- lead energy markets;
- EIA/API surprises;
- OPEC events;
- domain sentiment;
- graph relationships.

### Slide 7: Leakage Lesson

Unshifted technical indicators can produce fake high accuracy.

Clean shifted features fall back near 53-55%.

High numbers require audit before trust.

### Slide 8: Why DL Did Not Win

Current DL has:

- limited rows;
- limited feature richness;
- no graph structure;
- no text/event embeddings;
- no paper-level architecture replication.

### Slide 9: Real Interpretation

The model found:

- weak but plausible signal;
- ensemble stability;
- no strong predictive law yet.

### Slide 10: Next Plan

Build feature-first research pipeline:

1. feature registry;
2. timestamp-safe joins;
3. curve/supply/news/OPEC features;
4. ablations;
5. walk-forward evaluation;
6. separate paper replication track.

---

## Sources

Local project sources:

- `ml/improve-classification/REPORT.md`
- `ml/improve-classification/INTERPRETATION.md`
- `docs/improve/oil_direction_research_benchmark_brief.md`
- `docs/improve/oil_direction_leakage_audit_checklist.md`
- `docs/WHY_ML_FAILS.md`
- `docs/DATA_LEAKAGE.md`

External research sources:

- Pan, Haidar & Kulkarni (2009), *Daily prediction of short-term trends of crude oil prices using neural networks exploiting multimarket dynamics*, DOI: `10.1007/s11704-009-0025-3`, https://journal.hep.com.cn/fcs/EN/1159650239743320307
- Foroutan & Lahmiri (2024), *Deep learning-based spatial-temporal graph neural networks for price movement classification in crude oil and precious metal markets*, DOI: `10.1016/j.mlwa.2024.100552`, https://www.sciencedirect.com/science/article/pii/S2666827024000288
- Bai, Li, Yu & Jia (2022), *Crude oil price forecasting incorporating news text*, DOI: `10.1016/j.ijforecast.2021.06.006`, https://arxiv.org/abs/2002.02010
- White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5), 1097-1126, https://econpapers.repec.org/RePEc%3Aecm%3Aemetrp%3Av%3A68%3Ay%3A2000%3Ai%3A5%3Ap%3A1097-1126
