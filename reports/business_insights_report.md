# DeliveryIQ — Business Insights Report

**AI-Powered Quick Commerce Delivery Intelligence System** · Stage 5 · generated 2026-10-31 by `src/business_insights.py`  
Every number is computed from project artifacts and stored in `reports/stage5_metrics.json`. **Labels:** [Observed] measured · [Action] recommendation · [Scenario] hypothetical, not measured.

## 1. Executive Summary

- **Data:** 45,593 delivery orders across 22 Indian cities, 2022-02-11 → 2022-04-06. All rows were kept; data-quality problems are flagged, not deleted.
- **ETA prediction at order placement:** test MAE **6.19 min** vs 7.47 for a median guess, **17.1% better**. R² 0.310.
- **Late-order risk (> 30 min, project benchmark, not an official SLA):**
  - ROC-AUC **0.791**; catches **81.3%** of late orders; **46.7%** of alerts are correct (1.63× the base rate).
  - The riskiest 10% of orders were 69.8% late.
- **What drives delay (association, not proven cause):** time of day first, then delivery distance, festival days and city type. Restaurant location, calendar and order type add almost nothing.
- **When to surge-staff:** orders placed 19:00–21:59 are about half late, against 29.8% for the network (§4.1). The three evidence-backed findings are in §4.6.
- **Geography:** the 22 clusters are the 22 cities (silhouette 0.948 = city separation). Late rates do not differ significantly between them (p = 0.103). **No dark-store location can be justified from this data.**
- **Rider segmentation is not feasible:** rider IDs do not identify individual riders.
- **After rider assignment** a comparison model reaches MAE 3.67 min (without rider rating). That is strong evidence for refreshing the ETA at dispatch, but the model is not production-ready.
- **Three priorities:**
  1. Deploy order-placement late-risk alerts with tiered interventions
  2. Introduce a two-stage ETA: estimate at placement, refresh at rider assignment
  3. Surge-staff riders in the 19:00–21:59 window and plan promises around predictable peaks

## 2. Business Problem and Objectives

Customers need a reliable delivery promise, and operations need early warning of orders that will run late. The project answers four questions: how long will an order take, will it be late, where is demand, and what should management do.

| Objective | Approach | Status |
|---|---|---|
| Predict delivery ETA | Gradient-boosting regression vs median baseline, ridge, random forest | ✅ MAE 6.19 min |
| Predict late-delivery risk | Gradient-boosting classifier; > 30 / 35 / 40 min benchmarks | ✅ ROC-AUC 0.791 |
| Geographic delivery zones | K-Means with elbow + silhouette | ⚠️ Done; zones = cities, so limited operational value |
| Rider segmentation | — | ❌ Not feasible with the available rider IDs (§9) |
| Three business recommendations | Evidence from Stages 1–5 | ✅ §10 |
| Interactive demo | Streamlit | ⏳ Stage 6 |

## 3. Dataset and Data Quality Summary

| Item | Value | Handling |
|---|---|---|
| Orders | 45,593 (20 raw columns) | All 45,593 retained |
| Exact duplicate rows | 0 | — |
| Rows with ≥ 1 missing value (stored as text `NaN`) | 4,225 | Converted to real blanks; never filled in the data |
| Restaurant coordinates with a flipped minus sign | 431 | Corrected only after 4 geographic checks; originals kept |
| Restaurant location recorded as (0, 0) | 3,640 | Flagged invalid; coordinates blanked, never invented |
| Placeholder rider records (age 15 / rating 1, age 50 / rating 6) | 91 | Age and rating blanked; originals kept |
| Missing order time | 1,731 | Kept as missing; never rebuilt from pickup time |
| Missing festival flag | 228 | Kept; no 'is-missing' feature (it marks an artifact) |

**Food delivery, not quick commerce:** order types are Snack, Meal, Drinks and Buffet from restaurants. The median delivery takes 26 minutes over a median 9.2 km. Findings describe restaurant delivery and transfer to grocery quick commerce only by analogy.

**Synthetic patterns:** delivery points are diagonal offsets from restaurants (98.2% of rows); delivery times cluster in 5-minute blocks; traffic level follows order hour almost exactly (97.48%); several segments are late 100% of the time (§4). Real-world performance will differ.

## 4. Key EDA Findings

Overall: mean delivery 26.3 min, median 26; **29.8% of orders exceed 30 minutes**. These are observed associations across all orders.

| Factor | Group | Orders | Mean ETA (min) | % late > 30 |
|---|---|---|---|---|
| Order time | Morning 08–10 | 5,756 | 19.5 | 2.2 |
| Order time | Midday 11–14 | 4,429 | 26.9 | 32.0 |
| Order time | Afternoon 15–18 | 10,340 | 26.7 | 32.5 |
| Order time | Evening 19–23 | 22,907 | 27.8 | 35.5 |
| Order time | Night 00 (only hour 0 in data) | 430 | 22.2 | 9.8 |
| Order time | (order time missing) | 1,731 | 26.3 | 30.2 |
| Distance | 0–5 km | 11,303 | 22.2 | 12.4 |
| Distance | 5–10 km | 11,307 | 24.4 | 21.1 |
| Distance | 10–15 km | 11,293 | 29.8 | 45.5 |
| Distance | 15–25 km | 8,050 | 29.9 | 45.1 |
| Distance | (invalid location) | 3,640 | 26.1 | 28.8 |
| Festival | No | 44,469 | 26.0 | 28.6 |
| Festival | Yes | 896 | 45.5 | 100.0 |
| Festival | (missing) | 228 | 11.2 | 0.0 |
| City type | Urban | 10,136 | 23.0 | 18.7 |
| City type | Metropolitan | 34,093 | 27.3 | 33.3 |
| City type | Semi-Urban | 164 | 49.7 | 100.0 |
| City type | (missing) | 1,200 | 22.1 | 15.4 |
| Traffic* | Low | 15,477 | 21.3 | 7.9 |
| Traffic* | Medium | 10,947 | 26.7 | 32.5 |
| Traffic* | High | 4,425 | 27.2 | 33.6 |
| Traffic* | Jam | 14,143 | 31.2 | 50.6 |
| Traffic* | (missing) | 601 | 26.5 | 30.9 |
| Batched deliveries* | 0.0 | 14,095 | 22.9 | 18.1 |
| Batched deliveries* | 1.0 | 28,159 | 26.9 | 30.3 |
| Batched deliveries* | 2.0 | 1,985 | 40.5 | 100.0 |
| Batched deliveries* | 3.0 | 361 | 47.8 | 100.0 |
| Batched deliveries* | (missing) | 993 | 22.8 | 18.6 |
| Order type | Snack | 11,533 | 26.3 | 29.8 |
| Order type | Meal | 11,458 | 26.4 | 30.0 |
| Order type | Drinks | 11,322 | 26.2 | 29.8 |
| Order type | Buffet | 11,280 | 26.3 | 29.7 |

\* Known only after dispatch or of uncertain timing, so excluded from the placement models.

**Takeaways:**
- **Time of day is the largest observed contrast:** 2.2% late in the morning vs 35.5% in the evening. The 15:00–23:59 window carries 72.9% of orders and 84.5% of late orders.
- **Distance matters, as a step:** orders over 10 km are about 46% late vs 12% under 5 km.
- **Deterministic patterns (synthetic-data warning):** every festival-day order, every Semi-Urban order and every order batched with 2–3 others is late. Real operations rarely look like this, so treat these as dataset artifacts.
- **Order type has no effect** (all about 30% late).

### 4.1 Late rate by hour of order (PDF visual set)

| Hour | Orders | Mean ETA | % late | vs network (pp) |
|---|---|---|---|---|
| 00 | 430 | 22.2 | 9.8 | -20.1 |
| 08 | 1,818 | 19.6 | 2.8 | -27.1 |
| 09 | 1,947 | 19.5 | 1.6 | -28.3 |
| 10 | 1,991 | 19.5 | 2.2 | -27.7 |
| 11 | 1,962 | 26.4 | 30.0 | +0.1 |
| 12 | 892 | 26.8 | 32.0 | +2.1 |
| 13 | 784 | 27.5 | 35.5 | +5.6 |
| 14 | 791 | 27.5 | 33.8 | +3.9 |
| 15 | 873 | 23.2 | 15.8 | -14.0 |
| 16 | 709 | 22.9 | 14.8 | -15.0 |
| 17 | 4,278 | 27.4 | 36.1 | +6.3 |
| 18 | 4,480 | 27.3 | 35.1 | +5.2 |
| 19 | 4,595 | 30.8 | 49.4 | +19.6 |
| 20 | 4,539 | 31.2 | 50.6 | +20.7 |
| 21 | 4,686 | 31.1 | 50.1 | +20.3 |
| 22 | 4,576 | 23.2 | 14.6 | -15.2 |
| 23 | 4,511 | 22.5 | 12.3 | -17.6 |

Peak-risk hours: **19:00, 20:00, 21:00**. Hours above the network late rate: 11, 12, 13, 14, 17, 18, 19, 20, 21. Chart: `figures/26_late_rate_by_hour.png`.

### 4.2 Day of week

| Day | Orders | Mean ETA | % late |
|---|---|---|---|
| Mon | 6,207 | 26.3 | 29.5 |
| Tue | 6,375 | 25.3 | 25.5 |
| Wed | 7,093 | 27.8 | 36.2 |
| Thu | 6,348 | 25.2 | 25.2 |
| Fri | 7,031 | 26.8 | 32.2 |
| Sat | 6,290 | 26.1 | 29.0 |
| Sun | 6,249 | 26.4 | 30.2 |

Late rates range from 25.2% (Thu) to 36.2% (Wed); chi-square p = 1.4e-57. Caution: each weekday covers only 6–7 dates in the data, so this pattern may not repeat.

### 4.3 Weather vs traffic — does weather add delay beyond traffic? (PDF §8)

| Weather \ Traffic | Low | Medium | High | Jam |
|---|---|---|---|---|
| Sunny | 10.6% | 6.5% | 17.1% | 17.8% |
| Windy | 3.0% | 31.1% | 34.1% | 45.6% |
| Stormy | 2.8% | 31.1% | 38.7% | 43.4% |
| Sandstorms | 2.0% | 31.1% | 38.4% | 44.5% |
| Cloudy | 14.3% | 48.3% | 37.6% | 74.8% |
| Fog | 14.6% | 45.9% | 35.3% | 76.1% |

**Yes, mostly in heavier traffic** [Observed]. Inside the same traffic level, Fog is +4.1 pp vs Sunny in Low traffic, +39.3 pp in Medium, +18.2 pp in High and +58.3 pp in Jam. Sunny orders are rarely late at any traffic level. Because traffic follows the order hour almost exactly in this data, 'traffic' here is partly 'time of day'. Chart: `figures/27_weather_traffic_late_rate.png`.

### 4.4 Distance vs delivery time — how much does distance alone explain? (PDF §8)

A linear model on distance only (fitted on training rows, an additional analysis) explains **R² 0.087** on the test period (MAE 7.22 min; +0.54 min per km). Squared correlation on all valid rows: 0.103. **Distance is a real but secondary driver:** about one-tenth of the variance.

### 4.5 Rider experience and rating vs delay (PDF PS3, §8)

| Rider age band (experience proxy) | Orders | Mean ETA | % late |
|---|---|---|---|
| 20–24 | 10,782 | 23.0 | 17.8 |
| 25–29 | 10,853 | 23.0 | 18.1 |
| 30–34 | 10,901 | 29.6 | 41.4 |
| 35–39 | 11,112 | 29.5 | 41.6 |

| Rider rating band | Orders | Mean ETA | % late |
|---|---|---|---|
| 4.5–4.7 | 17,385 | 24.2 | 16.4 |
| 4.8–5.0 | 18,185 | 24.4 | 18.5 |
| < 4.5 | 8,024 | 35.1 | 84.6 |

**On similar routes** (same distance band and traffic level, 14 strata) [Observed]:
- Orders by riders rated 4.8–5.0, vs riders rated below 4.5: delivery time -8.9 min; late rate -58.9 pp.
- Orders by riders aged 35–39, vs riders aged 20–24: delivery time +6.6 min; late rate +24.2 pp.

Caveats:
- **Rating may be a consequence, not a cause:** a late delivery can lower the rider's rating, which is the leakage risk recorded since Stage 1.
- **There is no tenure field:** age is the only experience proxy, and older riders are slower here, not faster.
- **Rows are orders, not riders** (§9).
- These are dispatch-time fields, so they are excluded from the placement models.

### 4.6 Three evidence-backed findings (PDF §7)

- **Finding 1: delays concentrate in the evening peak.** [Observed] Orders placed 19:00–21:59 are 49.4–50.6% late, against 29.8% for the network. The 15:00–23:59 window carries 84.5% of all late orders. Time of day is also the models' strongest factor. → Recommendation 3.
- **Finding 2: late orders can be flagged at order placement.** [Observed] Using only order-time information, the classifier reaches ROC-AUC 0.791 on unseen days, catches 81.3% of late orders, and its riskiest 10% of orders were 69.8% late. → Recommendation 1.
- **Finding 3: most ETA uncertainty is resolved only after dispatch.** [Observed] The placement ETA explains 31% of the variance (MAE 6.19 min). A comparison model with dispatch-time data, without rider rating, explains 76% (MAE 3.67 min). Distance alone explains only 9%. → Recommendation 2.

Chart: `figures/25_late_rate_ranges.png` compares how much late rates vary by each factor.

## 5. Regression Model Findings (ETA at order placement)

| Model | Test MAE (min) | Test RMSE | Test R² |
|---|---|---|---|
| Median baseline | 7.47 | 9.31 | -0.000 |
| Ridge regression | 6.42 | 8.00 | 0.262 |
| Random forest | 6.20 | 7.76 | 0.305 |
| **Gradient boosting (selected)** | 6.19 | 7.73 | 0.310 |

- **What MAE 6.19 means:** a typical prediction is off by about 6 minutes on an average delivery of 26 minutes. That is usable for a delivery *window*, not a precise minute promise.
- The model explains 31% of the variation. The rest depends on dispatch-time factors (rider, batching, live conditions) that are deliberately unavailable at placement (§7, Recommendation 2).
- **Errors are largest where deliveries are slowest:** Jam traffic MAE 7.31 vs Low 5.12; 15–25 km 7.04 vs 0–5 km 5.25.
- Model selection, the feature choices (`order_month` dropped; missing-time and missing-festival rows kept) and tuning were all made on training data before the test set was scored (Stage 4).

Charts: `figures/12_regression_model_comparison.png`, `figures/13_eta_pred_vs_actual.png`, `figures/14_eta_error_by_segment.png`.

### 5.1 Model selection table (PDF §7)

| Task | Model | Selection metric (training CV) | Test result | Role |
|---|---|---|---|---|
| ETA | Mean baseline (training mean 26.34 min) | — | MAE 7.51 · RMSE 9.31 · R² -0.001 | Naive baseline named in the PDF |
| ETA | Median baseline | CV MAE 7.608 | MAE 7.47 · R² -0.000 | Naive baseline (Stage 4) |
| ETA | Distance-only linear model | — | MAE 7.22 · R² 0.087 | Additional analysis |
| ETA | Ridge regression | CV MAE 6.621 | MAE 6.42 · R² 0.262 | Candidate |
| ETA | Random forest | CV MAE 6.299 | MAE 6.20 · R² 0.305 | Candidate |
| ETA | **Gradient boosting** | **CV MAE 6.295** | **MAE 6.19 · RMSE 7.73 · R² 0.310** | **Final** |
| Late > 30 min | Logistic regression | CV PR-AUC 0.560 | ROC-AUC 0.766 · PR-AUC 0.570 · P 0.402 · R 0.900 · F1 0.556 | Candidate |
| Late > 30 min | **Gradient boosting** | **CV PR-AUC 0.595** | **ROC-AUC 0.791 · PR-AUC 0.608 · P 0.467 · R 0.813 · F1 0.593** | **Final** |
| Late > 30 min | Prior (constant) baseline | CV PR-AUC 0.300 | ROC-AUC 0.500 · recall 0 | Naive baseline |
| Zones | **K-Means k = 22** | Silhouette (max over k = 2–30) | Silhouette 0.948 | Final (zones = cities, §8) |
| Rider-attribute profiles | **K-Means k = 3** | Silhouette (max over k = 2–8) | Silhouette 0.458 | Order-level alternative (§9) |

**Reasons for the final choices:**
- **Gradient boosting for ETA:** lowest training-CV MAE, and 17.5% lower test MAE than the mean baseline. Random forest was within 0.004 min (a tie); gradient boosting is lighter to serve, so the extra complexity is not a cost here.
- **Gradient boosting for late risk:** highest CV PR-AUC, the right metric when late orders are the minority.
- **K-Means zones:** maximum silhouette.
- Every choice was made on training data before the test set was scored.

## 6. Late-Order Classification Findings

The 30-minute line is a **project-defined benchmark** (the dataset has no official quick-commerce SLA).

| Metric (test period) | Value | Meaning |
|---|---|---|
| ROC-AUC | 0.791 | A random late order is ranked above a random on-time order 79% of the time |
| PR-AUC | 0.608 | vs 0.287 for a no-skill model |
| Recall | 0.813 | 81.3% of late orders are flagged in advance |
| Precision | 0.467 | 46.7% of flagged orders are actually late; 53.3% are false alarms |
| F1 | 0.593 | Balance of the two at threshold 0.268 |
| Brier score | 0.159 | Probabilities are **not calibrated** — use them for ranking |

**Why recall matters:** a late order the model misses gets no proactive message and no dispatch attention, and it is the costliest failure. The threshold was therefore set below 0.5, at the F1-best point on training data, to favour catching late orders.

**What the precision means:** at this threshold about half of all orders are flagged, and more than half of those alerts are false. That is acceptable only for cheap interventions. Costly actions should be reserved for the highest-risk orders (Recommendation 1).

**Per 1,000 orders (test period):** 500 flagged → 234 truly late + 267 false alarms; 54 late orders missed.

**Risk deciles (test period; reporting only, not used for any model choice):**

| Risk decile (1 = riskiest) | Orders | % actually late |
|---|---|---|
| 1 | 847 | 69.8 |
| 2 | 847 | 52.4 |
| 3 | 847 | 44.5 |
| 4 | 847 | 34.1 |
| 5 | 847 | 32.7 |
| 6 | 847 | 20.0 |
| 7 | 847 | 15.8 |
| 8 | 847 | 11.8 |
| 9 | 847 | 3.5 |
| 10 | 847 | 2.5 |

Chart: `figures/24_late_risk_deciles.png`.

**Benchmark sensitivity:**

| Benchmark | Late share | Flagged share | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| > 30 min | 28.7% | 50.0% | 0.467 | 0.813 | 0.593 | 0.791 | 0.608 |
| > 35 min | 16.7% | 31.4% | 0.393 | 0.738 | 0.513 | 0.831 | 0.540 |
| > 40 min | 8.7% | 14.9% | 0.336 | 0.577 | 0.424 | 0.852 | 0.472 |

- **Stricter benchmarks rank better but alert worse.** ROC-AUC rises to 0.852 at > 40 min because very slow orders are more distinctive. But they are rarer (8.7%), so precision falls to 0.336 and recall to 0.577.
- **> 30 min is the most operationally useful benchmark:** it has the highest F1 (0.593 vs 0.513 and 0.424).

Charts: `figures/16_classification_roc_pr.png`, `figures/17_classification_threshold.png`, `figures/18_confusion_matrix.png`, `figures/21_late_threshold_sensitivity.png`.

**Class imbalance handling (PDF PS5):**
- Late orders are 30.1% of training data and 28.7% of test data.
- Models were selected on PR-AUC, which is sensitive to the minority class, not accuracy.
- The decision threshold was moved from 0.5 to the F1-best point on out-of-fold training predictions.
- No resampling was needed: threshold moving fixes the recall/precision balance directly and keeps probabilities interpretable.

**Chosen threshold on the curves:** at 0.268 → recall 0.813, precision 0.467, false positive rate 0.374, marked in `figures/29_threshold_on_pr_roc.png`.

## 7. Model Explainability

**Stage 4 already provided:**
- single-feature permutation importance (ETA, test set);
- predicted-vs-actual and residual plots;
- ROC and precision-recall curves;
- model comparison tables.

**Stage 5 adds** two things for the saved primary models, using only the 13 order-placement features, on **training split only (37,123 rows, 2022-02-11 → 2022-03-29)**, without retraining or test data:
- **grouped** permutation importance for **both** models;
- partial-dependence curves.

Grouping permutes collinear features together, so the four hour features are scored as one 'time of day' effect. SHAP was not used: the package isn't installed, and the brief asks for no unnecessary installs.

| Factor | ETA model: training MAE increase when shuffled (min) | Late model: ROC-AUC drop when shuffled |
|---|---|---|
| Time of day | 1.002 ± 0.021 | 0.1598 |
| Delivery distance | 0.642 ± 0.009 | 0.0849 |
| Festival day | 0.390 ± 0.004 | 0.0182 |
| City type | 0.263 ± 0.008 | 0.0169 |
| Restaurant location | 0.042 ± 0.001 | 0.0077 |
| Calendar (day of week / month day / weekend) | 0.038 ± 0.002 | 0.0043 |
| Order type | 0.010 ± 0.000 | 0.0017 |

Baselines on training data: ETA MAE 6.13 min; late-risk ROC-AUC 0.806. Chart: `figures/22_explainability_grouped_importance.png`.

**In plain language (what the model has learned):**
- **Time of day:** the model's average ETA is about 21.8 min for 08–10 orders and 29.1 min at 20:00. The predicted chance of being late rises from 4% at 09:00 to 42% at 20:00.
- **Distance:** predicted ETA rises from 22.7 min at 2 km to 28.5 min at 20 km, mostly as a step beyond about 10 km (24.5 → 28.5 min).
- **Festival day:** 26.1 → 40.2 min.
- **City type:** Urban 24.0, Metropolitan 27.0, Semi-Urban 41.7 min. Semi-Urban has only 164 orders, so it is unstable.
- **Little effect:** restaurant location, day of week or month and order type.

**Association, not causation:** these show what the model uses to predict, not what *causes* delay. For example, the hour effect may reflect traffic, demand or kitchen load; the data cannot separate them, since traffic follows the hour almost exactly. The test-set importance ranking from Stage 4 agrees: hour and distance are top. Chart: `figures/23_eta_partial_dependence.png`.

## 8. Geographic and Dark-Store Analysis

| Check | Result |
|---|---|
| K-Means zones selected (Stage 4) | k = 22, silhouette 0.948 |
| Zones vs cities | Every zone holds exactly one city, and every city one zone: **zones = cities** (yes) |
| Silhouette of the *city labels* themselves | **0.948** (identical), so the high score is just distance between cities |
| Late-rate range across zones | 28.7% – 34.3%; chi-square p = **0.103** (no significant difference) |
| Median ETA range across zones | 25 – 27 min |
| Late-rate range across order-time bands (comparison) | 2.2% – 35.5% |
| Zone sizes | 478 – 3,443 valid orders |
| Orders without a valid location | 3,640 overall; concentrated in VAD 50.2%, DEH 33.2%, BHP 30.8% |
| Delivery points | Synthetic diagonal offsets from 388 restaurant locations; median distance 9.2 km (max 21.0) |

### 8.1 Zone vs network benchmark (PDF PS6, §8)

Network benchmark (valid-location orders): mean ETA 26.31 min, late rate 29.94%. High demand = at least the median zone volume (2070 orders). Significance: chi-square (late rate) and Welch t-test (ETA) for each zone vs the rest, Bonferroni-corrected for 22 zones.

| Zone | City | Orders | Demand share | Mean ETA | vs network | % late | vs network | Late p | High demand & above benchmark |
|---|---|---|---|---|---|---|---|---|---|
| 19 | ALH | 568 | 1.4% | 27.40 | +1.09 | 34.3% | +4.4 pp | 0.024 |  |
| 16 | DEH | 492 | 1.2% | 26.64 | +0.33 | 33.5% | +3.6 pp | 0.088 |  |
| 17 | AURG | 555 | 1.3% | 26.85 | +0.54 | 33.2% | +3.2 pp | 0.105 |  |
| 4 | JAP | 3,443 | 8.2% | 26.69 | +0.38 | 31.9% | +2.0 pp | 0.008 | yes |
| 20 | BHP | 478 | 1.1% | 26.72 | +0.41 | 31.8% | +1.9 pp | 0.398 |  |
| 14 | AGR | 550 | 1.3% | 26.64 | +0.32 | 31.6% | +1.7 pp | 0.407 |  |
| 9 | LUDH | 687 | 1.6% | 26.23 | -0.08 | 31.6% | +1.7 pp | 0.363 |  |
| 18 | VAD | 1,576 | 3.8% | 26.66 | +0.35 | 30.8% | +0.9 pp | 0.442 |  |
| 0 | HYD | 3,181 | 7.6% | 26.43 | +0.12 | 30.5% | +0.6 pp | 0.488 | yes |
| 3 | RANCHI | 2,564 | 6.1% | 26.39 | +0.08 | 30.4% | +0.5 pp | 0.595 | yes |
| 6 | PUNE | 3,132 | 7.5% | 26.34 | +0.03 | 29.6% | -0.3 pp | 0.712 |  |
| 8 | COIMB | 3,170 | 7.6% | 26.29 | -0.02 | 29.6% | -0.4 pp | 0.644 |  |
| 10 | CHEN | 3,145 | 7.5% | 26.36 | +0.05 | 29.4% | -0.5 pp | 0.544 |  |
| 12 | KOL | 700 | 1.7% | 26.36 | +0.05 | 29.4% | -0.5 pp | 0.800 |  |
| 1 | INDO | 3,159 | 7.5% | 26.12 | -0.20 | 29.2% | -0.7 pp | 0.392 |  |
| 2 | BANG | 3,195 | 7.6% | 26.05 | -0.26 | 29.2% | -0.8 pp | 0.336 |  |
| 11 | GOA | 609 | 1.5% | 26.10 | -0.21 | 29.1% | -0.9 pp | 0.668 |  |
| 5 | SUR | 3,187 | 7.6% | 26.08 | -0.23 | 29.0% | -0.9 pp | 0.251 |  |
| 15 | MUM | 3,173 | 7.6% | 26.05 | -0.26 | 29.0% | -0.9 pp | 0.236 |  |
| 7 | KNP | 676 | 1.6% | 26.16 | -0.15 | 29.0% | -0.9 pp | 0.619 |  |
| 13 | MYS | 3,012 | 7.2% | 26.02 | -0.29 | 28.7% | -1.2 pp | 0.135 |  |
| 21 | KOC | 701 | 1.7% | 26.35 | +0.03 | 28.7% | -1.3 pp | 0.487 |  |

**Result:** the PDF's 'high demand and above benchmark' rule flags HYD, RANCHI, JAP. However, **0 zones differ significantly in late rate and 0 in ETA** after correction, so these are **provisional watch-list cities, not dark-store recommendations**. The largest gaps are about 1 minute of ETA, far smaller than the evening-peak effect.

**What the analysis can establish:** demand is spread across 22 well-separated cities, and the restaurant-to-customer distances in the data. Distance itself matters (§4).

**What it cannot establish:**
- operational zones inside a city;
- demand hot-spots;
- **any specific dark-store location.**

Customer positions are generated offsets, not real addresses. Zone late rates are statistically indistinguishable. Several cities lose 30–50% of their orders to invalid coordinates. The clusters should be presented as *city groupings*, not validated delivery zones.

**Data needed for dark-store placement decisions:**
- real customer delivery coordinates with order timestamps, to measure demand density by hour;
- SKU-level demand and the inventory constraints of a dark store;
- candidate site availability, rent and operating costs;
- rider supply and road-network travel times (not straight-line distance);
- a target service radius or promised delivery time.

Charts: `figures/19_kmeans_elbow_silhouette.png`, `figures/20_delivery_zones.png`, `figures/25_late_rate_ranges.png`.

## 9. Rider Segmentation Feasibility

| Evidence (cleaned data) | Value |
|---|---|
| Distinct `delivery_person_id` values | 1,320 |
| IDs with more than one recorded age | **100%** (ages per ID: 4–22, median 17) |
| IDs with more than one vehicle type | 100% |
| Valid restaurant locations per ID | at most 1 (each ID is tied to one restaurant) |
| ID suffixes | DEL01, DEL02, DEL03 only |

**Verdict: rider segmentation is not feasible.**
- The ID pattern `<CITY>RES<nn>DEL<nn>` is a restaurant-plus-slot code: one ID covers many people of different ages.
- Any 'rider segment' built on it would describe restaurants or slots, not riders.
- No rider segments were produced, and no individual rider performance is claimed.

**What is possible instead (labelled accurately):** an *order-level delivery-condition profile*. This is not rider segmentation; it is dispatch-time information and is confounded.

| Vehicle type | Orders | Mean ETA | % late > 30 |
|---|---|---|---|
| bicycle | 68 | 26.4 | 30.9 |
| electric_scooter | 3,814 | 24.5 | 23.4 |
| motorcycle | 26,435 | 27.6 | 33.9 |
| scooter | 15,276 | 24.5 | 24.3 |

Motorcycles look slower, but vehicle condition 0 occurs only on motorcycles and has a 41.7% late rate, so the two cannot be separated.

### 9.1 Alternative: order-level rider-attribute profiles (NOT rider segmentation)

PS7 asks for rider segments. Because rider identity is unreliable, K-Means was run on **orders** described by the assigned rider's age and rating (43,594 orders with both values; standardised). k = 3 was chosen by maximum silhouette (0.458 over k = 2–8). Each profile describes orders, not individual riders.

| Profile (descriptive name) | Orders | Mean age | Mean rating | Mean ETA | % late | Mean batched deliveries | % motorcycle |
|---|---|---|---|---|---|---|---|
| Younger, higher-rated | 19,948 | 24.5 | 4.74 | 22.1 | 13.3 | 0.65 | 57.4 |
| Older, higher-rated | 16,962 | 34.6 | 4.75 | 27.6 | 27.0 | 0.78 | 57.2 |
| Older, lower-rated | 6,684 | 32.0 | 4.05 | 35.3 | 86.4 | 0.94 | 64.8 |

**How to use it:**
- It is a **dispatch-time risk signal**: orders assigned to the lower-rated profile run late far more often.
- It **cannot** support rostering or training decisions for individual riders. The same 'rider ID' covers many people, and a low rating may *result* from late deliveries.
- A real rider segmentation needs the data listed below.

Chart: `figures/28_rider_attribute_profiles.png` (rating, age as the experience proxy, and late rate, as the PDF's visual set asks).

**Data needed for genuine rider segmentation:**
- a stable, unique rider identifier;
- per-rider trip history across restaurants;
- shift and login times;
- assignment, pickup and drop timestamps;
- the rider's own vehicle, and ratings captured before each order.

## 10. Exactly Three Prioritised Recommendations

Full detail (evidence, implementation, KPIs, risks): **`reports/business_recommendations.md`**.

| Priority | Recommendation | Core evidence [Observed] |
|---|---|---|
| 1 | Deploy order-placement late-risk alerts with tiered interventions | Recall 81.3%, ROC-AUC 0.791; riskiest 10% → 69.8% late |
| 2 | Introduce a two-stage ETA: estimate at placement, refresh at rider assignment | Placement MAE 6.19 → 3.67 min with dispatch-time data (comparison model, no rating) |
| 3 | Surge-staff riders in the 19:00–21:59 window and plan promises around predictable peaks | 15:00–23:59 = 72.9% of orders, 84.5% of late orders |

**Why this order:**
- **#1** uses a model that is already trained, saved and validated for order-placement use.
- **#2** has the largest measured accuracy gain, but needs a field-timing audit and a new model first.
- **#3** is a planning change supported by strong, consistent patterns. Their cause is not established.

A dark-store or zone recommendation was considered and **rejected**: the evidence does not support it (§8).

## 11. Expected Business Value and KPIs

| Recommendation | Measured in this project | Hypothetical scenario (not measured) | Primary KPI |
|---|---|---|---|
| 1. Late-risk alerts | 81.3% of late orders flagged before they are late | Late rate −2.3 to −7.0 pp if alerts prevent 10–30% of the late orders they catch | Late rate, pilot vs control; alert precision and recall |
| 2. Two-stage ETA | MAE 3.67 vs 6.19 min (comparison model) | Fewer status-enquiry contacts | ETA MAE at assignment; % within ±5 min |
| 3. Peak capacity planning | Peak window = 84.5% of late orders | Late rate −1.5 pp if the evening late rate matched the afternoon | Late rate by hour band and on festival days |

Scenario calculations are transparent arithmetic on observed rates (see the recommendations file). They are not forecasts and must be replaced by pilot measurements.

## 12. Limitations and Deployment Considerations

| Limitation | Effect on deployment and generalisation |
|---|---|
| Food-delivery data, not genuine quick-commerce grocery delivery | Distances and times are much longer than quick-commerce operations. Re-train on real quick-commerce data before use. |
| Synthetic or artificial patterns (diagonal delivery offsets, 5-minute target blocks, traffic set by hour, segments 100% late) | Metrics may be optimistic and some effects may be artifacts. Validate on real data. |
| Test period covers only 8 days (2022-03-30 → 2022-04-06) | No seasonality or longer-term drift is tested. Monitor performance weekly after launch. |
| April appears only in the test period | The month feature was dropped. Behaviour in unseen months is unverified. |
| 3,640 orders lack valid coordinates (up to 50% in some cities) | Distance is unknown for them (filled with the median at model time), and geographic conclusions under-represent those cities. |
| 1,731 orders lack an order time | Their delivery times look normal (p = 0.78), but their hour is unknown and their errors are higher. |
| Some values corrected by documented rules (431 sign fixes, 91 placeholders blanked) | The corrections are reversible and audited (`*_raw`, `dq_*` columns). |
| The > 30-minute late line is project-defined, not an official SLA | Re-set the threshold to the operator's real promise; sensitivity at 35/40 min is provided. |
| Classification probabilities are not calibrated | Use them for ranking and tiering. Calibrate before showing a probability to customers. |
| Clustering does not establish dark-store locations | Do not use the zones for site selection (§8). |
| Rider segmentation is not reliable with current IDs | No rider-level decisions should be based on this data (§9). |
| Dispatch-time comparison model is unsaved and may leak (rider rating) | Supports Recommendation 2's direction only; it is not a deployable model. |

**Deployment guidance:**
- Use only the saved order-placement models (13 features).
- Run them in shadow mode before acting on them.
- Recalibrate the threshold and probabilities on live data.
- Retrain on a schedule, re-checking the chronological split and the leakage guard.

## 13. Conclusion

With order-time information alone, DeliveryIQ beats a naive ETA by 17.1% and flags 81.3% of late orders before they happen. That supports an alerting workflow now (Recommendation 1) and capacity planning around predictable peaks (Recommendation 3).

The largest accuracy gain needs information available only after dispatch (Recommendation 2). The geographic and rider analyses were taken as far as the data honestly allows: the clusters are cities, and rider IDs are not people.

The next step is Stage 6: an interactive demo of the placement-time ETA and late-risk models.

## Appendix A — Verification of Stage 4 figures

| Figure | Value in brief | Recomputed from saved models / metrics | Match |
|---|---|---|---|
| eta_MAE | 6.19 | 6.1940 | ✅ |
| eta_RMSE | 7.73 | 7.7335 | ✅ |
| eta_R2 | 0.31 | 0.3101 | ✅ |
| baseline_MAE | 7.47 | 7.4737 | ✅ |
| improvement_pct | 17.1 | 17.1227 | ✅ |
| ROC_AUC | 0.791 | 0.7911 | ✅ |
| PR_AUC | 0.608 | 0.6084 | ✅ |
| precision | 0.467 | 0.4668 | ✅ |
| recall | 0.813 | 0.8133 | ✅ |
| F1 | 0.593 | 0.5932 | ✅ |
| threshold | 0.268 | 0.2681 | ✅ |
| dispatch_MAE | 3.5 | 3.4982 | ✅ |
| dispatch_ROC_AUC | 0.98 | 0.9825 | ✅ |
| k | 22 | 22.0000 | ✅ |
| silhouette | 0.948 | 0.9482 | ✅ |

## Appendix B — Stage 5 validation checks

| Check | Result | Detail |
|---|---|---|
| input datasets match the recorded SHA-256 hashes | ✅ pass |  |
| saved models use only primary order-placement features (strict leakage guard) | ✅ pass | 13 features |
| every Stage 4 figure in the brief matches recomputation from saved models / metrics (to stated precision) | ✅ pass | mismatches: none |
| saved models reproduce stored Stage 4 test metrics exactly | ✅ pass |  |
| K-Means zones recomputed from saved model reproduce Stage 4 silhouette (±0.005) | ✅ pass | 0.9483 vs 0.9482 |
| rider-ID evidence recomputed from cleaned data (every ID has >1 age) | ✅ pass |  |
| raw, cleaned and engineered datasets unchanged after Stage 5 | ✅ pass |  |
| explainability computed on training rows only | ✅ pass | 37,123 rows |
| exactly three prioritised recommendations | ✅ pass | 3 |
| every figure referenced in the Stage 5 documents exists | ✅ pass | all present |
| challenge guide PDF found and used for the compliance check | ✅ pass | ['Hackathon_Challenge_Guide.pdf'] |
| Business Report within the guide's 4–6 pages (estimated 1,800–3,300 words) | ✅ pass | 2,123 words |
| Business Solution Document at least 5 pages (estimated ≥ 2,500 words) | ✅ pass | 6,084 words |
| Stage 5 documents are complete (no stub markers; every document > 3,000 characters) | ✅ pass | stub: none; short files: none |
| rider profiles labelled as order-level, not rider segmentation | ✅ pass |  |
| 30-minute benchmark never described as an official SLA | ✅ pass |  |

## Appendix C — Charts added in Stage 5

- `figures/22_explainability_grouped_importance.png`
- `figures/23_eta_partial_dependence.png`
- `figures/24_late_risk_deciles.png`
- `figures/25_late_rate_ranges.png`
- `figures/26_late_rate_by_hour.png`
- `figures/27_weather_traffic_late_rate.png`
- `figures/28_rider_attribute_profiles.png`
- `figures/29_threshold_on_pr_roc.png`

Reproduce: `.venv\Scripts\python.exe -m src.business_insights`
