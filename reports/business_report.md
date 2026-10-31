# DeliveryIQ — Business Report

**KLU + PATH3 ML-01 Quick-Commerce Delivery Intelligence** · concise report (PDF §13: 4–6 pages) · generated 2026-10-31 · all numbers from `reports/stage5_metrics.json`.  
The full Business Solution Document is `reports/business_insights_report.md`; the recommendation detail is in `reports/business_recommendations.md`. **[Observed]** = measured; **[Scenario]** = estimate, not measured.

## 1. Executive Summary

Using 45,593 food-delivery orders as a quick-commerce stand-in (restaurants = stores):
- **ETA at order placement:** gradient boosting predicts delivery time with test MAE **6.19 min**, 17.5% better than the average-time baseline (7.51 min).
- **Late-order alerts:** the classifier flags **81.3%** of late orders (> 30 min, project benchmark) with ROC-AUC 0.791.
- **Where delays concentrate:** the evening peak, where hours 19, 20, 21 are about half late.
- **Dark stores:** the geographic clusters are simply the 22 cities, and no zone is significantly worse than the network.
- **Riders:** rider IDs do not identify individual riders, so rider segmentation is not possible.

Three actions follow: late-risk alerts, a two-stage ETA, and evening surge staffing.

## 2. Business Understanding

A quick-commerce operator loses orders, refunds and customers when deliveries run late. Management needs to know what drives delay, whether late orders can be flagged at placement, where new dark stores should go, and when to surge-staff riders. **Late** = delivery time > 30 minutes. This is a project-defined benchmark: the data and the guide give no official SLA. Late rate = late orders ÷ all orders × 100 (network: 29.8%).

## 3. Dataset and Data Quality

| Issue (PS1) | Count | Treatment |
|---|---|---|
| Missing values stored as text `NaN` | 4,225 rows | Converted to blanks; imputed only inside train-fitted pipelines |
| Duplicates | 0 | None to remove |
| GPS (0, 0) restaurant points | 3,640 | Flagged invalid; distance left blank; rows kept |
| Sign-flipped coordinates | 431 | Corrected after 4 geographic checks; originals kept |
| Impossible age/rating (15/1.0, 50/6.0) | 91 | Blanked as placeholders; originals kept |
| Missing order time | 1,731 | Kept; never rebuilt from pickup time (leakage) |
| Inconsistent labels (`Metropolitian`, padding, `conditions` prefix) | all rows | Normalised |
| Delivery-time outliers (1.5×IQR, 52–54 min) | 270 | Kept: inside the 10–54 min range |

No rows were deleted (45,593 kept). Distance is the Haversine distance from restaurant to customer. **Leakage control:** pickup time, the target, IDs and post-dispatch fields (rider, vehicle, batching, weather, traffic) are excluded from order-placement models, and a code guard enforces this.

## 4. EDA Findings (associations, not causes)

1. **Hour of day:** 19:00 49%, 20:00 51%, 21:00 50% late, vs 1.6% at 09:00 and a 29.8% network rate (`figures/26_late_rate_by_hour.png`).
2. **Weather beyond traffic:** in Jam traffic, Fog orders are +58.3 pp more often late than Sunny ones; in Low traffic +4.1 pp (`figures/27_weather_traffic_late_rate.png`).
3. **Distance:** explains only R² 0.087 on its own, with a step-up beyond about 10 km.
4. **Rider rating and age:** on similar routes, orders by riders rated 4.8–5.0 had a late rate 58.9 pp lower than riders below 4.5, and riders aged 35–39 took +6.6 min vs 20–24. Rating may be a *result* of lateness.
5. **Day of week** varies modestly. **Order type has no effect.** Festival days and Semi-Urban orders are 100% late, which is a synthetic-data artifact.

| Order-time band | Orders | % late | Distance band | Orders | % late |
|---|---|---|---|---|---|
| Night 00 (only hour 0 in data) | 430 | 9.8 | 0–5 km | 11,303 | 12.4 |
| Morning 08–10 | 5,756 | 2.2 | 5–10 km | 11,307 | 21.1 |
| Midday 11–14 | 4,429 | 32.0 | 10–15 km | 11,293 | 45.5 |
| Afternoon 15–18 | 10,340 | 32.5 | 15–25 km | 8,050 | 45.1 |
| Evening 19–23 | 22,907 | 35.5 |  | |  |

**Three evidence-backed findings:**
- **Finding 1: delays concentrate in the evening peak.** [Observed] Orders placed 19:00–21:59 are 49.4–50.6% late, against 29.8% for the network. The 15:00–23:59 window carries 84.5% of all late orders. Time of day is also the models' strongest factor. → Recommendation 3.
- **Finding 2: late orders can be flagged at order placement.** [Observed] Using only order-time information, the classifier reaches ROC-AUC 0.791 on unseen days, catches 81.3% of late orders, and its riskiest 10% of orders were 69.8% late. → Recommendation 1.
- **Finding 3: most ETA uncertainty is resolved only after dispatch.** [Observed] The placement ETA explains 31% of the variance (MAE 6.19 min). A comparison model with dispatch-time data, without rider rating, explains 76% (MAE 3.67 min). Distance alone explains only 9%. → Recommendation 2.

## 5. Modelling Approach

- **Split:** chronological on whole dates (train 2022-02-11 → 2022-03-29, test 2022-03-30 → 2022-04-06).
- **Tuning:** 3 forward-chaining CV folds on training data only; preprocessing fitted inside pipelines.
- **Regression:** mean and median baselines, ridge, random forest and gradient boosting on 13 order-placement features.
- **Classification:** prior baseline, logistic regression and gradient boosting. Class imbalance (30.1% late) is handled by selecting on PR-AUC and moving the decision threshold to the F1-best point on out-of-fold training predictions, rather than resampling.
- **Clustering:** K-Means on delivery coordinates (k by elbow and silhouette). A second K-Means on order-level rider attributes is an alternative to rider segments.

## 6. Model Selection and Explainability

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

- **Threshold:** 0.268, chosen on training out-of-fold predictions to maximise F1. A missed late order costs more than an unnecessary heads-up, so recall (0.813) is favoured over precision (0.467). It is marked on the PR and ROC curves in `figures/29_threshold_on_pr_roc.png`.
- **Explainability:** grouped permutation importance on training data (`figures/22_explainability_grouped_importance.png`). Time of day matters most, then distance, festival and city type. Restaurant location, calendar and order type barely matter.

## 7. Three Recommendations

**1. Deploy order-placement late-risk alerts with tiered interventions**
- Evidence: [Observed] The late-risk model (> 30 min project benchmark) ranks orders well on unseen test days: ROC-AUC 0.791, PR-AUC 0.608 against a 28.7% base rate.
- Action: Score every order at placement with the saved model (`models/late_classifier_gt30.joblib`).
- Why: Order-time information alone is enough to tell high-risk from low-risk orders. But a single alert list would be large (about half of all orders) and over half of it would be false alarms, so the response must be tiered by cost.
- Implementation: Weeks 1–2: run the model in **shadow mode**, logging scores without acting.
- Benefit: [Scenario, not measured] If interventions prevented 10% / 20% / 30% of the caught late orders, the late rate would fall by about 2.3 / 4.7 / 7.0 percentage points from 28.7%. Calculation: caught late orders as a share of all orders (23.35%) × prevention rate. No prevention rate has been measured.
- KPI: Weekly alert recall and precision against actual > 30-minute outcomes
- Main risk: Over half of Tier B alerts are false (267 per 1,000 orders), so only cheap actions belong there.

**2. Introduce a two-stage ETA: estimate at placement, refresh at rider assignment**
- Evidence: [Observed] Placement model: test MAE 6.19 min, R² 0.310 (median baseline MAE 7.47).
- Action: Build a **separate** dispatch-time model that refreshes the ETA and late risk once a rider is assigned.
- Why: Most of the delivery-time variation is explained by information that becomes known only after dispatch. The biggest ETA accuracy gain comes from updating the estimate, not from a better placement model.
- Implementation: First audit **when** each field is recorded (weather, traffic, `multiple_deliveries`, rider rating).
- Benefit: [Observed, comparison only] On the test period, the refreshed ETA error would be about 3.67 vs 6.19 min.
- KPI: ETA MAE and the share of orders within ±5 minutes, at placement and at assignment
- Main risk: Leakage risk: rider rating may include feedback on the same delivery. Batching, weather and traffic may be recorded during the trip. Traffic is almost a function of order hour (97.48% consistency).

**3. Surge-staff riders in the 19:00–21:59 window and plan promises around predictable peaks**
- Evidence: [Observed] Orders placed in hours 19:00, 20:00, 21:00 have the highest late rates (**49.4%, 50.6%, 50.1%**) against a network rate of 29.8%. They carry 13,820 orders (30.3% of all).
- Action: Add surge rider shifts for 19:00–21:59 first, then 17:00–18:59. Set kitchen and dispatch staffing to the same hours.
- Why: Delay risk is highly predictable from the clock and the calendar. That makes it a planning problem as much as a real-time problem.
- Implementation: Use the ETA model's hourly prediction curve and historical late rates by band to set initial capacity targets.
- Benefit: [Scenario, not measured] Bringing the evening late rate down to the afternoon level (32.5%) would cut the overall late rate by about 1.5 percentage points. Calculation: evening share 50.2% × gap 3.0 pp.
- KPI: Late rate (> 30 min) and ETA bias by hour band and on festival days
- Main risk: Association, not proven cause: traffic level in this data follows order hour almost exactly (97.48%), so the hour effect may be congestion, demand or both.

**Expected value: measured vs projected**

| Action | Measured in this project [Observed] | Projected estimate [Scenario] |
|---|---|---|
| 1. Late-risk alerts | 81.3% of late orders flagged before they are late | late rate −2.3 to −7.0 pp if 10–30% of caught late orders are prevented |
| 2. Two-stage ETA | ETA error 3.67 vs 6.19 min (comparison model) | fewer order-status contacts (unquantified) |
| 3. Evening surge staffing | peak hours about 50% late vs 29.8% network | late rate −1.5 pp if the evening late rate fell to the afternoon level |

**Dark stores:** the 'high demand and above benchmark' rule flags HYD, RANCHI, JAP, but no zone is statistically different from the network. A new-store decision needs real customer locations, costs and service radii (Solution Document §8).

## 8. Limitations

- Food-delivery data with synthetic patterns (diagonal delivery offsets, 5-minute time blocks, traffic set by hour, segments that are 100% late).
- Test period of 8 days; April appears only in the test period.
- 3,640 orders without valid coordinates; 1,731 without an order time.
- > 30 min is a project benchmark, not an official SLA. Probabilities are not calibrated.
- Clusters are cities, so no dark-store location can be established. Rider IDs are not people, so rider segments cannot be built.
- All patterns are associations. Projected benefits are scenarios, not measurements.

## 9. Technical Reproducibility

```powershell
cd D:\Project1
.venv\Scripts\python.exe -m src.data_cleaning
.venv\Scripts\python.exe -m src.feature_engineering
.venv\Scripts\python.exe -m src.modelling
.venv\Scripts\python.exe -m src.business_insights
```
Seed 42, pinned `requirements.txt`, SHA-256-checked raw data; every stage writes a metrics JSON and runs automated validation checks.
