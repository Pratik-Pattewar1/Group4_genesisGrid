# DeliveryIQ — Three Prioritised Business Recommendations

Stage 5 · generated 2026-10-31 by `src/business_insights.py` · evidence: `reports/stage5_metrics.json`  
**Labels:** [Observed] measured in this project · [Action] recommended step · [Scenario] hypothetical, not a measured result. The late benchmark is **> 30 minutes, project-defined, not an official SLA**.

## Priority 1 — Deploy order-placement late-risk alerts with tiered interventions

**1. Priority:** 1

**2. Business problem:** Late orders are only discovered once they are already late. Operations cannot act early, and customers are not warned.

**3. Numerical evidence:**
- [Observed] The late-risk model (> 30 min project benchmark) ranks orders well on unseen test days: ROC-AUC 0.791, PR-AUC 0.608 against a 28.7% base rate.
- [Observed] At the decision threshold 0.268 (chosen on training data), it catches **81.3% of late orders**, and **46.7% of alerts are correct** (1.63× the base rate).
- [Observed] Per 1,000 orders (test period), 500 are flagged: 234 of them truly late and 267 false alarms. 54 late orders are missed.
- [Observed] Risk ranking is sharp at the top. The riskiest 10% of orders were 69.8% late (2.43× average). The riskiest 20% hold 42.6% of all late orders.

**4. What the evidence indicates:** Order-time information alone is enough to tell high-risk from low-risk orders. But a single alert list would be large (about half of all orders) and over half of it would be false alarms, so the response must be tiered by cost.

**5. Recommended operational action [Action]:**
- Score every order at placement with the saved model (`models/late_classifier_gt30.joblib`).
- **Tier A, highest predicted risk:** higher-cost actions such as kitchen/dispatch priority and a proactive customer ETA update.
- **Tier B, above 0.268 but not Tier A:** low-cost actions only, such as a padded promised time or an automated message.
- **Below threshold:** no action.

**6. How to implement:**
- Weeks 1–2: run the model in **shadow mode**, logging scores without acting.
- Weeks 3–4: set the Tier A cut-off from shadow-mode data (capacity-based, e.g. the top 10–20% of risk), not from this report's test set.
- Then run an A/B pilot by city or by day, with intervention cost logged per alert.

**7. Expected operational benefit:**
- [Observed] 81.3% of late customers could be warned before the delay happens.
- [Scenario, not measured] If interventions prevented 10% / 20% / 30% of the caught late orders, the late rate would fall by about 2.3 / 4.7 / 7.0 percentage points from 28.7%. Calculation: caught late orders as a share of all orders (23.35%) × prevention rate. No prevention rate has been measured.

**8. KPIs:**
- Weekly alert recall and precision against actual > 30-minute outcomes
- Late-order rate (> 30 min) in pilot vs control
- Share of late orders notified before the promised time
- Intervention cost per prevented late order; customer complaints and cancellations per 1,000 orders

**9. Risks, assumptions and limitations:**
- Over half of Tier B alerts are false (267 per 1,000 orders), so only cheap actions belong there.
- Probabilities are not calibrated. Use them for ranking, and calibrate before quoting them as percentages to customers.
- The threshold and benchmark are project-defined (> 30 min, not an official SLA).
- The test period is 8 days of partly synthetic data. Performance must be re-measured in the pilot.

## Priority 2 — Introduce a two-stage ETA: estimate at placement, refresh at rider assignment

**1. Priority:** 2

**2. Business problem:** At order placement the ETA is uncertain: on average it is off by 6.19 minutes (test MAE), and it explains only 31% of the variation in delivery time.

**3. Numerical evidence:**
- [Observed] Placement model: test MAE 6.19 min, R² 0.310 (median baseline MAE 7.47).
- [Observed, comparison model only] Adding rider, vehicle, batching, weather and traffic fields cuts test MAE to **3.67 min without rider rating** (R² 0.762), 40.7% lower. With rating it is 3.50 min.
- [Observed, comparison model only] The late-risk ROC-AUC rises to 0.952 without rating (0.983 with rating).

**4. What the evidence indicates:** Most of the delivery-time variation is explained by information that becomes known only after dispatch. The biggest ETA accuracy gain comes from updating the estimate, not from a better placement model.

**5. Recommended operational action [Action]:**
- Build a **separate** dispatch-time model that refreshes the ETA and late risk once a rider is assigned.
- Keep the placement model for the first ETA.
- Never feed dispatch-time fields into the placement model.

**6. How to implement:**
- First audit **when** each field is recorded (weather, traffic, `multiple_deliveries`, rider rating).
- Keep only fields that are known at assignment. Exclude rider rating until it is shown to predate the order.
- Retrain and validate with the same chronological protocol. The comparison model was **not saved** and is not production-ready.
- Run it in shadow mode, then show the refreshed ETA in the app.

**7. Expected operational benefit:**
- [Observed, comparison only] On the test period, the refreshed ETA error would be about 3.67 vs 6.19 min.
- [Expected, unmeasured] More trustworthy customer ETAs after dispatch and fewer 'where is my order' contacts.

**8. KPIs:**
- ETA MAE and the share of orders within ±5 minutes, at placement and at assignment
- Late-risk recall at assignment
- Customer contacts about order status per 1,000 orders

**9. Risks, assumptions and limitations:**
- Leakage risk: rider rating may include feedback on the same delivery. Batching, weather and traffic may be recorded during the trip. Traffic is almost a function of order hour (97.48% consistency).
- The comparison figures come from one test period on partly synthetic data.
- The refresh only helps after assignment. The first promise to the customer still comes from the placement model.

## Priority 3 — Surge-staff riders in the 19:00–21:59 window and plan promises around predictable peaks

**1. Priority:** 3

**2. Business problem:** Delays concentrate in particular hours, days and weather conditions, but rider capacity and customer promises are not differentiated by them.

**3. Numerical evidence:**
- [Observed] Orders placed in hours 19:00, 20:00, 21:00 have the highest late rates (**49.4%, 50.6%, 50.1%**) against a network rate of 29.8%. They carry 13,820 orders (30.3% of all).
- [Observed] Orders placed 15:00–23:59 are 72.9% of orders but **84.5% of late orders**.
- [Observed] Bad weather adds delay beyond traffic, mostly in heavy traffic. Within Jam traffic, Fog orders are 58.3 pp more often late than Sunny ones; within Low traffic, only 4.1 pp.
- [Observed] Evening (19–23) late rate: 35.5%. Morning (08–10): 2.2%. Timed orders outside 15–23: 14.9%.
- [Observed] Time of day is the strongest model factor: shuffling it raises training ETA error by 1.00 min and lowers late-risk ROC-AUC by 0.160. The model predicts 29.1 min at 20:00 vs 21.8 min at 09:00.
- [Observed] Festival days: 896 orders, mean 45.5 min, 100.0% over 30 minutes.

**4. What the evidence indicates:** Delay risk is highly predictable from the clock and the calendar. That makes it a planning problem as much as a real-time problem.

**5. Recommended operational action [Action]:**
- Add surge rider shifts for 19:00–21:59 first, then 17:00–18:59. Set kitchen and dispatch staffing to the same hours.
- Trigger extra evening capacity when fog or cloudy weather is forecast. This needs a live weather feed known before dispatch.
- Publish time-band-specific promised delivery windows, wider in the evening peak.
- Run a festival-day surge plan: extra riders and wider promised windows announced in advance.

**6. How to implement:**
- Use the ETA model's hourly prediction curve and historical late rates by band to set initial capacity targets.
- Review weekly against actual late rates.
- Start in the cities with the highest evening volume.

**7. Expected operational benefit:**
- [Scenario, not measured] Bringing the evening late rate down to the afternoon level (32.5%) would cut the overall late rate by about 1.5 percentage points. Calculation: evening share 50.2% × gap 3.0 pp.
- [Expected, unmeasured] More honest customer promises in peak hours, and so fewer broken promises.

**8. KPIs:**
- Late rate (> 30 min) and ETA bias by hour band and on festival days
- Riders active per order, by hour band
- Promised-window hit rate

**9. Risks, assumptions and limitations:**
- Association, not proven cause: traffic level in this data follows order hour almost exactly (97.48%), so the hour effect may be congestion, demand or both.
- The 100% late rate on festival days is a deterministic pattern typical of synthetic data. Real festival effects must be measured before staffing to them.
- Capacity costs are not in the dataset, so the cost–benefit must be assessed by operations.
- Weather's recording time is unknown in this data. A weather trigger needs a forecast or live feed available before dispatch.

## Why not a dark-store location recommendation?

- The 22 K-Means zones are exactly the 22 cities. Silhouette of the zones = silhouette of the city labels = 0.948.
- Late rates do not differ significantly between zones (chi-square p = 0.103).
- Delivery points are synthetic offsets around restaurants.
- The evidence therefore cannot support choosing store locations. See the insights report §8 for the data this would need.
