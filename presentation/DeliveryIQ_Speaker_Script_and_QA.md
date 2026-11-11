# DeliveryIQ — Speaker Script, Timing & Q&A

**Manthan Hackathon 2026 · Calibo AI Academy · KLU + PATH3 · ML-01 · Team Genesis Grid · Group 4**
Deck: `presentation/DeliveryIQ_Final_Hackathon_Presentation.pptx` (10 slides, 16:9). The same script is in each
slide's speaker notes.

**Format:** 12 min presentation (including the optional demo), then 3 min Q&A. **Target: 11 min 25 s**, which leaves
a 35-second buffer.

## Timing

| Slide | Title | Speaker | Time |
|---|---|---|---|
| 1 | DeliveryIQ (opening + team) | Manda Lohitha | 0:35 |
| 2 | Late deliveries are expensive — and hard to see coming | Manda Lohitha | 0:50 |
| 3 | 45,593 orders: cleaned, not deleted | K Bonishira Akshaya | 1:10 |
| 4 | From raw data to a prediction — without leakage | K Bonishira Akshaya | 1:15 |
| 5 | From order details to an actionable prediction (optional live demo) | Pratik Jankiram Pattewar | 2:50 |
| 6 | Better than guessing — and built to catch late orders | Yatheesh Chandra Maram | 1:20 |
| 7 | Delays follow the clock, not the map | Yatheesh Chandra Maram | 1:10 |
| 8 | Three actions a delivery team can start with | Sanapalli Vikas | 1:15 |
| 9 | What better data would unlock next | Sanapalli Vikas | 0:35 |
| 10 | An ETA before the delay. A warning before it happens. | Sanapalli Vikas | 0:25 |
| | **Total** | | **11:25** |

**Per speaker:** Lohitha 1:25 · Bonishira 2:25 · Pratik 2:50 · Yatheesh 2:30 · Vikas 2:15.

**Animations:** they play **automatically** (short fades); nobody needs to click to reveal anything.
- Slide 2: flow arrows.
- Slide 4: pipeline steps.
- Slide 5: example tiles.
- Slide 6: right column.
- Slide 8: rows.
- Slide 9: roadmap.

**Transitions:** fade throughout; wipe on slides 4 and 9.

---

## Scripts

### Slide 1 — Manda Lohitha (0:35)
> Good morning. We are Team Genesis Grid, Group 4. I'm Lohitha, and with me are Pratik, Bonishira, Yatheesh and Vikas.
> Our project is DeliveryIQ. The idea is simple: at the moment a customer places an order, we predict how long it will
> take and how likely it is to be late. Then we turn that into decisions an operations team can actually use.

*Transition:* "Let me start with why this matters."

### Slide 2 — Manda Lohitha (0:50)
> In quick commerce, a broken delivery promise can mean a cancellation, a refund or a lost customer. In our dataset,
> almost 3 in every 10 orders took more than 30 minutes. That 30-minute line is our own benchmark, because the data has
> no official SLA. The real problem is timing: a late order is usually discovered only after it is already late. So we
> set four goals: predict the ETA, flag risky orders at checkout, understand where and when delays concentrate, and
> convert that into concrete actions.

*Transition:* "Bonishira will explain the data we worked with."

### Slide 3 — K Bonishira Akshaya (1:10)
> We used the food-delivery dataset recommended in the challenge guide, with restaurants standing in for dark stores:
> 45,593 orders across 22 cities. Before modelling, we audited every column, and the data was messier than it looked.
> Missing values were hidden as the text "NaN", so pandas first reported zero missing values; in fact 4,225 rows had
> gaps. 3,640 restaurants sat at latitude-longitude zero (that's the cross on this chart), and 431 had a flipped minus
> sign (the orange points). We corrected those only when four independent checks agreed. Our rule was simple: flag
> problems, never invent values, and delete nothing. Every cleaning step can be re-run from the raw file and gives an
> identical result.

*Transition:* "With the data investigated, here's how we built the models."

### Slide 4 — K Bonishira Akshaya (1:15)
> Here's our pipeline. The most important decision is in the feature selection. A prediction made at checkout can
> only use information that exists at checkout. Pickup time, rider details, traffic and weather are known later, or
> their recording time isn't documented, so using them in the checkout model would be data leakage. We kept 13
> order-time features: distance, time of day, day of week, order type, city type and festival. Second, we split by
> date: we trained on February and March and tested on the final eight days, 8,470 orders the models never saw during
> tuning. Third, every model had to beat a simple baseline, and all preprocessing was fitted on training data only. We
> built it in Python with pandas and scikit-learn, with a Streamlit interface.

*Transition:* "Pratik will show how DeliveryIQ turns those inputs into predictions."

### Slide 5 — Pratik Jankiram Pattewar (2:50, including the optional demo)
> This is DeliveryIQ's prediction workflow. A user enters only what is known at checkout (the store, delivery
> distance, date and time, order type, city type and festival) and gets back an ETA, a late-risk score, a risk level, a
> suggested action and a short "why this prediction". On the left is our actual app screen from a local run: this
> default order gets 27 minutes, a 32% risk score and an "Elevated" level, so the app suggests only a low-cost action.
> Two verified examples on the right:
> - **A**, a weekday morning order at 09:15, 3 km: 18.1 minutes, 2% risk, Low.
> - **B**, the evening peak at 20:00, 16 km: 32.9 minutes, 61% risk, High. That's an order to prioritise and to warn
>   the customer about early. Distance alone adds about 4.6 minutes here.
>
> The point is to find the orders that deserve attention before they become late. These are model outputs, not delivery
> guarantees: the risk percentage ranks orders; it isn't a calibrated probability.

*Transition:* "Now let's see how these models performed on unseen test data."

### Slide 6 — Yatheesh Chandra Maram (1:20)
> For ETA, we compared three models against simply guessing the average. Gradient boosting reached a typical error of
> about 6.2 minutes, 17.5% better than guessing. Random forest was practically tied, 6.20 against 6.19, so we chose
> using cross-validation on the training data only; it's also lighter to run. For late risk, we optimised for recall,
> because missing a genuinely late order costs more than sending an extra alert. We catch about 81% of late orders.
> Look at this chart: among the 10% of orders the model ranks riskiest, about 70% were actually late, roughly two and a
> half times the normal rate. The honest trade-off is false alarms: 2,259 against 1,978 caught. That's why our
> recommendations use different levels of intervention.

*Transition:* "So what did the data reveal about why orders are late?"

### Slide 7 — Yatheesh Chandra Maram (1:10)
> The strongest pattern is time. Orders placed between 7 and 10 pm are about half late (the orange bars), and the
> afternoon-to-night window holds about 85% of all late orders. Weather matters mainly when traffic is heavy: in a jam,
> fog orders are 58 percentage points more often late than sunny ones, but only 4 points in light traffic. Distance
> alone explains surprisingly little, about 9%. These are associations, not proven causes.
>
> We were also honest about where the data stopped us. Our 22 clusters are exactly the 22 cities; the city labels
> alone score the same 0.948, and no zone was statistically worse, so we don't claim dark-store locations. And each
> "rider ID" has up to 22 different ages, so the IDs aren't real riders. We tested these limits instead of hiding them.

*Transition:* "Vikas will now explain what a company can do with these findings."

### Slide 8 — Sanapalli Vikas (1:15)
> We translated our findings into three management actions.
> 1. **Risk-based alerts at checkout.** The highest-risk orders get priority dispatch and an early customer update.
>    Because false alarms exist, the rest get only a cheap automated message. What's measured is that 81% of late
>    orders can be flagged before they're late.
> 2. **A two-stage ETA:** one estimate at checkout, and a refreshed one once a rider is assigned. A separate
>    post-dispatch comparison experiment, using rider and traffic information, cut the ETA error from 6.19 to 3.67
>    minutes. It is not our checkout ETA model and not production-ready; it's evidence for the idea.
> 3. **Evening surge planning:** extra rider capacity from 7 to 10 pm, wider promises in the peak, and a festival-day
>    plan.
>
> We've marked clearly what is measured and what is a potential benefit; we're not claiming savings we haven't measured.

*Transition:* "Here's what we would build next."

### Slide 9 — Sanapalli Vikas (0:35)
> DeliveryIQ already has a reproducible data pipeline, saved prediction models and a validated app. The next
> improvements depend mostly on better data: real rider identities would unlock genuine rider segments, and real
> customer locations and costs would make dark-store placement meaningful. On the modelling side we'd build the
> two-stage ETA, calibrate the risk scores, and add live traffic and weather once we can confirm when they're recorded.
> Finally, deployment with monitoring, retraining and a pilot to measure the real impact.

*Transition:* "To close, here's what DeliveryIQ brings together."

### Slide 10 — Sanapalli Vikas (0:25)
> DeliveryIQ gives a quick-commerce team two things at checkout: an ETA that beats guessing by 17.5%, and a warning
> for about 8 in 10 late orders, built without data leakage and fully reproducible. Thank you. We are Team Genesis
> Grid, and we're happy to answer your questions.

---

## Optional live demo (slide 5) — only if the app is already running

1. **Start 5 minutes before the slot** with the exact command below. The first load takes about 25 s. **Do not** type
   plain `streamlit run`: on this laptop it starts the global Python, which can't load the models.
   ```powershell
   cd D:\Group4_GenesisGrid
   .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
   ```
2. Quick example → **Weekday morning, short trip** → Predict → expect **18.1 min · 2% · Low**.
3. Quick example → **Evening peak, long trip** → Predict → expect **32.9 min · 61% · High**.
4. If nothing responds within 10 seconds, return to the slide and carry on. Never debug on stage. The slide works
   without the demo.

---

## Q&A preparation (15 likely questions)

| # | Question | Answer |
|---|---|---|
| 1 | Why is R² only 0.31? | We only use what's known at checkout. Most of the variation comes after dispatch: a separate post-dispatch comparison experiment with rider and traffic data reaches R² 0.76 (MAE 6.19 → 3.67 min). It is not our checkout model and not production-ready. That's exactly why we recommend a two-stage ETA. |
| 2 | Why exclude traffic and weather? | Their recording time isn't documented. If they're observed during the delivery, using them at checkout would be leakage. They belong in the second-stage ETA once their timing is verified. |
| 3 | Accuracy is 0.680, below "always on time" (0.713). Why? | Accuracy rewards ignoring late orders. "Always on time" catches 0 of 2,432 late orders; we catch 1,978. The challenge guide itself says not to rely on accuracy; recall and PR-AUC are the right metrics here. |
| 4 | How did you pick the 0.268 threshold? | It's the F1-best point on out-of-fold **training** predictions, chosen before the test set was touched. It's below 0.5 because a missed late order costs more than an extra notification. |
| 5 | Half your alerts are false. Isn't that a problem? | Yes, so we tier the response. Costly actions go only to the riskiest 10% (≈70% late, 2.4× the average); the rest get a cheap automated message. |
| 6 | Why gradient boosting over random forest? | They tied: test MAE 6.19 vs 6.20, CV MAE 6.295 vs 6.299. We picked by training cross-validation only; gradient boosting is lighter to run. We don't claim it's clearly better. |
| 7 | How do you know there's no leakage? | A code guard rejects forbidden columns; the date split has no overlap; preprocessing is fitted on training data only; and a shuffled-label test showed no hidden signal. An independent audit also found none. |
| 8 | Why no dark-store locations? | K-Means peaked at k = 22 (silhouette 0.948), but the clusters are exactly the 22 cities: the city labels alone give the same score. After correction for 22 tests, no zone is significantly worse. Delivery points are synthetic offsets, so we'd need real customer locations, demand and costs. |
| 9 | Why no rider segments? | The 1,320 "rider IDs" each have 4–22 different ages and map to a single restaurant, so they're restaurant slot codes, not people. Segmenting them would describe restaurants. Real rider IDs and trip history are needed. |
| 10 | Is "every festival order is late" real? | It's 100% of 896 orders, a deterministic pattern typical of synthetic data. We treat it as an artifact, not a business finding. |
| 11 | Why a time-based split? | In real use you predict the future from the past. A random split would let future patterns leak into training and overstate accuracy. |
| 12 | How did you handle missing values? | Text "NaN" was converted to real blanks; nothing was guessed in the data. Blanks are filled only inside the model pipeline with training-data statistics. Order time is never rebuilt from pickup time. |
| 13 | How reliable is the app? | It passed 18/18 automated checks in our validation run, and its predictions were identical to the training pipeline on 600 test orders, including incomplete ones. |
| 14 | Is this real quick-commerce data? | No. It's the food-delivery dataset the guide recommends, with restaurants as stand-ins for dark stores, and it has synthetic patterns. Real deployment would need retraining on real data. |
| 15 | What would you do with another month? | Real rider and location data, the two-stage ETA, calibrated probabilities, then deployment with monitoring, retraining and a pilot to measure impact. |

---

## Charts and images used

| Slide | Source | What it proves | Treatment |
|---|---|---|---|
| 3 | `figures/07_geographic_distribution.png` (left panel) | Raw GPS errors: 431 sign-flipped points and 3,640 at (0, 0) | Cropped, not altered |
| 5 | Live app screenshot from the project audit (11 Nov 2026) | Real app output: 27.0 min · 32% · Elevated | Cropped to form + result; labelled as an actual screen |
| 6 | `figures/12_regression_model_comparison.png` | Models vs baseline (MAE) | As is |
| 6 | `figures/24_late_risk_deciles.png` | Risk ranking: riskiest 10% → 70% late | As is |
| 7 | `figures/26_late_rate_by_hour.png` | 19–21 h ≈50% late | As is |
| 1, 2, 4, 9 | Native PowerPoint shapes | Route motif, flows, pipeline, roadmap | Editable |

---

## Number verification (every value on the slides)

| # | Value on slides | Verified | Source | Raw value |
|---|---|---|---|---|
| 1 | 45,593 orders | yes | reports/audit_metrics.json: shape.rows | 45593 |
| 2 | 20 original columns | yes | audit_metrics.json: shape.cols | 20 |
| 3 | 22 cities | yes | audit_metrics.json: rider_id.city_codes_in_id | 22 |
| 4 | 11 Feb – 6 Apr 2022 | yes | audit_metrics.json: time | 2022-02-11 to 2022-04-06 |
| 5 | Delivery time 10–54 min | yes | audit_metrics.json: target | 10–54 |
| 6 | 4,225 rows with hidden missing values | yes | audit_metrics.json: rows_any_missing | 4225 |
| 7 | 3,640 restaurant GPS at (0, 0) | yes | cleaning_metrics.json: flag_counts.dq_geo_valid | 3640 |
| 8 | 431 sign-flipped coordinates, all passed 4 checks | yes | cleaning_metrics.json: r8 evidence counts | 431 |
| 9 | 91 placeholder rider records | yes | cleaning_metrics.json: dq_rider_sentinel | 91 |
| 10 | 1,731 missing order times | yes | missing_timestamp_metrics.json: n_missing | 1731 |
| 11 | 0 duplicate rows | yes | audit_metrics.json: duplicates | 0 |
| 12 | 0 rows deleted | yes | cleaning_metrics.json: rows_in / rows_out | 45593 / 45593 |
| 13 | 29.8% of orders over 30 min | yes | stage5_metrics.json: data.late30_overall_pct | 29.84 |
| 14 | Train 11 Feb–29 Mar; test 30 Mar–6 Apr; 8,470 test orders; 37,123 train | yes | models/model_card.json: split | 37,123 / 8,470 |
| 15 | 13 order-time features | yes | model_card.json: eta_regressor.features | 13 |
| 16 | Mean baseline MAE 7.51 | yes | stage5_metrics.json: challenge.mean_baseline | 7.5115 |
| 17 | Ridge MAE 6.42 | yes | modelling_metrics.json: families.ridge | 6.4186 |
| 18 | Random forest MAE 6.20 | yes | modelling_metrics.json: families.random_forest | 6.2023 |
| 19 | Gradient boosting MAE 6.19, RMSE 7.73, R² 0.31 | yes | modelling_metrics.json / model_card.json | 6.1940 / 7.7335 / 0.3101 |
| 20 | 17.5% lower MAE than the mean baseline | yes | derived: 1 − 6.1940 / 7.5115 | 17.54 |
| 21 | Recall 81%, precision 47%, ROC-AUC 0.79, PR-AUC 0.61, F1 0.59 | yes | model_card.json: late_classifiers.30.test | 0.8133 / 0.4668 / 0.7911 / 0.6084 |
| 22 | Threshold 0.268 chosen on training out-of-fold predictions | yes | model_card.json + modelling_metrics.json: threshold_rule | 0.2681 |
| 23 | 2,432 late; 1,978 caught; 454 missed; 2,259 false alarms | yes | model_card.json: confusion | tn 3779 / fp 2259 / fn 454 / tp 1978 |
| 24 | Accuracy 0.680 vs 0.713 always-on-time (Q&A only) | yes | derived from model_card.json confusion | 0.6797 |
| 25 | Riskiest 10% ≈70% late (2.4×) vs 28.7% | yes | stage5_metrics.json: top_decile (69.8%; re-ranked 69.9%) | 69.8% / 2.43× |
| 26 | 19:00–21:59 about 50% late | yes | stage5_metrics.json: challenge.by_hour | 49.4 / 50.6 / 50.1 |
| 27 | 15:00–23:59 holds 84.5% of late orders | yes | stage5_metrics.json: observed_peak | 84.52 |
| 28 | Fog vs sunny: +58 pp in jam traffic, +4 pp in low | yes | stage5_metrics.json: weather_traffic | 58.3 / 4.1 |
| 29 | Distance alone ≈9% (test R² 0.087) | yes | stage5_metrics.json: distance_only_model | 0.0869 |
| 30 | 22 clusters = 22 cities; 0.948 = city labels; 0 of 22 zones significant | yes | stage5_metrics.json: geography, zones_vs_network | 0.9482 vs 0.9483 |
| 31 | 1,320 rider IDs with 4–22 ages each | yes | stage5_metrics.json: rider | 1320 / 4–22 |
| 32 | Two-stage ETA 6.19 → 3.67 min (separate post-dispatch comparison experiment, no rider rating, not saved, not the checkout model) | yes | stage5_metrics.json: dispatch_comparison | 3.6708 |
| 33 | 18/18 app checks; parity on 600 orders | yes | reports/app_validation.json | 18/18 |
| 34 | Example A: 18.1 min, 2%, Low | yes | app_validation.json: ui_examples (re-run today: identical) | 18.1 / 2% / Low |
| 35 | Example B: 32.9 min, 61%, High | yes | app_validation.json: ui_examples (re-run today: identical) | 32.9 / 61% / High |

**Also re-checked from the saved models:**
- The slide 5 per-factor notes (morning saves ≈2 min; distance adds ≈4.6 min) come from the app's own "why this
  prediction" output: −2.1 and +4.6 min.
- The screenshot values (27.0 min · 32% · Elevated) are the live app result captured in the audit.
- **Nothing comes from the SwiftOpsAI Power BI report.**
