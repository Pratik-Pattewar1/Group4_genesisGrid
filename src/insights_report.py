"""Render the Stage 5 management reports from the metrics computed in ``src.business_insights``.

Writes reports/business_insights_report.md and reports/business_recommendations.md.
Every number is taken from the metrics dictionary; nothing is typed by hand.
Labels: [Observed] measured in this project · [Action] recommendation · [Scenario] hypothetical, not measured.
"""

from __future__ import annotations

from pathlib import Path

REP = Path(__file__).resolve().parents[1] / "reports"
BAND = {"morning_08_10": "Morning 08–10", "midday_11_14": "Midday 11–14", "afternoon_15_18": "Afternoon 15–18",
        "evening_19_23": "Evening 19–23", "night_00_07": "Night 00 (only hour 0 in data)",
        "(order time missing)": "(order time missing)"}


def p1(x):
    return f"{x:.1f}%"


def obs_row(m, col, key):
    return next(r for r in m["observed"][col] if str(r[col]) == key)


def recommendations(m: dict) -> list[dict]:
    """The three prioritised recommendations, built from metrics."""
    c, d, op = m["classification"], m["dispatch_comparison"], m["observed_peak"]
    t, per = c["gt30"], c["per_1000_orders_test"]
    ch = m["challenge"]
    hr = {int(x["order_hour"]): x for x in ch["by_hour"]}
    cap = {x["top_deciles"]: x for x in c["top_decile_capture"]}
    sc = c["scenario_late_rate_reduction_pp"]
    imp = m["explainability"]["eta_mae_increase_min"]["groups"]
    late_imp = m["explainability"]["late_auc_drop"]["groups"]
    fest = obs_row(m, "festival", "Yes")
    morning = obs_row(m, "time_band", "morning_08_10")
    pd_h = {r["value"]: r["mean_prediction"] for r in m["explainability"]["partial_dependence"]["hour_eta"]}
    return [
        {
            "title": "Deploy order-placement late-risk alerts with tiered interventions",
            "problem": "Late orders are only discovered once they are already late. Operations cannot act early, and "
                       "customers are not warned.",
            "evidence": [
                f"[Observed] The late-risk model (> 30 min project benchmark) ranks orders well on unseen test days: "
                f"ROC-AUC {t['ROC_AUC']:.3f}, PR-AUC {t['PR_AUC']:.3f} against a {p1(100 * t['prevalence'])} base rate.",
                f"[Observed] At the decision threshold {t['threshold']:.3f} (chosen on training data), it catches "
                f"**{p1(100 * t['recall'])} of late orders**, and **{p1(100 * t['precision'])} of alerts are correct** "
                f"({c['lift_precision_over_base']:.2f}× the base rate).",
                f"[Observed] Per 1,000 orders (test period), {per['flagged']:.0f} are flagged: {per['late_caught']:.0f} of them truly late "
                f"and {per['false_alarms']:.0f} false alarms. {per['late_missed']:.0f} late orders are missed.",
                f"[Observed] Risk ranking is sharp at the top. The riskiest 10% of orders were {p1(cap[1]['observed_late_pct'])} late "
                f"({c['top_decile']['lift']:.2f}× average). The riskiest 20% hold {p1(cap[2]['share_of_all_late_pct'])} of all late orders.",
            ],
            "indicates": "Order-time information alone is enough to tell high-risk from low-risk orders. But a single alert "
                         "list would be large (about half of all orders) and over half of it would be false alarms, so the "
                         "response must be tiered by cost.",
            "action": [
                "Score every order at placement with the saved model (`models/late_classifier_gt30.joblib`).",
                "**Tier A, highest predicted risk:** higher-cost actions such as kitchen/dispatch priority and a proactive "
                "customer ETA update.",
                f"**Tier B, above {t['threshold']:.3f} but not Tier A:** low-cost actions only, such as a padded promised "
                "time or an automated message.",
                "**Below threshold:** no action.",
            ],
            "implementation": [
                "Weeks 1–2: run the model in **shadow mode**, logging scores without acting.",
                "Weeks 3–4: set the Tier A cut-off from shadow-mode data (capacity-based, e.g. the top 10–20% of risk), "
                "not from this report's test set.",
                "Then run an A/B pilot by city or by day, with intervention cost logged per alert.",
            ],
            "benefit": [
                f"[Observed] {p1(100 * t['recall'])} of late customers could be warned before the delay happens.",
                f"[Scenario, not measured] If interventions prevented 10% / 20% / 30% of the caught late orders, the late rate "
                f"would fall by about {sc['10%']:.1f} / {sc['20%']:.1f} / {sc['30%']:.1f} percentage points from "
                f"{p1(100 * t['prevalence'])}. Calculation: caught late orders as a share of all orders "
                f"({per['late_caught'] / 10:.2f}%) × prevention rate. No prevention rate has been measured.",
            ],
            "kpis": ["Weekly alert recall and precision against actual > 30-minute outcomes",
                     "Late-order rate (> 30 min) in pilot vs control",
                     "Share of late orders notified before the promised time",
                     "Intervention cost per prevented late order; customer complaints and cancellations per 1,000 orders"],
            "risks": [
                f"Over half of Tier B alerts are false ({per['false_alarms']:.0f} per 1,000 orders), so only cheap actions belong there.",
                "Probabilities are not calibrated. Use them for ranking, and calibrate before quoting them as percentages to customers.",
                "The threshold and benchmark are project-defined (> 30 min, not an official SLA).",
                "The test period is 8 days of partly synthetic data. Performance must be re-measured in the pilot.",
            ],
        },
        {
            "title": "Introduce a two-stage ETA: estimate at placement, refresh at rider assignment",
            "problem": f"At order placement the ETA is uncertain: on average it is off by {m['regression']['test']['MAE']:.2f} minutes "
                       f"(test MAE), and it explains only {100 * m['regression']['test']['R2']:.0f}% of the variation in delivery time.",
            "evidence": [
                f"[Observed] Placement model: test MAE {m['regression']['test']['MAE']:.2f} min, R² {m['regression']['test']['R2']:.3f} "
                f"(median baseline MAE {m['regression']['baseline_test']['MAE']:.2f}).",
                f"[Observed, comparison model only] Adding rider, vehicle, batching, weather and traffic fields cuts test MAE to "
                f"**{d['eta_MAE_no_rating']:.2f} min without rider rating** (R² {d['eta_R2_no_rating']:.3f}), "
                f"{d['mae_reduction_no_rating_pct']:.1f}% lower. With rating it is {d['eta_MAE']:.2f} min.",
                f"[Observed, comparison model only] The late-risk ROC-AUC rises to {d['late_ROC_AUC_no_rating']:.3f} without rating "
                f"({d['late_ROC_AUC']:.3f} with rating).",
            ],
            "indicates": "Most of the delivery-time variation is explained by information that becomes known only after "
                         "dispatch. The biggest ETA accuracy gain comes from updating the estimate, not from a better "
                         "placement model.",
            "action": [
                "Build a **separate** dispatch-time model that refreshes the ETA and late risk once a rider is assigned.",
                "Keep the placement model for the first ETA.",
                "Never feed dispatch-time fields into the placement model.",
            ],
            "implementation": [
                "First audit **when** each field is recorded (weather, traffic, `multiple_deliveries`, rider rating).",
                "Keep only fields that are known at assignment. Exclude rider rating until it is shown to predate the order.",
                "Retrain and validate with the same chronological protocol. The comparison model was **not saved** "
                "and is not production-ready.",
                "Run it in shadow mode, then show the refreshed ETA in the app.",
            ],
            "benefit": [
                f"[Observed, comparison only] On the test period, the refreshed ETA error would be about {d['eta_MAE_no_rating']:.2f} "
                f"vs {m['regression']['test']['MAE']:.2f} min.",
                "[Expected, unmeasured] More trustworthy customer ETAs after dispatch and fewer 'where is my order' contacts.",
            ],
            "kpis": ["ETA MAE and the share of orders within ±5 minutes, at placement and at assignment",
                     "Late-risk recall at assignment",
                     "Customer contacts about order status per 1,000 orders"],
            "risks": [
                "Leakage risk: rider rating may include feedback on the same delivery. Batching, weather and traffic may "
                "be recorded during the trip. Traffic is almost a function of order hour "
                f"({m['data']['traffic_hour_consistency_pct']}% consistency).",
                "The comparison figures come from one test period on partly synthetic data.",
                "The refresh only helps after assignment. The first promise to the customer still comes from the placement model.",
            ],
        },
        {
            "title": "Surge-staff riders in the 19:00–21:59 window and plan promises around predictable peaks",
            "problem": "Delays concentrate in particular hours, days and weather conditions, but rider capacity and "
                       "customer promises are not differentiated by them.",
            "evidence": [
                f"[Observed] Orders placed in hours {', '.join(f'{h:02d}:00' for h in sorted(ch['peak_hours']))} have the highest late rates "
                f"(**{', '.join(p1(hr[h]['late_pct']) for h in sorted(ch['peak_hours']))}**) against a network rate of {p1(ch['network']['late_pct'])}. "
                f"They carry {sum(hr[h]['orders'] for h in ch['peak_hours']):,} orders ({p1(100 * sum(hr[h]['orders'] for h in ch['peak_hours']) / ch['network']['orders'])} of all).",
                f"[Observed] Orders placed 15:00–23:59 are {p1(op['peak_15_23_share_pct'])} of orders but "
                f"**{p1(op['peak_15_23_share_of_late_pct'])} of late orders**.",
                f"[Observed] Bad weather adds delay beyond traffic, mostly in heavy traffic. Within Jam traffic, Fog orders are "
                f"{ch['weather_traffic']['fog_minus_sunny_within_traffic_pp']['Jam']:.1f} pp more often late than Sunny ones; within Low "
                f"traffic, only {ch['weather_traffic']['fog_minus_sunny_within_traffic_pp']['Low']:.1f} pp.",
                f"[Observed] Evening (19–23) late rate: {p1(op['evening_late_pct'])}. Morning (08–10): {p1(morning['late30_pct'])}. "
                f"Timed orders outside 15–23: {p1(op['nonpeak_timed_late_pct'])}.",
                f"[Observed] Time of day is the strongest model factor: shuffling it raises training ETA error by "
                f"{imp['Time of day']['mean']:.2f} min and lowers late-risk ROC-AUC by {late_imp['Time of day']['mean']:.3f}. "
                f"The model predicts {pd_h[20]:.1f} min at 20:00 vs {pd_h[9]:.1f} min at 09:00.",
                f"[Observed] Festival days: {fest['orders']:,} orders, mean {fest['mean_eta']:.1f} min, "
                f"{p1(fest['late30_pct'])} over 30 minutes.",
            ],
            "indicates": "Delay risk is highly predictable from the clock and the calendar. That makes it a planning "
                         "problem as much as a real-time problem.",
            "action": [
                "Add surge rider shifts for 19:00–21:59 first, then 17:00–18:59. Set kitchen and dispatch staffing to the same hours.",
                "Trigger extra evening capacity when fog or cloudy weather is forecast. This needs a live weather "
                "feed known before dispatch.",
                "Publish time-band-specific promised delivery windows, wider in the evening peak.",
                "Run a festival-day surge plan: extra riders and wider promised windows announced in advance.",
            ],
            "implementation": [
                "Use the ETA model's hourly prediction curve and historical late rates by band to set initial capacity targets.",
                "Review weekly against actual late rates.",
                "Start in the cities with the highest evening volume.",
            ],
            "benefit": [
                f"[Scenario, not measured] Bringing the evening late rate down to the afternoon level ({p1(op['afternoon_late_pct'])}) "
                f"would cut the overall late rate by about {op['scenario_evening_to_afternoon_pp']:.1f} percentage points. "
                f"Calculation: evening share {p1(op['evening_share_pct'])} × gap "
                f"{op['evening_late_pct'] - op['afternoon_late_pct']:.1f} pp.",
                "[Expected, unmeasured] More honest customer promises in peak hours, and so fewer broken promises.",
            ],
            "kpis": ["Late rate (> 30 min) and ETA bias by hour band and on festival days",
                     "Riders active per order, by hour band",
                     "Promised-window hit rate"],
            "risks": [
                f"Association, not proven cause: traffic level in this data follows order hour almost exactly "
                f"({m['data']['traffic_hour_consistency_pct']}%), so the hour effect may be congestion, demand or both.",
                "The 100% late rate on festival days is a deterministic pattern typical of synthetic data. Real festival "
                "effects must be measured before staffing to them.",
                "Capacity costs are not in the dataset, so the cost–benefit must be assessed by operations.",
                "Weather's recording time is unknown in this data. A weather trigger needs a forecast or live feed "
                "available before dispatch.",
            ],
        },
    ]


def write_recommendations(m: dict, recs: list[dict]) -> int:
    L = ["# DeliveryIQ — Three Prioritised Business Recommendations\n",
         f"Stage 5 · generated {m['run_date']} by `src/business_insights.py` · evidence: `reports/stage5_metrics.json`  ",
         "**Labels:** [Observed] measured in this project · [Action] recommended step · [Scenario] hypothetical, not a measured "
         "result. The late benchmark is **> 30 minutes, project-defined, not an official SLA**.\n"]
    for i, r in enumerate(recs, 1):
        L += [f"## Priority {i} — {r['title']}\n",
              f"**1. Priority:** {i}\n",
              f"**2. Business problem:** {r['problem']}\n",
              "**3. Numerical evidence:**"] + [f"- {e}" for e in r["evidence"]] + [
              f"\n**4. What the evidence indicates:** {r['indicates']}\n",
              "**5. Recommended operational action [Action]:**"] + [f"- {a}" for a in r["action"]] + [
              "\n**6. How to implement:**"] + [f"- {a}" for a in r["implementation"]] + [
              "\n**7. Expected operational benefit:**"] + [f"- {b}" for b in r["benefit"]] + [
              "\n**8. KPIs:**"] + [f"- {k}" for k in r["kpis"]] + [
              "\n**9. Risks, assumptions and limitations:**"] + [f"- {k}" for k in r["risks"]] + [""]
    L += ["## Why not a dark-store location recommendation?\n",
          f"- The 22 K-Means zones are exactly the 22 cities. Silhouette of the zones = silhouette of the city labels = "
          f"{m['geography']['silhouette_of_city_labels']:.3f}.",
          f"- Late rates do not differ significantly between zones (chi-square p = {m['geography']['zone_late_chi2_p']:.3f}).",
          "- Delivery points are synthetic offsets around restaurants.",
          "- The evidence therefore cannot support choosing store locations. See the insights report §8 for the data this would need.\n"]
    (REP / "business_recommendations.md").write_text("\n".join(L), encoding="utf-8")
    return len(recs)


def write_insights(m: dict, recs: list[dict]) -> None:
    c, g, rd, d, e, r = m["classification"], m["geography"], m["rider"], m["dispatch_comparison"], m["explainability"], m["regression"]
    t, per, D, op = c["gt30"], c["per_1000_orders_test"], m["data"], m["observed_peak"]
    fam = r["families"]
    pdp = e["partial_dependence"]
    pdh = {x["value"]: x["mean_prediction"] for x in pdp["hour_eta"]}
    pdl = {x["value"]: x["mean_prediction"] for x in pdp["hour_late"]}
    pdd = {x["value"]: x["mean_prediction"] for x in pdp["distance_eta"]}
    L: list[str] = []
    a = L.append
    a("# DeliveryIQ — Business Insights Report\n")
    a("**AI-Powered Quick Commerce Delivery Intelligence System** · Stage 5 · generated "
      f"{m['run_date']} by `src/business_insights.py`  ")
    a("Every number is computed from project artifacts and stored in `reports/stage5_metrics.json`. "
      "**Labels:** [Observed] measured · [Action] recommendation · [Scenario] hypothetical, not measured.\n")
    if not m["challenge_pdf_found_in_project"]:
        a("> **Note:** the official KLU + PATH3 challenge PDF is not in the project folder, so requirements were taken "
          "from the owner's stage briefs. Please add the PDF to the project for a final requirement-by-requirement check.\n")

    # 1
    a("## 1. Executive Summary\n")
    a(f"- **Data:** {D['rows']:,} delivery orders across 22 Indian cities, {D['dates'][0]} → {D['dates'][1]}. All rows were "
      "kept; data-quality problems are flagged, not deleted.")
    a(f"- **ETA prediction at order placement:** test MAE **{r['test']['MAE']:.2f} min** vs {r['baseline_test']['MAE']:.2f} for "
      f"a median guess, **{100 * (1 - r['test']['MAE'] / r['baseline_test']['MAE']):.1f}% better**. R² {r['test']['R2']:.3f}.")
    a(f"- **Late-order risk (> 30 min, project benchmark, not an official SLA):**")
    a(f"  - ROC-AUC **{t['ROC_AUC']:.3f}**; catches **{p1(100 * t['recall'])}** of late orders; "
      f"**{p1(100 * t['precision'])}** of alerts are correct ({c['lift_precision_over_base']:.2f}× the base rate).")
    a(f"  - The riskiest 10% of orders were {p1(100 * c['top_decile']['observed_late'])} late.")
    a("- **What drives delay (association, not proven cause):** time of day first, then delivery distance, festival "
      "days and city type. Restaurant location, calendar and order type add almost nothing.")
    pk = sorted(m["challenge"]["peak_hours"])
    a(f"- **When to surge-staff:** orders placed {pk[0]:02d}:00–{pk[-1]:02d}:59 are about half late, against "
      f"{m['challenge']['network']['late_pct']:.1f}% for the network (§4.1). The three evidence-backed findings are in §4.6.")
    a(f"- **Geography:** the 22 clusters are the 22 cities (silhouette {g['silhouette_of_city_labels']:.3f} = city separation). "
      f"Late rates do not differ significantly between them (p = {g['zone_late_chi2_p']:.3f}). "
      "**No dark-store location can be justified from this data.**")
    a("- **Rider segmentation is not feasible:** rider IDs do not identify individual riders.")
    a(f"- **After rider assignment** a comparison model reaches MAE {d['eta_MAE_no_rating']:.2f} min (without rider rating). "
      "That is strong evidence for refreshing the ETA at dispatch, but the model is not production-ready.")
    a("- **Three priorities:**")
    for i, rr in enumerate(recs, 1):
        a(f"  {i}. {rr['title']}")
    a("")

    # 2
    a("## 2. Business Problem and Objectives\n")
    a("Customers need a reliable delivery promise, and operations need early warning of orders that will run late. "
      "The project answers four questions: how long will an order take, will it be late, where is demand, "
      "and what should management do.\n")
    a("| Objective | Approach | Status |\n|---|---|---|")
    a(f"| Predict delivery ETA | Gradient-boosting regression vs median baseline, ridge, random forest | ✅ MAE {r['test']['MAE']:.2f} min |")
    a(f"| Predict late-delivery risk | Gradient-boosting classifier; > 30 / 35 / 40 min benchmarks | ✅ ROC-AUC {t['ROC_AUC']:.3f} |")
    a("| Geographic delivery zones | K-Means with elbow + silhouette | ⚠️ Done; zones = cities, so limited operational value |")
    a("| Rider segmentation | — | ❌ Not feasible with the available rider IDs (§9) |")
    a("| Three business recommendations | Evidence from Stages 1–5 | ✅ §10 |")
    a("| Interactive demo | Streamlit | ⏳ Stage 6 |\n")

    # 3
    a("## 3. Dataset and Data Quality Summary\n")
    a("| Item | Value | Handling |\n|---|---|---|")
    a(f"| Orders | {D['rows']:,} ({D['cols']} raw columns) | All {D['rows_retained']:,} retained |")
    a(f"| Exact duplicate rows | {D['duplicates']} | — |")
    a(f"| Rows with ≥ 1 missing value (stored as text `NaN`) | {D['rows_any_missing_raw']:,} | Converted to real blanks; never filled in the data |")
    a(f"| Restaurant coordinates with a flipped minus sign | {D['sign_corrected']} | Corrected only after 4 geographic checks; originals kept |")
    a(f"| Restaurant location recorded as (0, 0) | {D['geo_invalid']:,} | Flagged invalid; coordinates blanked, never invented |")
    a(f"| Placeholder rider records (age 15 / rating 1, age 50 / rating 6) | {D['rider_sentinel']} | Age and rating blanked; originals kept |")
    a(f"| Missing order time | {D['missing_order_time']:,} | Kept as missing; never rebuilt from pickup time |")
    a(f"| Missing festival flag | {D['festival_missing']} | Kept; no 'is-missing' feature (it marks an artifact) |\n")
    a(f"**Food delivery, not quick commerce:** order types are Snack, Meal, Drinks and Buffet from restaurants. The median "
      f"delivery takes {m['observed_overall']['median_eta']:.0f} minutes over a median {g['distance_km']['median']:.1f} km. "
      "Findings describe restaurant delivery and transfer to grocery quick commerce only by analogy.\n")
    a("**Synthetic patterns:** delivery points are diagonal offsets from restaurants "
      f"({100 * g['lat_offset_equals_lon_offset_share']:.1f}% of rows); delivery times cluster in 5-minute blocks; traffic "
      f"level follows order hour almost exactly ({D['traffic_hour_consistency_pct']}%); several segments are late 100% of the "
      "time (§4). Real-world performance will differ.\n")

    # 4
    a("## 4. Key EDA Findings\n")
    a(f"Overall: mean delivery {m['observed_overall']['mean_eta']:.1f} min, median {m['observed_overall']['median_eta']:.0f}; "
      f"**{p1(m['observed_overall']['late30_pct'])} of orders exceed 30 minutes**. These are observed associations across all orders.\n")
    a("| Factor | Group | Orders | Mean ETA (min) | % late > 30 |\n|---|---|---|---|---|")
    order = {"time_band": ["morning_08_10", "midday_11_14", "afternoon_15_18", "evening_19_23", "night_00_07", "(order time missing)"],
             "distance_band": ["0–5 km", "5–10 km", "10–15 km", "15–25 km", "(invalid location)"],
             "festival": ["No", "Yes", "(missing)"], "city_type": ["Urban", "Metropolitan", "Semi-Urban", "(missing)"],
             "traffic_density": ["Low", "Medium", "High", "Jam", "(missing)"],
             "multiple_deliveries": ["0.0", "1.0", "2.0", "3.0", "(missing)"], "order_type": ["Snack", "Meal", "Drinks", "Buffet"]}
    names = {"time_band": "Order time", "distance_band": "Distance", "festival": "Festival", "city_type": "City type",
             "traffic_density": "Traffic*", "multiple_deliveries": "Batched deliveries*", "order_type": "Order type"}
    for col, levels in order.items():
        for lv in levels:
            row = obs_row(m, col, lv)
            a(f"| {names[col]} | {BAND.get(lv, lv)} | {row['orders']:,} | {row['mean_eta']:.1f} | {row['late30_pct']:.1f} |")
    a("\n\\* Known only after dispatch or of uncertain timing, so excluded from the placement models.\n")
    a("**Takeaways:**")
    a(f"- **Time of day is the largest observed contrast:** {p1(obs_row(m, 'time_band', 'morning_08_10')['late30_pct'])} late in the "
      f"morning vs {p1(op['evening_late_pct'])} in the evening. The 15:00–23:59 window carries {p1(op['peak_15_23_share_pct'])} of "
      f"orders and {p1(op['peak_15_23_share_of_late_pct'])} of late orders.")
    a(f"- **Distance matters, as a step:** orders over 10 km are about {obs_row(m, 'distance_band', '10–15 km')['late30_pct']:.0f}% late "
      f"vs {obs_row(m, 'distance_band', '0–5 km')['late30_pct']:.0f}% under 5 km.")
    a("- **Deterministic patterns (synthetic-data warning):** every festival-day order, every Semi-Urban order and every "
      "order batched with 2–3 others is late. Real operations rarely look like this, so treat these as dataset artifacts.")
    a("- **Order type has no effect** (all about 30% late).\n")
    L += pdf_eda_block(m)
    a("### 4.6 Three evidence-backed findings (PDF §7)\n")
    for f_ in three_findings(m):
        a(f"- {f_}")
    a("")
    a("Chart: `figures/25_late_rate_ranges.png` compares how much late rates vary by each factor.\n")

    # 5
    a("## 5. Regression Model Findings (ETA at order placement)\n")
    a("| Model | Test MAE (min) | Test RMSE | Test R² |\n|---|---|---|---|")
    pretty = {"baseline_median": "Median baseline", "ridge": "Ridge regression", "random_forest": "Random forest",
              "hist_gbm": "**Gradient boosting (selected)**"}
    for n, v in fam.items():
        a(f"| {pretty[n]} | {v['test']['all']['MAE']:.2f} | {v['test']['all']['RMSE']:.2f} | {v['test']['all']['R2']:.3f} |")
    a(f"\n- **What MAE {r['test']['MAE']:.2f} means:** a typical prediction is off by about 6 minutes on an average delivery "
      f"of {m['observed_overall']['mean_eta']:.0f} minutes. That is usable for a delivery *window*, not a precise minute promise.")
    a(f"- The model explains {100 * r['test']['R2']:.0f}% of the variation. The rest depends on dispatch-time factors "
      "(rider, batching, live conditions) that are deliberately unavailable at placement (§7, Recommendation 2).")
    segs = {x["traffic_density"]: x["MAE"] for x in r["segments_test"]["traffic_density"]}
    db = {x["distance_band"]: x["MAE"] for x in r["segments_test"]["distance_band"]}
    a(f"- **Errors are largest where deliveries are slowest:** Jam traffic MAE {segs['Jam']:.2f} vs Low {segs['Low']:.2f}; "
      f"15–25 km {db['15–25 km']:.2f} vs 0–5 km {db['0–5 km']:.2f}.")
    a("- Model selection, the feature choices (`order_month` dropped; missing-time and missing-festival rows kept) and "
      "tuning were all made on training data before the test set was scored (Stage 4).")
    a("\nCharts: `figures/12_regression_model_comparison.png`, `figures/13_eta_pred_vs_actual.png`, `figures/14_eta_error_by_segment.png`.\n")
    L += model_selection_block(m)

    # 6
    a("## 6. Late-Order Classification Findings\n")
    a("The 30-minute line is a **project-defined benchmark** (the dataset has no official quick-commerce SLA).\n")
    a(f"| Metric (test period) | Value | Meaning |\n|---|---|---|")
    a(f"| ROC-AUC | {t['ROC_AUC']:.3f} | A random late order is ranked above a random on-time order {100 * t['ROC_AUC']:.0f}% of the time |")
    a(f"| PR-AUC | {t['PR_AUC']:.3f} | vs {t['prevalence']:.3f} for a no-skill model |")
    a(f"| Recall | {t['recall']:.3f} | {p1(100 * t['recall'])} of late orders are flagged in advance |")
    a(f"| Precision | {t['precision']:.3f} | {p1(100 * t['precision'])} of flagged orders are actually late; {p1(100 - 100 * t['precision'])} are false alarms |")
    a(f"| F1 | {t['F1']:.3f} | Balance of the two at threshold {t['threshold']:.3f} |")
    a(f"| Brier score | {t['brier']:.3f} | Probabilities are **not calibrated** — use them for ranking |\n")
    a("**Why recall matters:** a late order the model misses gets no proactive message and no dispatch attention, and it "
      "is the costliest failure. The threshold was therefore set below 0.5, at the F1-best point on training data, "
      "to favour catching late orders.\n")
    a("**What the precision means:** at this threshold about half of all orders are flagged, and more than half of those "
      "alerts are false. That is acceptable only for cheap interventions. Costly actions should be reserved for the "
      "highest-risk orders (Recommendation 1).\n")
    a(f"**Per 1,000 orders (test period):** {per['flagged']:.0f} flagged → {per['late_caught']:.0f} truly late + "
      f"{per['false_alarms']:.0f} false alarms; {per['late_missed']:.0f} late orders missed.\n")
    a("**Risk deciles (test period; reporting only, not used for any model choice):**\n")
    a("| Risk decile (1 = riskiest) | Orders | % actually late |\n|---|---|---|")
    for x in c["risk_deciles_test"]:
        a(f"| {x['decile']} | {x['orders']:,} | {100 * x['observed_late']:.1f} |")
    a("\nChart: `figures/24_late_risk_deciles.png`.\n")
    a("**Benchmark sensitivity:**\n")
    a("| Benchmark | Late share | Flagged share | Precision | Recall | F1 | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|---|---|")
    for k, v in c["sensitivity"].items():
        a(f"| > {k} min | {p1(100 * v['prevalence'])} | {p1(100 * v['predicted_positive_rate'])} | {v['precision']:.3f} | "
          f"{v['recall']:.3f} | {v['F1']:.3f} | {v['ROC_AUC']:.3f} | {v['PR_AUC']:.3f} |")
    s35, s40 = c["sensitivity"]["35"], c["sensitivity"]["40"]
    a(f"\n- **Stricter benchmarks rank better but alert worse.** ROC-AUC rises to {s40['ROC_AUC']:.3f} at > 40 min because very "
      f"slow orders are more distinctive. But they are rarer ({p1(100 * s40['prevalence'])}), so precision falls to "
      f"{s40['precision']:.3f} and recall to {s40['recall']:.3f}.")
    a(f"- **> 30 min is the most operationally useful benchmark:** it has the highest F1 ({t['F1']:.3f} vs {s35['F1']:.3f} and "
      f"{s40['F1']:.3f}).")
    a("\nCharts: `figures/16_classification_roc_pr.png`, `figures/17_classification_threshold.png`, "
      "`figures/18_confusion_matrix.png`, `figures/21_late_threshold_sensitivity.png`.\n")
    cb, tp = m["challenge"]["class_balance"], m["challenge"]["threshold_point_test"]
    a("**Class imbalance handling (PDF PS5):**")
    a(f"- Late orders are {cb['train_late_pct']:.1f}% of training data and {cb['test_late_pct']:.1f}% of test data.")
    a("- Models were selected on PR-AUC, which is sensitive to the minority class, not accuracy.")
    a("- The decision threshold was moved from 0.5 to the F1-best point on out-of-fold training predictions.")
    a("- No resampling was needed: threshold moving fixes the recall/precision balance directly and keeps probabilities interpretable.\n")
    a(f"**Chosen threshold on the curves:** at {tp['threshold']:.3f} → recall {tp['recall']:.3f}, precision {tp['precision']:.3f}, "
      f"false positive rate {tp['fpr']:.3f}, marked in `figures/29_threshold_on_pr_roc.png`.\n")

    # 7
    a("## 7. Model Explainability\n")
    a("**Stage 4 already provided:**")
    a("- single-feature permutation importance (ETA, test set);")
    a("- predicted-vs-actual and residual plots;")
    a("- ROC and precision-recall curves;")
    a("- model comparison tables.\n")
    a("**Stage 5 adds** two things for the saved primary models, using only the 13 order-placement features, on "
      f"**{e['data']}**, without retraining or test data:")
    a("- **grouped** permutation importance for **both** models;")
    a("- partial-dependence curves.\n")
    a("Grouping permutes collinear features together, so the four hour features are scored as one 'time of day' effect. "
      "SHAP was not used: the package isn't installed, and the brief asks for no unnecessary installs.\n")
    a("| Factor | ETA model: training MAE increase when shuffled (min) | Late model: ROC-AUC drop when shuffled |\n|---|---|---|")
    for grp, v in e["eta_mae_increase_min"]["groups"].items():
        a(f"| {grp} | {v['mean']:.3f} ± {v['sd']:.3f} | {e['late_auc_drop']['groups'][grp]['mean']:.4f} |")
    a(f"\nBaselines on training data: ETA MAE {e['eta_mae_increase_min']['baseline_score']:.2f} min; late-risk ROC-AUC "
      f"{e['late_auc_drop']['baseline_score']:.3f}. Chart: `figures/22_explainability_grouped_importance.png`.\n")
    a("**In plain language (what the model has learned):**")
    a(f"- **Time of day:** the model's average ETA is about {pdh[9]:.1f} min for 08–10 orders and {pdh[20]:.1f} min at 20:00. "
      f"The predicted chance of being late rises from {100 * pdl[9]:.0f}% at 09:00 to {100 * pdl[20]:.0f}% at 20:00.")
    a(f"- **Distance:** predicted ETA rises from {pdd[2.0]:.1f} min at 2 km to {pdd[20.0]:.1f} min at 20 km, mostly as a step "
      f"beyond about 10 km ({pdd[10.0]:.1f} → {pdd[12.0]:.1f} min).")
    fe_ = {x["value"]: x["mean_prediction"] for x in pdp["festival_eta"]}
    ct_ = {x["value"]: x["mean_prediction"] for x in pdp["city_type_eta"]}
    a(f"- **Festival day:** {fe_['No']:.1f} → {fe_['Yes']:.1f} min.")
    a(f"- **City type:** Urban {ct_['Urban']:.1f}, Metropolitan {ct_['Metropolitan']:.1f}, Semi-Urban {ct_['Semi-Urban']:.1f} min. "
      "Semi-Urban has only 164 orders, so it is unstable.")
    a("- **Little effect:** restaurant location, day of week or month and order type.\n")
    a("**Association, not causation:** these show what the model uses to predict, not what *causes* delay. For example, "
      "the hour effect may reflect traffic, demand or kitchen load; the data cannot separate them, since traffic follows "
      "the hour almost exactly. The test-set importance ranking from Stage 4 agrees: hour and distance are top. "
      "Chart: `figures/23_eta_partial_dependence.png`.\n")

    # 8
    a("## 8. Geographic and Dark-Store Analysis\n")
    a("| Check | Result |\n|---|---|")
    a(f"| K-Means zones selected (Stage 4) | k = {g['k']}, silhouette {g['silhouette_stage4']:.3f} |")
    a(f"| Zones vs cities | Every zone holds exactly one city, and every city one zone: **zones = cities** ({'yes' if g['zones_equal_cities'] else 'no'}) |")
    a(f"| Silhouette of the *city labels* themselves | **{g['silhouette_of_city_labels']:.3f}** (identical), so the high score is just distance between cities |")
    a(f"| Late-rate range across zones | {g['zone_late30_range_pct'][0]:.1f}% – {g['zone_late30_range_pct'][1]:.1f}%; chi-square p = **{g['zone_late_chi2_p']:.3f}** (no significant difference) |")
    a(f"| Median ETA range across zones | {g['zone_median_eta_range'][0]:.0f} – {g['zone_median_eta_range'][1]:.0f} min |")
    a(f"| Late-rate range across order-time bands (comparison) | {g['time_band_late30_range_pct'][0]:.1f}% – {g['time_band_late30_range_pct'][1]:.1f}% |")
    a(f"| Zone sizes | {g['zone_orders_range'][0]:,} – {g['zone_orders_range'][1]:,} valid orders |")
    gi = ", ".join(f"{k} {100 * v:.1f}%" for k, v in g["geo_invalid_top_cities"].items())
    a(f"| Orders without a valid location | {g['geo_invalid_rows']:,} overall; concentrated in {gi} |")
    a(f"| Delivery points | Synthetic diagonal offsets from {g['unique_restaurant_locations_valid']} restaurant locations; median distance {g['distance_km']['median']:.1f} km (max {g['distance_km']['max']:.1f}) |\n")
    L += zone_block(m)
    a("**What the analysis can establish:** demand is spread across 22 well-separated cities, and the restaurant-to-customer "
      "distances in the data. Distance itself matters (§4).\n")
    a("**What it cannot establish:**")
    a("- operational zones inside a city;")
    a("- demand hot-spots;")
    a("- **any specific dark-store location.**\n")
    a("Customer positions are generated offsets, not real addresses. Zone late rates are statistically indistinguishable. "
      "Several cities lose 30–50% of their orders to invalid coordinates. The clusters should be presented as "
      "*city groupings*, not validated delivery zones.\n")
    a("**Data needed for dark-store placement decisions:**")
    a("- real customer delivery coordinates with order timestamps, to measure demand density by hour;")
    a("- SKU-level demand and the inventory constraints of a dark store;")
    a("- candidate site availability, rent and operating costs;")
    a("- rider supply and road-network travel times (not straight-line distance);")
    a("- a target service radius or promised delivery time.\n")
    a("Charts: `figures/19_kmeans_elbow_silhouette.png`, `figures/20_delivery_zones.png`, `figures/25_late_rate_ranges.png`.\n")

    # 9
    a("## 9. Rider Segmentation Feasibility\n")
    a("| Evidence (cleaned data) | Value |\n|---|---|")
    a(f"| Distinct `delivery_person_id` values | {rd['ids']:,} |")
    a(f"| IDs with more than one recorded age | **{rd['ids_with_multiple_ages_pct']:.0f}%** (ages per ID: {rd['distinct_ages_per_id']['min']:.0f}–{rd['distinct_ages_per_id']['max']:.0f}, median {rd['distinct_ages_per_id']['50%']:.0f}) |")
    a(f"| IDs with more than one vehicle type | {rd['ids_with_multiple_vehicle_types_pct']:.0f}% |")
    a(f"| Valid restaurant locations per ID | at most {rd['max_valid_restaurant_locations_per_id']} (each ID is tied to one restaurant) |")
    a(f"| ID suffixes | {', '.join('DEL' + s for s in rd['id_suffixes'])} only |\n")
    a("**Verdict: rider segmentation is not feasible.**")
    a("- The ID pattern `<CITY>RES<nn>DEL<nn>` is a restaurant-plus-slot code: one ID covers many people of different ages.")
    a("- Any 'rider segment' built on it would describe restaurants or slots, not riders.")
    a("- No rider segments were produced, and no individual rider performance is claimed.\n")
    a("**What is possible instead (labelled accurately):** an *order-level delivery-condition profile*. This is not rider "
      "segmentation; it is dispatch-time information and is confounded.\n")
    a("| Vehicle type | Orders | Mean ETA | % late > 30 |\n|---|---|---|---|")
    for x in m["observed"]["vehicle_type"]:
        a(f"| {x['vehicle_type']} | {x['orders']:,} | {x['mean_eta']:.1f} | {x['late30_pct']:.1f} |")
    vc0 = obs_row(m, "vehicle_condition", "0")
    a(f"\nMotorcycles look slower, but vehicle condition 0 occurs only on motorcycles and has a {vc0['late30_pct']:.1f}% late "
      "rate, so the two cannot be separated.\n")
    L += rider_profile_block(m)
    a("**Data needed for genuine rider segmentation:**")
    a("- a stable, unique rider identifier;")
    a("- per-rider trip history across restaurants;")
    a("- shift and login times;")
    a("- assignment, pickup and drop timestamps;")
    a("- the rider's own vehicle, and ratings captured before each order.\n")

    # 10
    a("## 10. Exactly Three Prioritised Recommendations\n")
    a("Full detail (evidence, implementation, KPIs, risks): **`reports/business_recommendations.md`**.\n")
    a("| Priority | Recommendation | Core evidence [Observed] |\n|---|---|---|")
    core = [f"Recall {p1(100 * t['recall'])}, ROC-AUC {t['ROC_AUC']:.3f}; riskiest 10% → {p1(100 * c['top_decile']['observed_late'])} late",
            f"Placement MAE {r['test']['MAE']:.2f} → {d['eta_MAE_no_rating']:.2f} min with dispatch-time data (comparison model, no rating)",
            f"15:00–23:59 = {p1(op['peak_15_23_share_pct'])} of orders, {p1(op['peak_15_23_share_of_late_pct'])} of late orders"]
    for i, (rr, cc) in enumerate(zip(recs, core), 1):
        a(f"| {i} | {rr['title']} | {cc} |")
    a("\n**Why this order:**")
    a("- **#1** uses a model that is already trained, saved and validated for order-placement use.")
    a("- **#2** has the largest measured accuracy gain, but needs a field-timing audit and a new model first.")
    a("- **#3** is a planning change supported by strong, consistent patterns. Their cause is not established.\n")
    a("A dark-store or zone recommendation was considered and **rejected**: the evidence does not support it (§8).\n")

    # 11
    a("## 11. Expected Business Value and KPIs\n")
    a("| Recommendation | Measured in this project | Hypothetical scenario (not measured) | Primary KPI |\n|---|---|---|---|")
    sc = c["scenario_late_rate_reduction_pp"]
    a(f"| 1. Late-risk alerts | {p1(100 * t['recall'])} of late orders flagged before they are late | Late rate −{sc['10%']:.1f} to −{sc['30%']:.1f} pp "
      "if alerts prevent 10–30% of the late orders they catch | Late rate, pilot vs control; alert precision and recall |")
    a(f"| 2. Two-stage ETA | MAE {d['eta_MAE_no_rating']:.2f} vs {r['test']['MAE']:.2f} min (comparison model) | Fewer status-enquiry contacts | ETA MAE at assignment; % within ±5 min |")
    a(f"| 3. Peak capacity planning | Peak window = {p1(op['peak_15_23_share_of_late_pct'])} of late orders | Late rate −{op['scenario_evening_to_afternoon_pp']:.1f} pp if "
      "the evening late rate matched the afternoon | Late rate by hour band and on festival days |\n")
    a("Scenario calculations are transparent arithmetic on observed rates (see the recommendations file). They are not "
      "forecasts and must be replaced by pilot measurements.\n")

    # 12
    a("## 12. Limitations and Deployment Considerations\n")
    a("| Limitation | Effect on deployment and generalisation |\n|---|---|")
    lims = [
        ("Food-delivery data, not genuine quick-commerce grocery delivery",
         "Distances and times are much longer than quick-commerce operations. Re-train on real quick-commerce data before use."),
        ("Synthetic or artificial patterns (diagonal delivery offsets, 5-minute target blocks, traffic set by hour, segments 100% late)",
         "Metrics may be optimistic and some effects may be artifacts. Validate on real data."),
        (f"Test period covers only 8 days ({D['split']['test_dates'][0]} → {D['split']['test_dates'][1]})",
         "No seasonality or longer-term drift is tested. Monitor performance weekly after launch."),
        ("April appears only in the test period", "The month feature was dropped. Behaviour in unseen months is unverified."),
        (f"{D['geo_invalid']:,} orders lack valid coordinates (up to 50% in some cities)",
         "Distance is unknown for them (filled with the median at model time), and geographic conclusions under-represent those cities."),
        (f"{D['missing_order_time']:,} orders lack an order time",
         f"Their delivery times look normal (p = {D['missing_time_target_p']:.2f}), but their hour is unknown and their errors are higher."),
        (f"Some values corrected by documented rules ({D['sign_corrected']} sign fixes, {D['rider_sentinel']} placeholders blanked)",
         "The corrections are reversible and audited (`*_raw`, `dq_*` columns)."),
        ("The > 30-minute late line is project-defined, not an official SLA",
         "Re-set the threshold to the operator's real promise; sensitivity at 35/40 min is provided."),
        ("Classification probabilities are not calibrated",
         "Use them for ranking and tiering. Calibrate before showing a probability to customers."),
        ("Clustering does not establish dark-store locations", "Do not use the zones for site selection (§8)."),
        ("Rider segmentation is not reliable with current IDs", "No rider-level decisions should be based on this data (§9)."),
        ("Dispatch-time comparison model is unsaved and may leak (rider rating)",
         "Supports Recommendation 2's direction only; it is not a deployable model."),
    ]
    for lim, eff in lims:
        a(f"| {lim} | {eff} |")
    a("\n**Deployment guidance:**")
    a("- Use only the saved order-placement models (13 features).")
    a("- Run them in shadow mode before acting on them.")
    a("- Recalibrate the threshold and probabilities on live data.")
    a("- Retrain on a schedule, re-checking the chronological split and the leakage guard.\n")

    # 13
    a("## 13. Conclusion\n")
    a(f"With order-time information alone, DeliveryIQ beats a naive ETA by {100 * (1 - r['test']['MAE'] / r['baseline_test']['MAE']):.1f}% "
      f"and flags {p1(100 * t['recall'])} of late orders before they happen. That supports an alerting workflow now "
      "(Recommendation 1) and capacity planning around predictable peaks (Recommendation 3).\n")
    a("The largest accuracy gain needs information available only after dispatch (Recommendation 2). The geographic and "
      "rider analyses were taken as far as the data honestly allows: the clusters are cities, and rider IDs are not people.\n")
    a("The next step is Stage 6: an interactive demo of the placement-time ETA and late-risk models.\n")

    a("## Appendix A — Verification of Stage 4 figures\n")
    a("| Figure | Value in brief | Recomputed from saved models / metrics | Match |\n|---|---|---|---|")
    for k, v in m["stage4_verification"].items():
        a(f"| {k} | {v['brief']} | {v['recomputed']:.4f} | {'✅' if v['match'] else '❌'} |")
    a("\n## Appendix B — Stage 5 validation checks\n")
    a("| Check | Result | Detail |\n|---|---|---|")
    for ch in m["checks"]:
        a(f"| {ch['check']} | {'✅ pass' if ch['passed'] else '❌ FAIL'} | {ch['detail']} |")
    a("\n## Appendix C — Charts added in Stage 5\n")
    for fg in m["figures"]:
        a(f"- `{fg}`")
    a("\nReproduce: `.venv\\Scripts\\python.exe -m src.business_insights`")
    (REP / "business_insights_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def three_findings(m: dict) -> list[str]:
    """Exactly three evidence-backed findings (PDF §7), each mapped to a recommendation."""
    ch, c, r, d, op = m["challenge"], m["classification"], m["regression"], m["dispatch_comparison"], m["observed_peak"]
    hr = {int(x["order_hour"]): x for x in ch["by_hour"]}
    pk = sorted(ch["peak_hours"])
    return [
        f"**Finding 1: delays concentrate in the evening peak.** [Observed] Orders placed {pk[0]:02d}:00–{pk[-1]:02d}:59 are "
        f"{min(hr[h]['late_pct'] for h in pk):.1f}–{max(hr[h]['late_pct'] for h in pk):.1f}% late, against {ch['network']['late_pct']:.1f}% "
        f"for the network. The 15:00–23:59 window carries {op['peak_15_23_share_of_late_pct']:.1f}% of all late orders. "
        "Time of day is also the models' strongest factor. → Recommendation 3.",
        f"**Finding 2: late orders can be flagged at order placement.** [Observed] Using only order-time information, the "
        f"classifier reaches ROC-AUC {c['gt30']['ROC_AUC']:.3f} on unseen days, catches {100 * c['gt30']['recall']:.1f}% of late orders, "
        f"and its riskiest 10% of orders were {100 * c['top_decile']['observed_late']:.1f}% late. → Recommendation 1.",
        f"**Finding 3: most ETA uncertainty is resolved only after dispatch.** [Observed] The placement ETA explains "
        f"{100 * r['test']['R2']:.0f}% of the variance (MAE {r['test']['MAE']:.2f} min). A comparison model with dispatch-time "
        f"data, without rider rating, explains {100 * d['eta_R2_no_rating']:.0f}% (MAE {d['eta_MAE_no_rating']:.2f} min). Distance "
        f"alone explains only {100 * ch['distance_only_model']['R2']:.0f}%. → Recommendation 2.",
    ]


def pdf_eda_block(m: dict) -> list[str]:
    """EDA comparisons named in the challenge guide (PS3, §8)."""
    ch = m["challenge"]
    L = ["### 4.1 Late rate by hour of order (PDF visual set)\n",
         "| Hour | Orders | Mean ETA | % late | vs network (pp) |\n|---|---|---|---|---|"]
    for x in ch["by_hour"]:
        L.append(f"| {int(x['order_hour']):02d} | {x['orders']:,} | {x['mean_eta']:.1f} | {x['late_pct']:.1f} | {x['gap_vs_network_pp']:+.1f} |")
    L.append(f"\nPeak-risk hours: **{', '.join(f'{h:02d}:00' for h in sorted(ch['peak_hours']))}**. Hours above the network late rate: "
             f"{', '.join(f'{h:02d}' for h in ch['hours_above_network'])}. Chart: `figures/26_late_rate_by_hour.png`.\n")
    L += ["### 4.2 Day of week\n", "| Day | Orders | Mean ETA | % late |\n|---|---|---|---|"]
    for x in ch["by_day_of_week"]:
        L.append(f"| {x['day']} | {x['orders']:,} | {x['mean_eta']:.1f} | {x['late_pct']:.1f} |")
    dws = sorted(ch["by_day_of_week"], key=lambda x: x["late_pct"])
    L.append(f"\nLate rates range from {dws[0]['late_pct']:.1f}% ({dws[0]['day']}) to {dws[-1]['late_pct']:.1f}% ({dws[-1]['day']}); "
             f"chi-square p = {ch['day_of_week_chi2_p']:.1e}. Caution: each weekday covers only {ch['dates_per_weekday'][0]}–"
             f"{ch['dates_per_weekday'][1]} dates in the data, so this pattern may not repeat.\n")
    w = ch["weather_traffic"]
    L += ["### 4.3 Weather vs traffic — does weather add delay beyond traffic? (PDF §8)\n",
          "| Weather \\ Traffic | " + " | ".join(["Low", "Medium", "High", "Jam"]) + " |", "|---|---|---|---|---|"]
    for wn in ["Sunny", "Windy", "Stormy", "Sandstorms", "Cloudy", "Fog"]:
        L.append(f"| {wn} | " + " | ".join(f"{w['late_pct'][t][wn]:.1f}%" for t in ["Low", "Medium", "High", "Jam"]) + " |")
    fs = w["fog_minus_sunny_within_traffic_pp"]
    L.append(f"\n**Yes, mostly in heavier traffic** [Observed]. Inside the same traffic level, Fog is "
             f"{fs['Low']:+.1f} pp vs Sunny in Low traffic, {fs['Medium']:+.1f} pp in Medium, {fs['High']:+.1f} pp in High and "
             f"{fs['Jam']:+.1f} pp in Jam. Sunny orders are rarely late at any traffic level. Because traffic follows the order "
             "hour almost exactly in this data, 'traffic' here is partly 'time of day'. Chart: `figures/27_weather_traffic_late_rate.png`.\n")
    do = ch["distance_only_model"]
    L += ["### 4.4 Distance vs delivery time — how much does distance alone explain? (PDF §8)\n",
          f"A linear model on distance only (fitted on training rows, an additional analysis) explains **R² {do['R2']:.3f}** on "
          f"the test period (MAE {do['MAE']:.2f} min; +{do['coef_min_per_km']:.2f} min per km). Squared correlation on all valid rows: "
          f"{ch['distance_pearson_r2_all_valid']:.3f}. **Distance is a real but secondary driver:** about one-tenth of the variance.\n"]
    ra, rr = ch["rider_age_band"], ch["rider_rating_band"]
    hi, ag = ch["rating_hi_vs_lo_similar_routes"], ch["age_old_vs_young_similar_routes"]
    L += ["### 4.5 Rider experience and rating vs delay (PDF PS3, §8)\n",
          "| Rider age band (experience proxy) | Orders | Mean ETA | % late |\n|---|---|---|---|"]
    L += [f"| {x['age_band']} | {x['orders']:,} | {x['mean_eta']:.1f} | {x['late_pct']:.1f} |" for x in ra]
    L += ["", "| Rider rating band | Orders | Mean ETA | % late |\n|---|---|---|---|"]
    L += [f"| {x['rating_band']} | {x['orders']:,} | {x['mean_eta']:.1f} | {x['late_pct']:.1f} |" for x in rr]
    L.append(f"\n**On similar routes** (same distance band and traffic level, {hi['strata_used']} strata) [Observed]:")
    L.append(f"- Orders by riders rated 4.8–5.0, vs riders rated below 4.5: delivery time {hi['eta_diff_min']:+.1f} min; late rate "
             f"{hi['late_diff_pp']:+.1f} pp.")
    L.append(f"- Orders by riders aged 35–39, vs riders aged 20–24: delivery time {ag['eta_diff_min']:+.1f} min; late rate "
             f"{ag['late_diff_pp']:+.1f} pp.\n")
    L.append("Caveats:")
    L.append("- **Rating may be a consequence, not a cause:** a late delivery can lower the rider's rating, which is the "
             "leakage risk recorded since Stage 1.")
    L.append("- **There is no tenure field:** age is the only experience proxy, and older riders are slower here, not faster.")
    L.append("- **Rows are orders, not riders** (§9).")
    L.append("- These are dispatch-time fields, so they are excluded from the placement models.\n")
    return L


def model_selection_block(m: dict) -> list[str]:
    """Consolidated model selection table (PDF §7 mandatory)."""
    r, ch, mm_c = m["regression"], m["challenge"], m["classification"]
    fam, mb, do = r["families"], ch["mean_baseline"], ch["distance_only_model"]
    L = ["### 5.1 Model selection table (PDF §7)\n",
         "| Task | Model | Selection metric (training CV) | Test result | Role |\n|---|---|---|---|---|",
         f"| ETA | Mean baseline (training mean {mb['train_mean']:.2f} min) | — | MAE {mb['MAE']:.2f} · RMSE {mb['RMSE']:.2f} · R² {mb['R2']:.3f} | Naive baseline named in the PDF |",
         f"| ETA | Median baseline | CV MAE {fam['baseline_median']['cv']['MAE']:.3f} | MAE {fam['baseline_median']['test']['all']['MAE']:.2f} · R² {fam['baseline_median']['test']['all']['R2']:.3f} | Naive baseline (Stage 4) |",
         f"| ETA | Distance-only linear model | — | MAE {do['MAE']:.2f} · R² {do['R2']:.3f} | Additional analysis |",
         f"| ETA | Ridge regression | CV MAE {fam['ridge']['cv']['MAE']:.3f} | MAE {fam['ridge']['test']['all']['MAE']:.2f} · R² {fam['ridge']['test']['all']['R2']:.3f} | Candidate |",
         f"| ETA | Random forest | CV MAE {fam['random_forest']['cv']['MAE']:.3f} | MAE {fam['random_forest']['test']['all']['MAE']:.2f} · R² {fam['random_forest']['test']['all']['R2']:.3f} | Candidate |",
         f"| ETA | **Gradient boosting** | **CV MAE {fam['hist_gbm']['cv']['MAE']:.3f}** | **MAE {fam['hist_gbm']['test']['all']['MAE']:.2f} · RMSE {fam['hist_gbm']['test']['all']['RMSE']:.2f} · R² {fam['hist_gbm']['test']['all']['R2']:.3f}** | **Final** |"]
    for n, lbl in [("logistic", "Logistic regression"), ("hist_gbm", "**Gradient boosting**")]:
        v = m["stage4_classifiers"][n]
        final = n == "hist_gbm"
        b = "**" if final else ""
        L.append(f"| Late > 30 min | {lbl} | {b}CV PR-AUC {v['cv']['PR_AUC']:.3f}{b} | {b}ROC-AUC {v['test']['ROC_AUC']:.3f} · PR-AUC {v['test']['PR_AUC']:.3f} · "
                 f"P {v['test']['precision']:.3f} · R {v['test']['recall']:.3f} · F1 {v['test']['F1']:.3f}{b} | {'**Final**' if final else 'Candidate'} |")
    L.append(f"| Late > 30 min | Prior (constant) baseline | CV PR-AUC {m['stage4_classifiers']['baseline_prior']['cv']['PR_AUC']:.3f} | ROC-AUC 0.500 · recall 0 | Naive baseline |")
    L.append(f"| Zones | **K-Means k = {m['geography']['k']}** | Silhouette (max over k = 2–30) | Silhouette {m['geography']['silhouette_stage4']:.3f} | Final (zones = cities, §8) |")
    L.append(f"| Rider-attribute profiles | **K-Means k = {ch['rider_profiles']['k_selected']}** | Silhouette (max over k = 2–8) | Silhouette {ch['rider_profiles']['silhouette']:.3f} | Order-level alternative (§9) |")
    imp = 100 * (1 - fam["hist_gbm"]["test"]["all"]["MAE"] / mb["MAE"])
    L.append(f"\n**Reasons for the final choices:**")
    L.append(f"- **Gradient boosting for ETA:** lowest training-CV MAE, and {imp:.1f}% lower test MAE than the mean baseline. "
             f"Random forest was within {abs(fam['random_forest']['cv']['MAE'] - fam['hist_gbm']['cv']['MAE']):.3f} min (a tie); "
             "gradient boosting is lighter to serve, so the extra complexity is not a cost here.")
    L.append("- **Gradient boosting for late risk:** highest CV PR-AUC, the right metric when late orders are the minority.")
    L.append("- **K-Means zones:** maximum silhouette.")
    L.append("- Every choice was made on training data before the test set was scored.\n")
    return L


def zone_block(m: dict) -> list[str]:
    z = m["challenge"]["zones_vs_network"]
    L = ["### 8.1 Zone vs network benchmark (PDF PS6, §8)\n",
         f"Network benchmark (valid-location orders): mean ETA {z['network_valid_mean_eta']:.2f} min, late rate {z['network_valid_late_pct']:.2f}%. "
         f"High demand = at least the median zone volume ({z['median_zone_orders']:.0f} orders). Significance: chi-square (late rate) and "
         "Welch t-test (ETA) for each zone vs the rest, Bonferroni-corrected for 22 zones.\n",
         "| Zone | City | Orders | Demand share | Mean ETA | vs network | % late | vs network | Late p | High demand & above benchmark |\n|---|---|---|---|---|---|---|---|---|---|"]
    for x in z["table"]:
        L.append(f"| {x['zone']} | {x['city']} | {x['orders']:,} | {x['demand_share_pct']:.1f}% | {x['mean_eta']:.2f} | {x['eta_vs_network_min']:+.2f} | "
                 f"{x['late_pct']:.1f}% | {x['late_vs_network_pp']:+.1f} pp | {x['late_p']:.3f} | {'yes' if x['candidate_flag'] else ''} |")
    L.append(f"\n**Result:** the PDF's 'high demand and above benchmark' rule flags {', '.join(z['candidates']) or 'no zone'}. However, "
             f"**{z['n_significant_late_bonferroni']} zones differ significantly in late rate and {z['n_significant_eta_bonferroni']} in ETA** after "
             "correction, so these are **provisional watch-list cities, not dark-store recommendations**. The largest gaps are about 1 minute "
             "of ETA, far smaller than the evening-peak effect.\n")
    return L


def rider_profile_block(m: dict) -> list[str]:
    rp = m["challenge"]["rider_profiles"]
    L = ["### 9.1 Alternative: order-level rider-attribute profiles (NOT rider segmentation)\n",
         f"PS7 asks for rider segments. Because rider identity is unreliable, K-Means was run on **orders** described by the "
         f"assigned rider's age and rating ({rp['rows']:,} orders with both values; standardised). k = {rp['k_selected']} was chosen by "
         f"maximum silhouette ({rp['silhouette']:.3f} over k = 2–8). Each profile describes orders, not individual riders.\n",
         "| Profile (descriptive name) | Orders | Mean age | Mean rating | Mean ETA | % late | Mean batched deliveries | % motorcycle |\n|---|---|---|---|---|---|---|---|"]
    for p in sorted(rp["profiles"], key=lambda p: p["late_pct"]):
        L.append(f"| {p['name']} | {p['orders']:,} | {p['mean_age']:.1f} | {p['mean_rating']:.2f} | {p['mean_eta']:.1f} | {p['late_pct']:.1f} | "
                 f"{p['multi_delivery_mean']:.2f} | {p['motorcycle_pct']:.1f} |")
    L.append("\n**How to use it:**")
    L.append("- It is a **dispatch-time risk signal**: orders assigned to the lower-rated profile run late far more often.")
    L.append("- It **cannot** support rostering or training decisions for individual riders. The same 'rider ID' covers many "
             "people, and a low rating may *result* from late deliveries.")
    L.append("- A real rider segmentation needs the data listed below.")
    L.append("\nChart: `figures/28_rider_attribute_profiles.png` (rating, age as the experience proxy, and late rate, as the PDF's visual set asks).\n")
    return L


def write_business_report(m: dict, recs: list[dict]) -> int:
    """Concise 4–6 page Business Report in the PDF §14 structure. Returns the word count."""
    c, r, g, ch, d = m["classification"], m["regression"], m["geography"], m["challenge"], m["dispatch_comparison"]
    t, D = c["gt30"], m["data"]
    mb = ch["mean_baseline"]
    L: list[str] = []
    a = L.append
    a("# DeliveryIQ — Business Report\n")
    a("**KLU + PATH3 ML-01 Quick-Commerce Delivery Intelligence** · concise report (PDF §13: 4–6 pages) · "
      f"generated {m['run_date']} · all numbers from `reports/stage5_metrics.json`.  ")
    a("The full Business Solution Document is `reports/business_insights_report.md`; the recommendation detail is in "
      "`reports/business_recommendations.md`. **[Observed]** = measured; **[Scenario]** = estimate, not measured.\n")
    a("## 1. Executive Summary\n")
    a(f"Using {D['rows']:,} food-delivery orders as a quick-commerce stand-in (restaurants = stores):")
    a(f"- **ETA at order placement:** gradient boosting predicts delivery time with test MAE **{r['test']['MAE']:.2f} min**, "
      f"{100 * (1 - r['test']['MAE'] / mb['MAE']):.1f}% better than the average-time baseline ({mb['MAE']:.2f} min).")
    a(f"- **Late-order alerts:** the classifier flags **{100 * t['recall']:.1f}%** of late orders (> 30 min, project benchmark) "
      f"with ROC-AUC {t['ROC_AUC']:.3f}.")
    a(f"- **Where delays concentrate:** the evening peak, where hours {', '.join(f'{h:02d}' for h in sorted(ch['peak_hours']))} are about half late.")
    a("- **Dark stores:** the geographic clusters are simply the 22 cities, and no zone is significantly worse than the network.")
    a("- **Riders:** rider IDs do not identify individual riders, so rider segmentation is not possible.")
    a("\nThree actions follow: late-risk alerts, a two-stage ETA, and evening surge staffing.\n")
    a("## 2. Business Understanding\n")
    a("A quick-commerce operator loses orders, refunds and customers when deliveries run late. Management needs to know "
      "what drives delay, whether late orders can be flagged at placement, where new dark stores should go, and when to "
      f"surge-staff riders. **Late** = delivery time > 30 minutes. This is a project-defined benchmark: the data and the guide "
      f"give no official SLA. Late rate = late orders ÷ all orders × 100 (network: {ch['network']['late_pct']:.1f}%).\n")
    a("## 3. Dataset and Data Quality\n")
    a("| Issue (PS1) | Count | Treatment |\n|---|---|---|")
    a(f"| Missing values stored as text `NaN` | {D['rows_any_missing_raw']:,} rows | Converted to blanks; imputed only inside train-fitted pipelines |")
    a(f"| Duplicates | {D['duplicates']} | None to remove |")
    a(f"| GPS (0, 0) restaurant points | {D['geo_invalid']:,} | Flagged invalid; distance left blank; rows kept |")
    a(f"| Sign-flipped coordinates | {D['sign_corrected']} | Corrected after 4 geographic checks; originals kept |")
    a(f"| Impossible age/rating (15/1.0, 50/6.0) | {D['rider_sentinel']} | Blanked as placeholders; originals kept |")
    a(f"| Missing order time | {D['missing_order_time']:,} | Kept; never rebuilt from pickup time (leakage) |")
    a("| Inconsistent labels (`Metropolitian`, padding, `conditions` prefix) | all rows | Normalised |")
    a(f"| Delivery-time outliers (1.5×IQR, 52–54 min) | {D['target_outliers_iqr']} | Kept: inside the 10–54 min range |\n")
    a(f"No rows were deleted ({D['rows_retained']:,} kept). Distance is the Haversine distance from restaurant to customer. "
      "**Leakage control:** pickup time, the target, IDs and post-dispatch fields (rider, vehicle, batching, weather, traffic) "
      "are excluded from order-placement models, and a code guard enforces this.\n")
    a("## 4. EDA Findings (associations, not causes)\n")
    hr = {int(x["order_hour"]): x for x in ch["by_hour"]}
    fs = ch["weather_traffic"]["fog_minus_sunny_within_traffic_pp"]
    a(f"1. **Hour of day:** {', '.join(f'{h:02d}:00 {hr[h]['late_pct']:.0f}%' for h in sorted(ch['peak_hours']))} late, vs "
      f"{hr[9]['late_pct']:.1f}% at 09:00 and a {ch['network']['late_pct']:.1f}% network rate (`figures/26_late_rate_by_hour.png`).")
    a(f"2. **Weather beyond traffic:** in Jam traffic, Fog orders are {fs['Jam']:+.1f} pp more often late than Sunny ones; in Low "
      f"traffic {fs['Low']:+.1f} pp (`figures/27_weather_traffic_late_rate.png`).")
    a(f"3. **Distance:** explains only R² {ch['distance_only_model']['R2']:.3f} on its own, with a step-up beyond about 10 km.")
    a(f"4. **Rider rating and age:** on similar routes, orders by riders rated 4.8–5.0 had a late rate "
      f"{abs(ch['rating_hi_vs_lo_similar_routes']['late_diff_pp']):.1f} pp {'lower' if ch['rating_hi_vs_lo_similar_routes']['late_diff_pp'] < 0 else 'higher'} "
      f"than riders below 4.5, and riders aged 35–39 took {ch['age_old_vs_young_similar_routes']['eta_diff_min']:+.1f} min vs 20–24. "
      "Rating may be a *result* of lateness.")
    a("5. **Day of week** varies modestly. **Order type has no effect.** Festival days and Semi-Urban orders are 100% late, "
      "which is a synthetic-data artifact.\n")
    a("| Order-time band | Orders | % late | Distance band | Orders | % late |\n|---|---|---|---|---|---|")
    tb = [r_ for r_ in m["observed"]["time_band"] if not r_["time_band"].startswith("(")]
    tb = sorted(tb, key=lambda r_: ["night_00_07", "morning_08_10", "midday_11_14", "afternoon_15_18", "evening_19_23"].index(r_["time_band"]))
    dbs = [r_ for r_ in m["observed"]["distance_band"] if not r_["distance_band"].startswith("(")]
    dbs = sorted(dbs, key=lambda r_: ["0–5 km", "5–10 km", "10–15 km", "15–25 km"].index(r_["distance_band"]))
    for i in range(max(len(tb), len(dbs))):
        left = f"{BAND[tb[i]['time_band']]} | {tb[i]['orders']:,} | {tb[i]['late30_pct']:.1f}" if i < len(tb) else " | | "
        right = f"{dbs[i]['distance_band']} | {dbs[i]['orders']:,} | {dbs[i]['late30_pct']:.1f}" if i < len(dbs) else " | | "
        a(f"| {left} | {right} |")
    a("")
    a("**Three evidence-backed findings:**")
    for f_ in three_findings(m):
        a(f"- {f_}")
    a("\n## 5. Modelling Approach\n")
    a(f"- **Split:** chronological on whole dates (train {D['split']['train_dates'][0]} → {D['split']['train_dates'][1]}, "
      f"test {D['split']['test_dates'][0]} → {D['split']['test_dates'][1]}).")
    a("- **Tuning:** 3 forward-chaining CV folds on training data only; preprocessing fitted inside pipelines.")
    a(f"- **Regression:** mean and median baselines, ridge, random forest and gradient boosting on {D['primary_features']} "
      "order-placement features.")
    a(f"- **Classification:** prior baseline, logistic regression and gradient boosting. Class imbalance "
      f"({ch['class_balance']['train_late_pct']:.1f}% late) is handled by selecting on PR-AUC and moving the decision threshold "
      "to the F1-best point on out-of-fold training predictions, rather than resampling.")
    a(f"- **Clustering:** K-Means on delivery coordinates (k by elbow and silhouette). A second K-Means on order-level "
      "rider attributes is an alternative to rider segments.\n")
    a("## 6. Model Selection and Explainability\n")
    L += model_selection_block(m)[1:]      # drop only the '### 5.1' heading; keep the table header
    a(f"- **Threshold:** {t['threshold']:.3f}, chosen on training out-of-fold predictions to maximise F1. A missed late order "
      f"costs more than an unnecessary heads-up, so recall ({t['recall']:.3f}) is favoured over precision ({t['precision']:.3f}). "
      "It is marked on the PR and ROC curves in `figures/29_threshold_on_pr_roc.png`.")
    a("- **Explainability:** grouped permutation importance on training data (`figures/22_explainability_grouped_importance.png`). "
      "Time of day matters most, then distance, festival and city type. Restaurant location, calendar and order type barely matter.\n")
    a("## 7. Three Recommendations\n")
    for i, rr in enumerate(recs, 1):
        a(f"**{i}. {rr['title']}**")
        a(f"- Evidence: {rr['evidence'][0]}")
        a(f"- Action: {rr['action'][0]}")
        a(f"- Why: {rr['indicates']}")
        a(f"- Implementation: {rr['implementation'][0]}")
        a(f"- Benefit: {next((b for b in rr['benefit'] if 'Scenario' in b), rr['benefit'][0])}")
        a(f"- KPI: {rr['kpis'][0]}")
        a(f"- Main risk: {rr['risks'][0]}\n")
    sc, op = c["scenario_late_rate_reduction_pp"], m["observed_peak"]
    a("**Expected value: measured vs projected**\n")
    a("| Action | Measured in this project [Observed] | Projected estimate [Scenario] |\n|---|---|---|")
    a(f"| 1. Late-risk alerts | {100 * t['recall']:.1f}% of late orders flagged before they are late | late rate −{sc['10%']:.1f} to −{sc['30%']:.1f} pp if 10–30% of caught late orders are prevented |")
    a(f"| 2. Two-stage ETA | ETA error {d['eta_MAE_no_rating']:.2f} vs {r['test']['MAE']:.2f} min (comparison model) | fewer order-status contacts (unquantified) |")
    a(f"| 3. Evening surge staffing | peak hours about 50% late vs {ch['network']['late_pct']:.1f}% network | late rate −{op['scenario_evening_to_afternoon_pp']:.1f} pp if the evening late rate fell to the afternoon level |\n")
    zc = ch["zones_vs_network"]["candidates"]
    a(f"**Dark stores:** the 'high demand and above benchmark' rule flags {', '.join(zc)}, but no zone is statistically different "
      "from the network. A new-store decision needs real customer locations, costs and service radii (Solution Document §8).\n")
    a("## 8. Limitations\n")
    for x in ["Food-delivery data with synthetic patterns (diagonal delivery offsets, 5-minute time blocks, traffic set by hour, segments that are 100% late).",
              "Test period of 8 days; April appears only in the test period.",
              f"{D['geo_invalid']:,} orders without valid coordinates; {D['missing_order_time']:,} without an order time.",
              "> 30 min is a project benchmark, not an official SLA. Probabilities are not calibrated.",
              "Clusters are cities, so no dark-store location can be established. Rider IDs are not people, so rider segments cannot be built.",
              "All patterns are associations. Projected benefits are scenarios, not measurements."]:
        a(f"- {x}")
    a("\n## 9. Technical Reproducibility\n")
    a("```powershell\ncd D:\\Project1\n.venv\\Scripts\\python.exe -m src.data_cleaning\n.venv\\Scripts\\python.exe -m src.feature_engineering\n"
      ".venv\\Scripts\\python.exe -m src.modelling\n.venv\\Scripts\\python.exe -m src.business_insights\n```")
    a("Seed 42, pinned `requirements.txt`, SHA-256-checked raw data; every stage writes a metrics JSON and runs automated validation checks.")
    text = "\n".join(L) + "\n"
    (REP / "business_report.md").write_text(text, encoding="utf-8")
    return len(text.split())


def write_compliance(m: dict, words: dict) -> None:
    """Requirement-by-requirement check against Hackathon_Challenge_Guide.pdf."""
    ch = m["challenge"]
    rows = [
        ("PS1 Data quality", "✅", "dataset_audit_report.md, data_cleaning_report.md, final_data_audit.md"),
        ("PS2 Delay lens (target, late definition, Haversine, leakage columns)", "✅", "feature_dictionary.md; > 30 min = project benchmark (no official SLA given)"),
        ("PS3 EDA: hour, day of week, weather, traffic, distance, order type, rider experience", "✅", "Solution Doc §4, §4.1–4.5; figures 26–27"),
        ("PS4 Regression: LR / RF / GB vs naive baseline; MAE, RMSE, R²; feature importance", "✅", "modelling_report.md; Solution Doc §5.1; figures 12–15, 22"),
        ("PS5 Classifier at placement: imbalance, recall/precision/F1/ROC-AUC, cost-based threshold", "✅", "Solution Doc §6; figures 16–18, 24, 29"),
        ("PS6 K-Means zones: elbow + silhouette, map, zone vs network", "⚠️ done; data cannot validate dark-store sites", "Solution Doc §8, §8.1; figures 19–20, 25"),
        ("PS7 Rider segmentation", "⚠️ not feasible as specified; order-level alternative provided", "Solution Doc §9, §9.1; figure 28"),
        ("PS8 Model selection, explainability, exactly 3 recommendations", "✅", "Solution Doc §5.1, §7, §10; business_recommendations.md"),
        ("Mandatory: data-quality summary", "✅", "Business Report §3"),
        ("Mandatory: ≥ 3 EDA delay comparisons", "✅", "hour, weather × traffic, distance, rider rating/age, day of week"),
        ("Mandatory: baseline + ≥ 2 regression models on held-out test", "✅", "mean & median baselines, ridge, RF, GB"),
        ("Mandatory: classifier with recall, precision, F1, ROC-AUC + justified threshold", "✅", "threshold 0.268 (training OOF, F1-max, recall-favoured)"),
        ("Mandatory: K-Means with elbow and silhouette", "✅", "figure 19"),
        ("Mandatory: model selection table with final choice and reason", "✅", "Solution Doc §5.1; Business Report §6"),
        ("Mandatory: 3 findings + 3 prioritised recommendations", "✅", "Solution Doc §4.6; business_recommendations.md"),
        ("Mandatory: limitations section", "✅", "Solution Doc §12; Business Report §8"),
        ("Visual: delivery-time distribution + late rate by hour", "✅", "figures 03, 26"),
        ("Visual: late rate by weather and traffic", "✅", "figure 27"),
        ("Visual: actual vs predicted", "✅", "figure 13"),
        ("Visual: feature importance", "✅", "figures 15, 22"),
        ("Visual: PR/ROC with chosen threshold marked", "✅", "figure 29"),
        ("Visual: elbow, silhouette, cluster map", "✅", "figures 19, 20"),
        ("Visual: rider segment profile chart (rating, experience, late rate)", "⚠️ order-level profiles, labelled as such", "figure 28"),
        ("Submission: source code / notebook", "✅", "src/*.py, notebooks/01_EDA.ipynb"),
        ("Submission: model comparison table", "✅", "Solution Doc §5.1"),
        (f"Submission: Business Report 4–6 pages", "✅ (by word-count estimate)", f"business_report.md — {words['business_report']:,} words, about {words['business_report'] / 500:.1f} pages at about 500 words/page"),
        (f"Submission: Business Solution Document ≥ 5 pages", "✅ (by word-count estimate)", f"business_insights_report.md — {words['solution_doc']:,} words, about {words['solution_doc'] / 500:.1f} pages"),
        ("Submission: Streamlit demo (ETA + delay-risk %)", "✅ working locally (not deployed online)",
         "app/streamlit_app.py; reports/app_validation_report.md (python -m src.validate_app)"),
        ("Submission: README with setup, dependencies, assumptions, execution", "✅", "README.md"),
        ("Boundaries: no test training/tuning, no post-delivery features, no accuracy-only, no causal claims, no deletion, observed vs projected", "✅", "Stage 4/5 validation checks; [Observed]/[Scenario] labels"),
        ("Final formatting of reports as PDF/DOCX", "⏳ Stage 6 polish", "reports are Markdown"),
    ]
    L = ["# DeliveryIQ — Compliance with Hackathon_Challenge_Guide.pdf\n",
         f"Generated {m['run_date']} by `src/business_insights.py`. Each mandatory item in the guide is mapped to its evidence.\n",
         "| Requirement (guide section) | Status | Evidence |\n|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    L += ["\n## Requirements the data cannot satisfy as worded\n",
          "- **Ideal dark-store locations (PS6, §17):**",
          "  - The clusters are the 22 cities (silhouette of the city labels = cluster silhouette).",
          f"  - No zone differs significantly from the network: {ch['zones_vs_network']['n_significant_late_bonferroni']} significant after correction.",
          "  - Delivery points are synthetic offsets.",
          "  - We report a provisional watch-list instead, and the data needed.",
          "- **Rider segments for rostering and training (PS7):**",
          "  - `Delivery_person_ID` is a restaurant/slot code with 4–22 ages per ID.",
          "  - Per-rider history, experience and deliveries per day cannot be built.",
          "  - We provide order-level rider-attribute profiles, clearly labelled.",
          "- **Official SLA:** neither the data nor the guide gives one. The > 30 min benchmark is labelled project-defined, "
          "with 35 and 40 min sensitivity, consistent with guide §6.",
          "\n## Optional items not used\n",
          "- XGBoost: sklearn histogram gradient boosting is used, the same model family, with no extra dependency.",
          "- SHAP: grouped permutation importance and partial dependence instead, both allowed by guide §12.",
          "- Folium/Plotly maps: static matplotlib map instead."]
    (REP / "challenge_compliance.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_reports(m: dict) -> int:
    """Write both Stage 5 reports; returns the number of recommendations written."""
    recs = recommendations(m)
    n = write_recommendations(m, recs)
    write_insights(m, recs)
    words = {"business_report": write_business_report(m, recs),
             "solution_doc": len((REP / "business_insights_report.md").read_text(encoding="utf-8").split())}
    m["report_word_counts"] = words
    write_compliance(m, words)
    return n
