"""DeliveryIQ — Streamlit demo (Stage 6).

Enter an order as known at placement time → predicted ETA (minutes) and late-delivery risk
(> 30 min, project benchmark — not an official SLA). Uses the saved Stage 4 primary models
read-only, through src/predict.py (same feature logic as training).

Run from the project root:
    .venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.insights_report import model_selection_block, recommendations, three_findings
from src.predict import (CITY_TYPES, FESTIVAL, ORDER_TYPES, UNKNOWN, OrderInput, OrderValidationError,
                         load_models, predict_order, reference)

FIG = ROOT / "figures"
REP = ROOT / "reports"
# City codes come from the data's rider-ID prefix; names are the evident expansions (for display only).
CITY_NAMES = {"AGR": "Agra", "ALH": "Allahabad", "AURG": "Aurangabad", "BANG": "Bengaluru", "BHP": "Bhopal",
              "CHEN": "Chennai", "COIMB": "Coimbatore", "DEH": "Dehradun", "GOA": "Goa", "HYD": "Hyderabad",
              "INDO": "Indore", "JAP": "Jaipur", "KNP": "Kanpur", "KOC": "Kochi", "KOL": "Kolkata", "LUDH": "Ludhiana",
              "MUM": "Mumbai", "MYS": "Mysuru", "PUNE": "Pune", "RANCHI": "Ranchi", "SUR": "Surat", "VAD": "Vadodara"}
PRESETS = {
    "Weekday morning, short trip": dict(hour=9, minute=15, dist=3.0, order_type="Snack", city_type="Urban", festival="No", weekday=1),
    "Evening peak, long trip": dict(hour=20, minute=0, dist=16.0, order_type="Meal", city_type="Metropolitan", festival="No", weekday=2),
    "Afternoon, medium trip": dict(hour=16, minute=30, dist=8.0, order_type="Drinks", city_type="Metropolitan", festival="No", weekday=4),
    "Late night, short trip": dict(hour=23, minute=10, dist=4.0, order_type="Buffet", city_type="Urban", festival="No", weekday=5),
}
RISK_STYLE = {"Low": ("🟢", "success"), "Elevated": ("🟠", "warning"), "High": ("🔴", "error")}

st.set_page_config(page_title="DeliveryIQ — Delivery Intelligence", page_icon="🛵", layout="wide")


# ----------------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading saved models…")
def _models():
    return load_models()


@st.cache_resource(show_spinner="Preparing reference data (training period only)…")
def _reference():
    return reference()


@st.cache_data
def _stage5() -> dict:
    return json.loads((REP / "stage5_metrics.json").read_text(encoding="utf-8"))


def _figure(name: str, caption: str) -> None:
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chart `{name}` not found — run `python -m src.business_insights`.")


DEFAULTS = {"city": "BANG", "loc_mode": "Delivery distance (km)", "dist_unknown": False, "distance": 8.0,
            "order_date": dt.date(2022, 3, 16), "time_unknown": False, "order_time": dt.time(19, 30),
            "order_type": "Snack", "city_type": "Metropolitan", "festival": "No"}


def _init_state() -> None:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _apply_preset() -> None:
    p = PRESETS.get(st.session_state.get("preset"))
    if not p:
        return
    base = dt.date(2022, 3, 14)                              # a Monday inside the data period
    st.session_state.update({
        "order_time": dt.time(p["hour"], p["minute"]), "time_unknown": False, "loc_mode": "Delivery distance (km)",
        "dist_unknown": False, "distance": p["dist"], "order_type": p["order_type"], "city_type": p["city_type"], "festival": p["festival"],
        "order_date": base + dt.timedelta(days=p["weekday"]),
    })


def _drivers_chart(drivers: list[dict]):
    d = pd.DataFrame(drivers).iloc[::-1]
    ink = "#8a8984"                                   # mid-grey text: readable on light and dark themes
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.patch.set_alpha(0); ax.set_facecolor("none")  # transparent, so the app theme shows through
    colors = ["#eb6834" if v > 0 else "#2a78d6" for v in d["eta_effect_min"]]
    ax.barh(d["factor"], d["eta_effect_min"], color=colors, height=0.55)
    ax.axvline(0, color=ink, lw=1)
    ax.set_xlabel("minutes added (+) or saved (−) vs a typical order", color=ink)
    ax.tick_params(colors=ink)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink)
    ax.grid(axis="x", color=ink, alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------- layout
models, ref, s5 = _models(), _reference(), _stage5()
card = models["card"]
eta_test = card["eta_regressor"]["test"]
late_test = card["late_classifiers"]["30"]["test"]

st.title("🛵 DeliveryIQ — Delivery Intelligence")
st.caption("AI-powered delivery ETA and late-order risk at the moment an order is placed · KLU + PATH3 ML-01 · "
           "quick-commerce style demo on a food-delivery dataset (restaurants stand in for dark stores)")

with st.sidebar:
    st.header("About this demo")
    st.markdown(
        f"- **ETA model:** gradient boosting, test MAE **{eta_test['MAE']:.2f} min**, R² {eta_test['R2']:.3f}\n"
        f"- **Late-risk model:** gradient boosting, ROC-AUC **{late_test['ROC_AUC']:.3f}**, recall {late_test['recall']:.3f}, "
        f"precision {late_test['precision']:.3f}\n"
        f"- Trained on orders from {card['split']['train_dates'][0]} → {card['split']['train_dates'][1]}; "
        f"tested on {card['split']['test_dates'][0]} → {card['split']['test_dates'][1]}")
    st.warning("**Late = delivery > 30 minutes.** This is a *project-defined benchmark*, **not an official SLA**.\n\n"
               "Risk percentages are **uncalibrated model scores**: use them to rank orders, not as exact probabilities.")
    st.markdown("**Not used as inputs** (unknown when an order is placed, or leakage risk): pickup time, rider age, "
                "rider rating, vehicle type and condition, multiple deliveries, weather, traffic.")

tab_pred, tab_ins, tab_perf, tab_dq = st.tabs(["🔮 Predict an order", "📊 Business insights",
                                                "📈 Model performance", "🧹 Data quality & limitations"])

# ============================================================================ PREDICT
with tab_pred:
    _init_state()
    st.selectbox("Quick example (optional)", ["Custom"] + list(PRESETS), key="preset", on_change=_apply_preset,
                 help="Fills the form with a realistic example; you can still edit every field.")
    stores = ref["stores"]
    if True:                                                  # plain widgets (a form would hide conditional fields)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1 · Store")
            codes = sorted(stores["city_code"].unique())
            if st.session_state["city"] not in codes:
                st.session_state["city"] = codes[0]
            city = st.selectbox("City", codes,
                                format_func=lambda c: f"{CITY_NAMES.get(c, c)} ({c})", key="city",
                                help="Used only to list store locations; the city itself is not a model input.")
            cs = stores[stores["city_code"] == city].reset_index(drop=True)
            labels = [f"Store {i + 1} · {r.restaurant_lat:.4f}, {r.restaurant_lon:.4f} · {r.orders:,} past orders" for i, r in cs.iterrows()]
            store_choice = st.selectbox("Store location", labels + ["Custom coordinates…", "Unknown"], key="store")
            if store_choice == "Custom coordinates…":
                rlat = st.number_input("Store latitude", value=float(cs.restaurant_lat.iloc[0]), format="%.6f", key="rlat")
                rlon = st.number_input("Store longitude", value=float(cs.restaurant_lon.iloc[0]), format="%.6f", key="rlon")
            elif store_choice == "Unknown":
                rlat = rlon = None
            else:
                r = cs.iloc[labels.index(store_choice)]
                rlat, rlon = float(r.restaurant_lat), float(r.restaurant_lon)
        with c2:
            st.subheader("2 · Customer")
            loc_mode = st.radio("Customer location as", ["Delivery distance (km)", "Customer coordinates"], key="loc_mode")
            clat = clon = dist = None
            if loc_mode == "Delivery distance (km)":
                dist_unknown = st.checkbox("Distance unknown", key="dist_unknown")
                dist = None if dist_unknown else st.slider("Delivery distance (km)", 0.5, 25.0, step=0.5, key="distance",
                                                           help="Straight-line (Haversine) distance, as used in training.")
            else:
                base_lat, base_lon = (rlat, rlon) if rlat is not None else (float(cs.restaurant_lat.iloc[0]), float(cs.restaurant_lon.iloc[0]))
                clat = st.number_input("Customer latitude", value=round(base_lat + 0.05, 6), format="%.6f", key="clat")
                clon = st.number_input("Customer longitude", value=round(base_lon + 0.05, 6), format="%.6f", key="clon")
        with c3:
            st.subheader("3 · Order")
            order_date = st.date_input("Order date", key="order_date")
            time_unknown = st.checkbox("Order time unknown", key="time_unknown")
            order_time = None if time_unknown else st.time_input("Order time", key="order_time", step=300)
            order_type = st.selectbox("Order type", ORDER_TYPES, key="order_type")
            city_type = st.selectbox("City type", CITY_TYPES + [UNKNOWN], key="city_type")
            festival = st.selectbox("Festival day?", FESTIVAL + [UNKNOWN], key="festival")
        submitted = st.button("Predict ETA & delay risk", type="primary", width="stretch", key="predict")

    if submitted:
        order = OrderInput(restaurant_lat=rlat, restaurant_lon=rlon, order_date=order_date, order_time=order_time,
                           city_type=city_type, order_type=order_type, festival=festival,
                           customer_lat=clat, customer_lon=clon, distance_km=dist)
        try:
            pred = predict_order(order)
        except OrderValidationError as e:
            st.error(f"⚠️ This order cannot be scored: {e}")
        except Exception as e:                                # never show a raw traceback to the user
            st.error(f"Unexpected error while predicting ({type(e).__name__}). Please check the inputs.")
        else:
            st.session_state["last_prediction"] = pred
            icon, kind = RISK_STYLE[pred.risk_level]
            st.markdown(f"### ETA: **{pred.eta_min:.0f} min** · **{100 * pred.late_probability:.0f}% risk of delay** {icon}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted delivery time", f"{pred.eta_min:.1f} min",
                      help=f"Typical error on unseen days: ±{eta_test['MAE']:.1f} min (test MAE).")
            m2.metric("Likely window", f"{max(10, pred.eta_min - eta_test['MAE']):.0f}–{pred.eta_min + eta_test['MAE']:.0f} min")
            m3.metric("Delay risk score (> 30 min)", f"{100 * pred.late_probability:.0f}%", help="Uncalibrated model score.")
            m4.metric("Risk level", pred.risk_level)
            msg = {
                "High": "Among the riskiest 10% of orders seen in training. **Suggested (Recommendation 1, tier A):** "
                        "prioritise kitchen and dispatch, and send a proactive ETA update to the customer.",
                "Elevated": f"Above the alert threshold ({100 * ref['decision_threshold']:.1f}%). **Suggested (tier B):** "
                            "low-cost action only, such as a padded promise or an automated update. Many such alerts are false alarms.",
                "Low": f"Below the alert threshold ({100 * ref['decision_threshold']:.1f}%). No intervention suggested.",
            }[pred.risk_level]
            getattr(st, kind)(msg)
            st.caption("Late = > 30 min (project benchmark, not an official SLA). The risk score is uncalibrated: "
                       f"in testing, {100 * late_test['precision']:.0f}% of flagged orders were actually late and "
                       f"{100 * late_test['recall']:.0f}% of late orders were flagged.")
            for w in pred.warnings:
                st.warning(w)
            left, right = st.columns([1.2, 1])
            with left:
                st.markdown("**Why this prediction?** How much each factor adds to or saves from the ETA, compared "
                            "with a typical training order. This shows model behaviour, not proven causes.")
                st.pyplot(_drivers_chart(pred.drivers), width="stretch")
            with right:
                dd = pd.DataFrame(pred.drivers).rename(columns={"factor": "Factor", "eta_effect_min": "ETA effect (min)",
                                                                 "risk_effect_pp": "Risk effect (pp)"})
                st.dataframe(dd.style.format({"ETA effect (min)": "{:+.1f}", "Risk effect (pp)": "{:+.1f}"}),
                             hide_index=True, width="stretch")
                with st.expander("Model inputs sent (13 order-placement features)"):
                    st.json({k: (None if v != v else v) for k, v in pred.features.items()})

# ============================================================================ INSIGHTS
with tab_ins:
    st.header("Three evidence-backed findings")
    for f in three_findings(s5):
        st.markdown(f"- {f}")
    st.header("Three prioritised recommendations")
    st.caption("[Observed] = measured in this project · [Scenario] = hypothetical estimate, not a measured result.")
    for i, r in enumerate(recommendations(s5), 1):
        with st.expander(f"Priority {i} — {r['title']}", expanded=(i == 1)):
            st.markdown(f"**Problem:** {r['problem']}")
            st.markdown("**Evidence:**\n" + "\n".join(f"- {e}" for e in r["evidence"]))
            st.markdown("**Action:**\n" + "\n".join(f"- {a}" for a in r["action"]))
            st.markdown("**Expected benefit:**\n" + "\n".join(f"- {b}" for b in r["benefit"]))
            st.markdown("**KPIs:** " + "; ".join(r["kpis"]))
            st.markdown("**Risks:**\n" + "\n".join(f"- {x}" for x in r["risks"]))
    c1, c2 = st.columns(2)
    with c1:
        _figure("26_late_rate_by_hour.png", "Late rate by hour of order — the 19:00–21:59 surge window")
    with c2:
        _figure("27_weather_traffic_late_rate.png", "Late rate by weather × traffic")
    g = s5["geography"]
    st.info(f"**Dark stores:** the 22 K-Means zones are the 22 cities (silhouette {g['silhouette_of_city_labels']:.3f} = city separation), "
            f"and late rates do not differ significantly between zones (p = {g['zone_late_chi2_p']:.3f}). The data cannot justify "
            "specific new dark-store locations. **Rider segments:** rider IDs are not individual riders, so only order-level "
            "rider-attribute profiles are shown in the report.")

# ============================================================================ PERFORMANCE
with tab_perf:
    st.header("Model performance (held-out test period)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ETA MAE", f"{eta_test['MAE']:.2f} min", f"{100 * (1 - eta_test['MAE'] / s5['challenge']['mean_baseline']['MAE']):.1f}% better than mean baseline")
    k2.metric("ETA R²", f"{eta_test['R2']:.3f}")
    k3.metric("Late-risk ROC-AUC", f"{late_test['ROC_AUC']:.3f}")
    k4.metric("Late-risk recall / precision", f"{late_test['recall']:.2f} / {late_test['precision']:.2f}")
    st.markdown("\n".join(model_selection_block(s5)))
    sens = pd.DataFrame([{"Benchmark": f"> {k} min", "Late share": v["test"]["prevalence"], "Threshold": v["decision_threshold"],
                          "Precision": v["test"]["precision"], "Recall": v["test"]["recall"], "F1": v["test"]["F1"],
                          "ROC-AUC": v["test"]["ROC_AUC"], "PR-AUC": v["test"]["PR_AUC"]} for k, v in card["late_classifiers"].items()])
    st.subheader("Late-benchmark sensitivity")
    st.dataframe(sens.style.format({c: "{:.3f}" for c in sens.columns if c != "Benchmark"} | {"Late share": "{:.1%}"}),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        _figure("29_threshold_on_pr_roc.png", "Precision–recall and ROC with the chosen threshold marked")
        _figure("13_eta_pred_vs_actual.png", "ETA: predicted vs actual (test)")
    with c2:
        _figure("22_explainability_grouped_importance.png", "What the models rely on (training data)")
        _figure("24_late_risk_deciles.png", "Observed late rate by predicted-risk decile (test)")

# ============================================================================ DATA QUALITY
with tab_dq:
    d = s5["data"]
    st.header("Data quality")
    st.dataframe(pd.DataFrame([
        ["Orders", f"{d['rows']:,}", f"All {d['rows_retained']:,} retained; no rows deleted"],
        ["Rows with missing values (stored as text 'NaN')", f"{d['rows_any_missing_raw']:,}", "Converted to blanks; filled only inside train-fitted pipelines"],
        ["Duplicate rows", f"{d['duplicates']}", "None"],
        ["Restaurant location (0, 0)", f"{d['geo_invalid']:,}", "Flagged invalid; never estimated"],
        ["Sign-flipped coordinates", f"{d['sign_corrected']}", "Corrected after 4 geographic checks"],
        ["Placeholder rider records", f"{d['rider_sentinel']}", "Age and rating blanked"],
        ["Missing order time", f"{d['missing_order_time']:,}", "Kept; never rebuilt from pickup time"],
    ], columns=["Issue", "Count", "Treatment"]), hide_index=True, width="stretch")
    st.header("Limitations")
    st.markdown(
        "- **Food-delivery data, not grocery quick-commerce;** it contains synthetic patterns (diagonal delivery offsets, "
        "5-minute time blocks, traffic set by hour, segments that are 100% late).\n"
        f"- **Short evaluation:** {card['split']['test_dates'][0]} → {card['split']['test_dates'][1]} (8 days). April appears only in the test period.\n"
        "- **> 30 min is a project benchmark,** not an official SLA. **Risk scores are not calibrated.**\n"
        "- **Hours 01:00–07:59 have no training orders,** so predictions there are extrapolations.\n"
        "- **The clusters are cities,** so no dark-store location can be justified. Rider IDs are not people, so no rider segments.\n"
        "- **All patterns are associations, not causes.** Projected benefits are scenarios, not measured results.")
    st.caption("Full details: reports/business_report.md, reports/business_insights_report.md, reports/challenge_compliance.md.")
