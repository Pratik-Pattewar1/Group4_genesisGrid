"""DeliveryIQ — Stage 6 validation of the prediction service and the Streamlit demo.

Checks (writes reports/app_validation_report.md and reports/app_validation.json):
    1. Saved model files are byte-identical to the Stage 4 artifacts (SHA-256).
    2. Parity: 600 test-period orders, scored ONE AT A TIME through the app path (src/predict.py),
       give exactly the same ETA and risk as the training pipeline on the engineered dataset —
       including orders with unknown city type / festival / order time and invalid locations.
    3. Leakage: app inputs and model features contain no pickup, rider, vehicle, batching, weather,
       traffic, target, ID or audit columns.
    4. Invalid inputs are rejected with clear messages; unknown inputs produce warnings, not crashes.
    5. UI (Streamlit AppTest): every tab renders, realistic example orders produce ETA + risk that
       match the service, invalid UI inputs show an error, and the required disclaimers are visible.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.validate_app
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import EXCLUDED_BY_DECISION, chronological_split, get_model_inputs
from src.predict import OrderInput, OrderValidationError, load_models, predict_order

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
MODEL_HASHES = {   # recorded before Stage 6 started
    "delivery_zones_kmeans.joblib": "f19b59d06faae8ae4b0a31c79f14d86830d8351aa38dc73a2b0f75a7bac50f17",
    "eta_regressor.joblib": "e71a6f272eafb05a5da6d67a45e40f863cc3fd75b894fc0ec7355c391da6ec49",
    "late_classifier_gt30.joblib": "a81d16f00472aaff6b7d58e4afe3fbdcde4354ed465d8bd20e857df3d87fdc5f",
    "late_classifier_gt35.joblib": "6e2a2c195d25675e1469115a083b2339985103077765a0bdb21cc31a7a7e38d6",
    "late_classifier_gt40.joblib": "edd7d25bc36b40ef116fbf75a35fe7e276b2bddac36ee7d3e594df9c498fef30",
    "model_card.json": "5c155c668a32af3d2a9d0337f39bb0c5312c8bc257dfa7368c6a22c22947210f",
}
DATA_HASHES = {
    "data/raw/train.csv": "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1",
    "data/processed/cleaned_data.csv": "3f2f1ee971632876fe41d09da8fc35a618303b5f945cdd1816c63e6a2047bcfa",
    "data/processed/feature_engineered_data.csv": "86f90becbb69ce664d107bfee8d5d3542eadec4b2729770ed7d69e94d7c00421",
}
LEAKY = ["pickup", "rating", "rider", "delivery_person", "vehicle", "multiple", "weather", "traffic",
         "time_taken", "order_id", "dq_", "_raw", "city_code"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def order_from_clean(c) -> OrderInput:
    geo = bool(c.dq_geo_valid)
    return OrderInput(c.restaurant_lat if geo else None, c.restaurant_lon if geo else None,
                      dt.date.fromisoformat(c.order_date),
                      None if pd.isna(c.order_time) else dt.time.fromisoformat(c.order_time),
                      c.city_type if isinstance(c.city_type, str) else "Unknown", c.order_type,
                      c.festival if isinstance(c.festival, str) else "Unknown",
                      c.delivery_lat if geo else None, c.delivery_lon if geo else None)


def main() -> None:
    checks, scen = [], []
    check = lambda n, ok, d="": checks.append({"check": n, "passed": bool(ok), "detail": str(d)})

    # 1 ---- artifacts unchanged
    mh = {k: sha(ROOT / "models" / k) for k in MODEL_HASHES}
    check("saved model files unchanged (SHA-256 vs pre-Stage-6)", mh == MODEL_HASHES,
          [k for k in MODEL_HASHES if mh[k] != MODEL_HASHES[k]] or "all 6 identical")

    # 2 ---- parity, one order at a time
    m = load_models()
    fe = pd.read_csv(ROOT / "data/processed/feature_engineered_data.csv")
    cl = pd.read_csv(ROOT / "data/processed/cleaned_data.csv")
    _, te, _ = chronological_split(fe)
    t = fe.loc[te]
    hard = t[t.festival.isna() | t.city_type.isna() | t.order_hour.isna() | ~t.dq_geo_valid].index[:300]
    rows = list(hard) + list(t.drop(hard).sample(300, random_state=42).index)
    Xe, _ = get_model_inputs(fe.loc[rows], m["features"])
    pe, qe = m["eta"]["pipeline"].predict(Xe), m["late"]["pipeline"].predict_proba(Xe)[:, 1]
    preds = [predict_order(order_from_clean(cl.loc[i])) for i in rows]
    de = float(np.abs(pe - np.array([p.eta_min for p in preds])).max())
    dq = float(np.abs(qe - np.array([p.late_probability for p in preds])).max())
    check("parity: 600 orders scored one at a time match the training pipeline exactly",
          de == 0.0 and dq == 0.0, f"max |ΔETA| {de}, max |Δrisk| {dq}; {len(hard)} with unknown/invalid inputs")

    # 3 ---- leakage
    inputs = [f.name for f in fields(OrderInput)]
    bad_in = [f for f in inputs if any(k in f for k in LEAKY)]
    bad_feat = [f for f in m["features"] if any(k in f for k in LEAKY) or f in EXCLUDED_BY_DECISION]
    check("app inputs contain no post-placement or leakage fields", not bad_in, inputs)
    check("model features are the 13 order-placement features only", not bad_feat and len(m["features"]) == 13, m["features"])
    app_src = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    widget_keys = [k for k in ["pickup", "rating\"", "weather\"", "traffic\"", "vehicle", "multiple"] if f'key="{k}' in app_src]
    check("no leakage field exists as an app widget", not widget_keys, widget_keys or "none")

    # 4 ---- invalid & unknown inputs
    base = dict(restaurant_lat=12.913041, restaurant_lon=77.683237, order_date=dt.date(2022, 3, 16),
                order_time=dt.time(19, 30), city_type="Metropolitan", order_type="Meal", festival="No")
    bad_cases = {
        "store outside India": dict(base, restaurant_lat=0.0, restaurant_lon=0.0, distance_km=5.0),
        "customer outside India": dict(base, customer_lat=51.5, customer_lon=-0.1),
        "only one store coordinate": dict(base, restaurant_lon=None, distance_km=5.0),
        "only one customer coordinate": dict(base, customer_lat=12.95),
        "distance 0 km": dict(base, distance_km=0.0),
        "distance 30 km (beyond data limit)": dict(base, distance_km=30.0),
        "customer 60 km away": dict(base, customer_lat=13.45, customer_lon=77.68),
        "invalid order type": dict(base, order_type="Pizza", distance_km=5.0),
        "invalid city type": dict(base, city_type="Rural", distance_km=5.0),
    }
    rejected = {}
    for name, kw in bad_cases.items():
        try:
            predict_order(OrderInput(**kw)); rejected[name] = "NOT REJECTED"
        except OrderValidationError as e:
            rejected[name] = str(e)
    check("invalid inputs are rejected with a clear message", all(v != "NOT REJECTED" for v in rejected.values()),
          f"{sum(v != 'NOT REJECTED' for v in rejected.values())}/{len(rejected)} rejected")
    unknown = predict_order(OrderInput(None, None, dt.date(2022, 3, 16), None, "Unknown", "Snack", "Unknown"))
    check("all-unknown order is scored with warnings, not a crash",
          np.isfinite(unknown.eta_min) and 0 <= unknown.late_probability <= 1 and len(unknown.warnings) >= 4,
          f"ETA {unknown.eta_min:.1f}, risk {unknown.late_probability:.3f}, {len(unknown.warnings)} warnings")
    night = predict_order(OrderInput(**dict(base, order_time=dt.time(3, 0), distance_km=5.0)))
    check("hours with no training data (01–07) produce an extrapolation warning",
          any("extrapolation" in w for w in night.warnings))

    # 5 ---- realistic scenarios through the service (business sense, not hard-coded values)
    S = {
        "Bengaluru, weekday 09:15, 3 km snack": dict(base, order_time=dt.time(9, 15), distance_km=3.0, order_type="Snack", city_type="Urban", order_date=dt.date(2022, 3, 15)),
        "Bengaluru, weekday 20:00, 16 km meal": dict(base, order_time=dt.time(20, 0), distance_km=16.0, order_date=dt.date(2022, 3, 16)),
        "Mumbai, Fri 16:30, 8 km drinks": dict(base, restaurant_lat=19.176269, restaurant_lon=72.836721, order_time=dt.time(16, 30), distance_km=8.0, order_type="Drinks", order_date=dt.date(2022, 3, 18)),
        "Jaipur, Sat 23:10, 4 km buffet": dict(base, restaurant_lat=26.902328, restaurant_lon=75.794257, order_time=dt.time(23, 10), distance_km=4.0, order_type="Buffet", city_type="Urban", order_date=dt.date(2022, 3, 19)),
        "Chennai, festival evening 19:45, 12 km": dict(base, restaurant_lat=13.022394, restaurant_lon=80.242439, order_time=dt.time(19, 45), distance_km=12.0, festival="Yes", order_date=dt.date(2022, 3, 17)),
        "Hyderabad, customer coordinates 21:00": dict(base, restaurant_lat=17.431668, restaurant_lon=78.408321, customer_lat=17.481668, customer_lon=78.458321, order_time=dt.time(21, 0), order_date=dt.date(2022, 3, 23)),
    }
    for name, kw in S.items():
        p = predict_order(OrderInput(**kw))
        scen.append({"scenario": name, "eta_min": round(p.eta_min, 2), "risk_pct": round(100 * p.late_probability, 1),
                     "risk_level": p.risk_level, "distance_km": None if p.distance_km is None else round(p.distance_km, 2),
                     "warnings": len(p.warnings), "top_driver": p.drivers[0]["factor"]})
    sd = {x["scenario"]: x for x in scen}
    check("business sense: evening 16 km order has a longer ETA and higher risk than morning 3 km order",
          sd["Bengaluru, weekday 20:00, 16 km meal"]["eta_min"] > sd["Bengaluru, weekday 09:15, 3 km snack"]["eta_min"]
          and sd["Bengaluru, weekday 20:00, 16 km meal"]["risk_pct"] > sd["Bengaluru, weekday 09:15, 3 km snack"]["risk_pct"])
    check("all scenario predictions are finite, ETA within 5–60 min, risk within 0–100%",
          all(np.isfinite(x["eta_min"]) and 5 <= x["eta_min"] <= 60 and 0 <= x["risk_pct"] <= 100 for x in scen))

    # 6 ---- UI via Streamlit AppTest
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=240).run()
    check("app loads without exceptions; 4 tabs render", not at.exception and len(at.tabs) == 4,
          [t.label for t in at.tabs])
    ui = []
    for preset in ["Weekday morning, short trip", "Evening peak, long trip", "Afternoon, medium trip", "Late night, short trip"]:
        at.selectbox(key="preset").set_value(preset).run()
        at.button(key="predict").click().run()
        mets = {mm.label: mm.value for mm in at.metric}
        st_ = at.session_state
        feats = st_["last_prediction"].features                     # exact store coordinates the UI sent
        exp = predict_order(OrderInput(feats["restaurant_lat"], feats["restaurant_lon"], st_["order_date"], st_["order_time"],
                                       st_["city_type"], st_["order_type"], st_["festival"], distance_km=st_["distance"]))
        ok = (not at.exception and mets.get("Predicted delivery time") == f"{exp.eta_min:.1f} min"
              and mets.get("Delay risk score (> 30 min)") == f"{100 * exp.late_probability:.0f}%"
              and mets.get("Risk level") == exp.risk_level)
        ui.append({"preset": preset, "ui_eta": mets.get("Predicted delivery time"), "ui_risk": mets.get("Delay risk score (> 30 min)"),
                   "ui_level": mets.get("Risk level"), "matches_service": ok})
    check("UI: 4 example orders show ETA, risk % and level that match the prediction service",
          all(u["matches_service"] for u in ui), [(u["preset"], u["ui_eta"], u["ui_risk"], u["ui_level"]) for u in ui])
    at.selectbox(key="store").set_value("Custom coordinates…").run()
    at.number_input(key="rlat").set_value(0.0).run()
    at.number_input(key="rlon").set_value(0.0).run()
    at.button(key="predict").click().run()
    check("UI: invalid store coordinates show an error message, not a crash",
          not at.exception and any("cannot be scored" in e.value for e in at.error), [e.value[:90] for e in at.error])
    at.selectbox(key="store").set_value(at.selectbox(key="store").options[0]).run()
    at.checkbox(key="time_unknown").check().run()
    at.button(key="predict").click().run()
    check("UI: unknown order time is handled with a warning",
          not at.exception and any("Order time unknown" in w.value for w in at.warning))
    text = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.warning) + " ".join(x.value for x in at.caption)
    check("UI states '> 30 min' is a project benchmark, not an official SLA", "not an official SLA" in text)
    check("UI states risk scores are uncalibrated", "uncalibrated" in text.lower())
    check("UI shows the three approved recommendations", sum(1 for e in at.expander if e.label.startswith("Priority ")) == 3,
          [e.label[:60] for e in at.expander if e.label.startswith("Priority ")])

    # 7 ---- datasets untouched
    dh = {k: sha(ROOT / k) for k in DATA_HASHES}
    check("raw, cleaned and engineered datasets unchanged", dh == DATA_HASHES)

    passed = sum(c["passed"] for c in checks)
    out = {"run_date": dt.date.today().isoformat(), "checks": checks, "passed": passed, "total": len(checks),
           "scenarios": scen, "ui_examples": ui, "rejections": rejected}
    (REP / "app_validation.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    L = ["# DeliveryIQ — Stage 6 App Validation Report\n",
         f"Generated {out['run_date']} by `src/validate_app.py` · **{passed}/{len(checks)} checks passed**\n",
         "| # | Check | Result | Detail |\n|---|---|---|---|"]
    L += [f"| {i} | {c['check']} | {'✅ pass' if c['passed'] else '❌ FAIL'} | {str(c['detail']).replace('|', '/')[:220]} |" for i, c in enumerate(checks, 1)]
    L += ["\n## Realistic test orders (prediction service)\n", "| Scenario | ETA (min) | Risk score | Level | Distance km | Warnings | Biggest driver |\n|---|---|---|---|---|---|---|"]
    L += [f"| {x['scenario']} | {x['eta_min']:.1f} | {x['risk_pct']:.1f}% | {x['risk_level']} | {x['distance_km']} | {x['warnings']} | {x['top_driver']} |" for x in scen]
    L += ["\n## Example orders through the Streamlit UI\n", "| Preset | ETA shown | Risk shown | Level | Matches service |\n|---|---|---|---|---|"]
    L += [f"| {u['preset']} | {u['ui_eta']} | {u['ui_risk']} | {u['ui_level']} | {'✅' if u['matches_service'] else '❌'} |" for u in ui]
    L += ["\n## Rejected invalid inputs\n", "| Case | Message |\n|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rejected.items()]
    L += ["\nRisk scores are uncalibrated model scores; late = > 30 min is a project benchmark, not an official SLA."]
    (REP / "app_validation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"checks: {passed}/{len(checks)} passed")
    for c in checks:
        print(("  PASS " if c["passed"] else "  FAIL ") + c["check"] + (f"  [{str(c['detail'])[:160]}]" if c["detail"] else ""))
    if passed != len(checks):
        raise SystemExit("validation failed")


if __name__ == "__main__":
    main()
