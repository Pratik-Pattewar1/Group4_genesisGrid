"""DeliveryIQ — Stage 5 business insights, explainability and recommendation evidence.

Reads the Stage 1–4 artifacts (reports/*.json, models/*.joblib, processed datasets), verifies the
Stage 4 numbers by recomputing them from the saved models, adds lightweight explainability on the
TRAINING split only, investigates geography and rider-segmentation feasibility, and writes:

    reports/stage5_metrics.json
    reports/business_insights_report.md
    reports/business_recommendations.md
    figures/22–25_*.png

Rules: no retraining, no tuning, no test-set-driven choices, primary (order-placement) features only
for explainability, every reported number computed here or read from earlier stage metrics.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.business_insights
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, roc_auc_score, silhouette_score

from src.clustering import project_km
from src.evaluation import classification_report, regression_report
from src.feature_engineering import PRIMARY_FEATURES, TIME_BANDS, assert_no_leakage, chronological_split, get_model_inputs

ROOT = Path(__file__).resolve().parents[1]
REP, FIG, MOD = ROOT / "reports", ROOT / "figures", ROOT / "models"
RAW = ROOT / "data" / "raw" / "train.csv"
CLEAN = ROOT / "data" / "processed" / "cleaned_data.csv"
FE = ROOT / "data" / "processed" / "feature_engineered_data.csv"
OUT_JSON = REP / "stage5_metrics.json"
SEED = 42
EXPECTED_HASHES = {
    "train.csv": "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1",
    "cleaned_data.csv": "3f2f1ee971632876fe41d09da8fc35a618303b5f945cdd1816c63e6a2047bcfa",
    "feature_engineered_data.csv": "86f90becbb69ce664d107bfee8d5d3542eadec4b2729770ed7d69e94d7c00421",
}
# Values quoted in the owner's Stage 5 brief — verified against recomputation, never assumed.
BRIEF = {"eta_MAE": 6.19, "eta_RMSE": 7.73, "eta_R2": 0.310, "baseline_MAE": 7.47, "improvement_pct": 17.1,
         "ROC_AUC": 0.791, "PR_AUC": 0.608, "precision": 0.467, "recall": 0.813, "F1": 0.593, "threshold": 0.268,
         "dispatch_MAE": 3.50, "dispatch_ROC_AUC": 0.98, "k": 22, "silhouette": 0.948}
FEATURE_GROUPS = {   # business-language groups of the 13 primary features
    "Time of day": ["order_hour", "order_hour_sin", "order_hour_cos", "order_time_band"],
    "Delivery distance": ["distance_km"],
    "Festival day": ["festival"],
    "City type": ["city_type"],
    "Restaurant location": ["restaurant_lat", "restaurant_lon"],
    "Calendar (day of week / month day / weekend)": ["order_day_of_week", "order_day_of_month", "order_is_weekend"],
    "Order type": ["order_type"],
}
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#8a8984",
                                                   "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlecolor": INK, "font.size": 10})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_json(name: str) -> dict:
    return json.loads((REP / name).read_text(encoding="utf-8"))


def time_band(h: int) -> str:
    return next(n for lo, hi, n in TIME_BANDS if lo <= h < hi)


def grouped_permutation(predict, X: pd.DataFrame, y, score, higher_is_better: bool, n_repeats: int = 5) -> dict:
    """Permute each feature GROUP jointly (same row shuffle for all its columns) and record the score loss.
    Joint permutation avoids splitting importance across collinear features (e.g. the four hour features)."""
    base = score(y, predict(X))
    rng = np.random.default_rng(SEED)
    out = {}
    for g, cols in FEATURE_GROUPS.items():
        losses = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[idx]
            s = score(y, predict(Xp))
            losses.append((base - s) if higher_is_better else (s - base))
        out[g] = {"mean": float(np.mean(losses)), "sd": float(np.std(losses))}
    return {"baseline_score": float(base), "groups": dict(sorted(out.items(), key=lambda kv: -kv[1]["mean"]))}


def partial_dependence(predict, X: pd.DataFrame, setter, values) -> list[dict]:
    """Average model prediction when every row is set to each value (model behaviour, not causation)."""
    res = []
    for v in values:
        Xv = setter(X.copy(), v)
        res.append({"value": v, "mean_prediction": float(np.mean(predict(Xv)))})
    return res


def save(fig, name: str) -> str:
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    return f"figures/{name}"


def main() -> None:
    hashes_before = {p.name: sha(p) for p in (RAW, CLEAN, FE)}
    M: dict = {"run_date": date.today().isoformat(), "checks": []}
    check = lambda n, ok, d="": M["checks"].append({"check": n, "passed": bool(ok), "detail": str(d)})
    check("input datasets match the recorded SHA-256 hashes", hashes_before == EXPECTED_HASHES)
    challenge_pdf = [p.name for p in ROOT.rglob("*.pdf") if ".venv" not in p.parts]
    M["challenge_pdf_found_in_project"] = challenge_pdf

    audit, cleaning = load_json("audit_metrics.json"), load_json("cleaning_metrics.json")
    mts, fem, mm = load_json("missing_timestamp_metrics.json"), load_json("feature_engineering_metrics.json"), load_json("modelling_metrics.json")
    card = json.loads((MOD / "model_card.json").read_text(encoding="utf-8"))

    fe = pd.read_csv(FE)
    tr_idx, te_idx, cut = chronological_split(fe)
    train, test = fe.loc[tr_idx].reset_index(drop=True), fe.loc[te_idx].reset_index(drop=True)
    eta = joblib.load(MOD / "eta_regressor.joblib")
    late = {m: joblib.load(MOD / f"late_classifier_gt{m}.joblib") for m in (30, 35, 40)}
    feats = eta["features"]
    assert_no_leakage(feats)
    check("saved models use only primary order-placement features (strict leakage guard)",
          set(feats) <= set(PRIMARY_FEATURES) and all(late[m]["features"] == feats for m in late), f"{len(feats)} features")

    # ======================= 1. verify Stage 4 by recomputation =======================
    Xte, yte = get_model_inputs(test, feats)
    Xtr, ytr = get_model_inputs(train, feats)
    eta_pred = eta["pipeline"].predict(Xte)
    reg = regression_report(yte, eta_pred)
    base = mm["regression"]["families_final_config"]["baseline_median"]["test"]["all"]
    base_re = regression_report(yte, np.full(len(yte), np.median(ytr)))
    p30 = late[30]["pipeline"].predict_proba(Xte)[:, 1]
    c30 = classification_report((yte > 30).astype(int), p30, late[30]["decision_threshold"])
    sens = {}
    for m in (30, 35, 40):
        p = late[m]["pipeline"].predict_proba(Xte)[:, 1]
        sens[m] = classification_report((yte > m).astype(int), p, late[m]["decision_threshold"])
    sel = mm["regression"]["families_final_config"][mm["regression"]["selected"]]["test"]["all"]
    recomputed = {"eta_MAE": reg["MAE"], "eta_RMSE": reg["RMSE"], "eta_R2": reg["R2"], "baseline_MAE": base_re["MAE"],
                  "improvement_pct": 100 * (1 - reg["MAE"] / base_re["MAE"]), "ROC_AUC": c30["ROC_AUC"],
                  "PR_AUC": c30["PR_AUC"], "precision": c30["precision"], "recall": c30["recall"], "F1": c30["F1"],
                  "threshold": c30["threshold"],
                  "dispatch_MAE": mm["regression"]["variants"]["dispatch_comparison"]["test"]["all"]["MAE"],
                  "dispatch_ROC_AUC": mm["classification"]["variants_gt30"]["dispatch_comparison"]["test"]["ROC_AUC"],
                  "k": mm["clustering"]["k_selected"], "silhouette": mm["clustering"]["silhouette_selected"]}
    tol = {"eta_MAE": .005, "eta_RMSE": .005, "baseline_MAE": .005, "improvement_pct": .05, "dispatch_MAE": .005,
           "dispatch_ROC_AUC": .005, "k": 0}
    verify = {k: {"brief": BRIEF[k], "recomputed": recomputed[k],
                  "match": abs(recomputed[k] - BRIEF[k]) <= tol.get(k, 0.0005 if BRIEF[k] < 1 else 0.005)}
              for k in BRIEF}
    M["stage4_verification"] = verify
    check("every Stage 4 figure in the brief matches recomputation from saved models / metrics (to stated precision)",
          all(v["match"] for v in verify.values()),
          "mismatches: " + str([k for k, v in verify.items() if not v["match"]] or "none"))
    check("saved models reproduce stored Stage 4 test metrics exactly",
          abs(reg["MAE"] - sel["MAE"]) < 1e-6 and c30["confusion"] == card["late_classifiers"]["30"]["test"]["confusion"]
          and abs(base_re["MAE"] - base["MAE"]) < 1e-6)
    M["regression"] = {"test": reg, "baseline_test": base_re, "variants": mm["regression"]["variants"],
                       "families": mm["regression"]["families_final_config"], "segments_test": mm["regression"]["segments_test"]}
    M["classification"] = {"gt30": c30, "sensitivity": {str(k): v for k, v in sens.items()},
                           "variants_gt30": mm["classification"]["variants_gt30"]}

    # ---- operational translation (test period, threshold fixed in Stage 4 on training OOF) ----
    cm, n = c30["confusion"], c30["n"]
    per1000 = {k: 1000 * v / n for k, v in {"flagged": cm["tp"] + cm["fp"], "late_caught": cm["tp"], "false_alarms": cm["fp"],
                                             "late_missed": cm["fn"], "late_total": cm["tp"] + cm["fn"]}.items()}
    M["classification"]["per_1000_orders_test"] = per1000
    M["classification"]["lift_precision_over_base"] = c30["precision"] / c30["prevalence"]
    dec = pd.qcut(pd.Series(p30).rank(method="first"), 10, labels=False)
    tab = pd.DataFrame({"p": p30, "late": (yte > 30).astype(int).to_numpy(), "dec": dec})
    decile = tab.groupby("dec").agg(orders=("late", "size"), mean_pred=("p", "mean"), observed_late=("late", "mean")).reset_index()
    decile["decile"] = 10 - decile["dec"]                     # 1 = highest risk
    decile = decile.sort_values("decile")
    M["classification"]["risk_deciles_test"] = decile[["decile", "orders", "mean_pred", "observed_late"]].to_dict("records")
    dd = decile.assign(late_n=decile["orders"] * decile["observed_late"]).reset_index(drop=True)
    M["classification"]["top_decile_capture"] = [
        {"top_deciles": k, "orders_share_pct": float(100 * dd.orders[:k].sum() / n),
         "observed_late_pct": float(100 * dd.late_n[:k].sum() / dd.orders[:k].sum()),
         "share_of_all_late_pct": float(100 * dd.late_n[:k].sum() / dd.late_n.sum())} for k in (1, 2, 3)]
    top = decile.iloc[0]
    M["classification"]["top_decile"] = {"observed_late": float(top.observed_late), "lift": float(top.observed_late / c30["prevalence"])}
    M["classification"]["scenario_late_rate_reduction_pp"] = {
        f"{int(f * 100)}%": 100 * cm["tp"] / n * f for f in (0.1, 0.2, 0.3)}

    # ======================= 2. explainability (TRAINING split only) =======================
    mae = lambda y, p: mean_absolute_error(y, p)
    auc = lambda y, p: roc_auc_score(y, p)
    imp_eta = grouped_permutation(eta["pipeline"].predict, Xtr, ytr, mae, higher_is_better=False)
    ytr30 = (ytr > 30).astype(int)
    imp_late = grouped_permutation(lambda X: late[30]["pipeline"].predict_proba(X)[:, 1], Xtr, ytr30, auc, higher_is_better=True)
    M["explainability"] = {"data": f"training split only ({len(train):,} rows, {mm['split']['train_dates'][0]} → {mm['split']['train_dates'][1]})",
                           "method": "grouped permutation importance (5 repeats, seed 42) on saved models; collinear features permuted together",
                           "eta_mae_increase_min": imp_eta, "late_auc_drop": imp_late,
                           "stage4_single_feature_test_importance": mm["regression"]["permutation_importance_test_MAE_increase"],
                           "shap": "not used — package not installed; grouped permutation + partial dependence chosen to avoid new dependencies"}
    sample = Xtr.sample(5000, random_state=SEED)

    def set_hour(X, h):
        X["order_hour"] = float(h); X["order_hour_sin"] = np.sin(2 * np.pi * h / 24); X["order_hour_cos"] = np.cos(2 * np.pi * h / 24)
        X["order_time_band"] = time_band(h); return X

    def set_col(col):
        def f(X, v):
            X[col] = v; return X
        return f

    hours = [0] + list(range(8, 24))
    pdp = {
        "hour_eta": partial_dependence(eta["pipeline"].predict, sample, set_hour, hours),
        "hour_late": partial_dependence(lambda X: late[30]["pipeline"].predict_proba(X)[:, 1], sample, set_hour, hours),
        "distance_eta": partial_dependence(eta["pipeline"].predict, sample, set_col("distance_km"), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
        "festival_eta": partial_dependence(eta["pipeline"].predict, sample, set_col("festival"), ["No", "Yes"]),
        "city_type_eta": partial_dependence(eta["pipeline"].predict, sample, set_col("city_type"), ["Urban", "Metropolitan", "Semi-Urban"]),
        "festival_late": partial_dependence(lambda X: late[30]["pipeline"].predict_proba(X)[:, 1], sample, set_col("festival"), ["No", "Yes"]),
        "city_type_late": partial_dependence(lambda X: late[30]["pipeline"].predict_proba(X)[:, 1], sample, set_col("city_type"), ["Urban", "Metropolitan", "Semi-Urban"]),
    }
    M["explainability"]["partial_dependence"] = pdp
    M["explainability"]["partial_dependence_sample"] = "5,000 training rows (seed 42)"

    # ======================= 3. observed associations (descriptive, all rows) =======================
    d = fe.copy()
    d["late30"] = (d["time_taken_min"] > 30).astype(int)
    d["time_band"] = d["order_time_band"].fillna("(order time missing)")
    d["distance_band"] = pd.cut(d["distance_km"], [0, 5, 10, 15, 25], labels=["0–5 km", "5–10 km", "10–15 km", "15–25 km"]).astype("string").fillna("(invalid location)")
    obs = {}
    for col in ["time_band", "festival", "city_type", "distance_band", "traffic_density", "weather", "multiple_deliveries", "vehicle_type", "vehicle_condition", "order_type"]:
        g = d.assign(**{col: d[col].astype("string").fillna("(missing)")}).groupby(col)
        t = pd.DataFrame({"orders": g.size(), "share_pct": 100 * g.size() / len(d), "mean_eta": g["time_taken_min"].mean(),
                          "median_eta": g["time_taken_min"].median(), "late30_pct": 100 * g["late30"].mean()})
        obs[col] = t.reset_index().to_dict("records")
    M["observed"] = obs
    M["observed_overall"] = {"orders": len(d), "mean_eta": float(d["time_taken_min"].mean()), "median_eta": float(d["time_taken_min"].median()),
                             "late30_pct": float(100 * d["late30"].mean())}
    eve = d["order_time_band"] == "evening_19_23"
    aft = d["order_time_band"] == "afternoon_15_18"
    peak = eve | aft
    M["observed_peak"] = {"evening_share_pct": float(100 * eve.mean()), "evening_late_pct": float(100 * d.loc[eve, "late30"].mean()),
                          "peak_15_23_share_pct": float(100 * peak.mean()),
                          "peak_15_23_share_of_late_pct": float(100 * d.loc[peak, "late30"].sum() / d["late30"].sum()),
                          "nonpeak_timed_late_pct": float(100 * d.loc[~peak & d["order_time_band"].notna(), "late30"].mean()),
                          "afternoon_late_pct": float(100 * d.loc[aft, "late30"].mean())}
    op = M["observed_peak"]
    op["scenario_evening_to_afternoon_pp"] = op["evening_share_pct"] / 100 * (op["evening_late_pct"] - op["afternoon_late_pct"])

    # ======================= 4. geography =======================
    km = joblib.load(MOD / "delivery_zones_kmeans.joblib")
    geo = fe[fe["dq_geo_valid"]].reset_index(drop=True)
    Xg = project_km(geo["delivery_lat"], geo["delivery_lon"], km["projection_ref_lat"])
    zone = km["kmeans"].predict(Xg)
    ct = pd.crosstab(zone, geo["city_code"])
    purity_zone = float((ct.max(axis=1) / ct.sum(axis=1)).min())
    purity_city = float((ct.max(axis=0) / ct.sum(axis=0)).min())
    sil_city = float(silhouette_score(Xg, geo["city_code"], sample_size=10_000, random_state=SEED))
    sil_zone = float(silhouette_score(Xg, zone, sample_size=10_000, random_state=SEED))
    geo = geo.assign(zone=zone, late30=(geo["time_taken_min"] > 30).astype(int))
    chi = stats.chi2_contingency(pd.crosstab(geo["zone"], geo["late30"]))
    by_zone = geo.groupby("zone").agg(orders=("late30", "size"), late30=("late30", "mean"), median_eta=("time_taken_min", "median"))
    # spread of delivery points around their zone centre (km)
    cent = km["kmeans"].cluster_centers_[zone]
    radius = np.sqrt(((Xg - cent) ** 2).sum(axis=1))
    band_late = pd.Series({b["time_band"]: b["late30_pct"] for b in obs["time_band"] if not b["time_band"].startswith("(")})
    M["geography"] = {
        "k": int(km["k"]), "silhouette_stage4": mm["clustering"]["silhouette_selected"],
        "silhouette_recomputed_zones": sil_zone, "silhouette_of_city_labels": sil_city,
        "min_zone_purity_by_city": purity_zone, "min_city_purity_by_zone": purity_city,
        "zones_equal_cities": purity_zone == 1.0 and purity_city == 1.0 and ct.shape == (22, 22),
        "zone_late30_range_pct": [float(100 * by_zone.late30.min()), float(100 * by_zone.late30.max())],
        "zone_median_eta_range": [float(by_zone.median_eta.min()), float(by_zone.median_eta.max())],
        "zone_orders_range": [int(by_zone.orders.min()), int(by_zone.orders.max())],
        "zone_late_chi2_p": float(chi[1]),
        "time_band_late30_range_pct": [float(band_late.min()), float(band_late.max())],
        "delivery_point_radius_km": {"median": float(np.median(radius)), "p95": float(np.quantile(radius, .95)), "max": float(radius.max())},
        "distance_km": {"median": float(fe["distance_km"].median()), "min": float(fe["distance_km"].min()), "max": float(fe["distance_km"].max())},
        "lat_offset_equals_lon_offset_share": audit["geo"]["lat_offset_equals_lon_offset_share"],
        "geo_invalid_rows": int((~fe["dq_geo_valid"]).sum()),
        "geo_invalid_top_cities": dict(list(mm["clustering"]["geo_invalid_share_by_city"].items())[:3]),
        "unique_restaurant_locations_valid": int(fe.loc[fe.dq_geo_valid, ["restaurant_lat", "restaurant_lon"]].drop_duplicates().shape[0]),
    }
    check("K-Means zones recomputed from saved model reproduce Stage 4 silhouette (±0.005)",
          abs(sil_zone - mm["clustering"]["silhouette_selected"]) <= 0.005, f"{sil_zone:.4f} vs {mm['clustering']['silhouette_selected']:.4f}")

    # ======================= 5. rider-segmentation feasibility =======================
    cl = pd.read_csv(CLEAN, usecols=["delivery_person_id", "delivery_person_age_raw", "vehicle_type", "restaurant_lat",
                                     "restaurant_lon", "dq_geo_valid"])
    g = cl.groupby("delivery_person_id")
    locs = cl[cl.dq_geo_valid].groupby("delivery_person_id")[["restaurant_lat", "restaurant_lon"]].apply(lambda x: len(x.drop_duplicates()))
    M["rider"] = {
        "ids": int(cl["delivery_person_id"].nunique()),
        "orders_per_id": {k: float(v) for k, v in g.size().describe()[["min", "50%", "max"]].items()},
        "distinct_ages_per_id": {k: float(v) for k, v in g["delivery_person_age_raw"].nunique().describe()[["min", "50%", "max"]].items()},
        "ids_with_multiple_ages_pct": float(100 * (g["delivery_person_age_raw"].nunique() > 1).mean()),
        "ids_with_multiple_vehicle_types_pct": float(100 * (g["vehicle_type"].nunique() > 1).mean()),
        "max_valid_restaurant_locations_per_id": int(locs.max()),
        "id_suffixes": sorted(cl["delivery_person_id"].str.extract(r"DEL(\d+)$")[0].unique().tolist()),
        "verdict": "not feasible",
    }
    check("rider-ID evidence recomputed from cleaned data (every ID has >1 age)", M["rider"]["ids_with_multiple_ages_pct"] == 100.0)

    # ======================= 6. dispatch-time comparison (evidence only) =======================
    rv, cv = mm["regression"]["variants"], mm["classification"]["variants_gt30"]
    M["dispatch_comparison"] = {
        "persisted_as_model": (MOD / "dispatch_eta.joblib").exists(),
        "inputs": "14 placement features (incl. order_month) + rider age, rider rating, vehicle condition, multiple deliveries, weather, traffic, vehicle type",
        "eta_MAE": rv["dispatch_comparison"]["test"]["all"]["MAE"], "eta_R2": rv["dispatch_comparison"]["test"]["all"]["R2"],
        "eta_MAE_no_rating": rv["dispatch_no_rating"]["test"]["all"]["MAE"], "eta_R2_no_rating": rv["dispatch_no_rating"]["test"]["all"]["R2"],
        "late_ROC_AUC": cv["dispatch_comparison"]["test"]["ROC_AUC"], "late_ROC_AUC_no_rating": cv["dispatch_no_rating"]["test"]["ROC_AUC"],
        "late_PR_AUC_no_rating": cv["dispatch_no_rating"]["test"]["PR_AUC"], "late_F1_no_rating": cv["dispatch_no_rating"]["test"]["F1"],
        "mae_reduction_no_rating_pct": 100 * (1 - rv["dispatch_no_rating"]["test"]["all"]["MAE"] / reg["MAE"]),
        "placement_variant_same_protocol_MAE": rv["primary"]["test"]["all"]["MAE"],
    }

    # ======================= 7. data-quality summary (from earlier stages) =======================
    M["data"] = {"rows": audit["shape"]["rows"], "cols": audit["shape"]["cols"], "dates": [audit["time"]["order_date_min"], audit["time"]["order_date_max"]],
                 "rows_any_missing_raw": audit["rows_any_missing"], "duplicates": audit["duplicates"]["full_row_duplicates"],
                 "sign_corrected": cleaning["flag_counts"]["dq_restaurant_sign_corrected"], "geo_invalid": cleaning["flag_counts"]["dq_geo_valid"],
                 "rider_sentinel": cleaning["flag_counts"]["dq_rider_sentinel"], "missing_order_time": mts["n_missing"],
                 "missing_time_target_p": mts["target"]["mann_whitney_p"], "traffic_hour_consistency_pct": mts["recovery"]["traffic_hour_band_consistency_pct"],
                 "festival_missing": cleaning["steps"]["missing_after_cleaning"]["festival"], "rows_retained": cleaning["rows_out"],
                 "split": mm["split"], "late30_overall_pct": float(100 * (fe["time_taken_min"] > 30).mean()),
                 "target_quantiles": audit["target"]["quantiles"], "primary_features": len(feats),
                 "target_outliers_iqr": audit["outliers_iqr"]["target_min"]["n_above"]}
    M["stage4_classifiers"] = {n: {"cv": v["cv"], "test": v["test_at_f1_threshold"]} for n, v in mm["classification"]["gt30"].items()}

    # ======================= 8. charts =======================
    figs = []
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, imp, xl, title in [(ax[0], imp_eta, "increase in training MAE (minutes)", "ETA model"),
                              (ax[1], imp_late, "drop in training ROC-AUC", "Late-risk model (> 30 min)")]:
        gs = list(imp["groups"].items())[::-1]
        a.barh([g for g, _ in gs], [v["mean"] for _, v in gs], xerr=[v["sd"] for _, v in gs], color=BLUE, height=0.6, ecolor=INK2)
        a.set_xlabel(xl); a.set_title(f"{title}: what the model relies on"); a.grid(axis="y", visible=False)
    fig.suptitle("Grouped permutation importance — saved primary models, training data only", fontweight="bold", color=INK)
    figs.append(save(fig, "22_explainability_grouped_importance.png"))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    hp = pd.DataFrame(pdp["hour_eta"]); ax[0].plot(hp.value, hp.mean_prediction, color=BLUE, lw=2, marker="o", ms=4)
    ax[0].set(xlabel="order hour (01–07 not in data)", ylabel="avg predicted ETA (min)", title="Time of day"); ax[0].set_xticks(hours[::2])
    dp = pd.DataFrame(pdp["distance_eta"]); ax[1].plot(dp.value, dp.mean_prediction, color=BLUE, lw=2, marker="o", ms=4)
    ax[1].set(xlabel="distance (km)", ylabel="avg predicted ETA (min)", title="Delivery distance")
    cats = [(f"Festival: {r['value']}", r["mean_prediction"]) for r in pdp["festival_eta"]] + \
           [(f"{r['value']}", r["mean_prediction"]) for r in pdp["city_type_eta"]]
    ax[2].barh([c for c, _ in cats][::-1], [v for _, v in cats][::-1], color=BLUE, height=0.6)
    for i, (_, v) in enumerate(cats[::-1]):
        ax[2].text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9, color=INK2)
    ax[2].set(xlabel="avg predicted ETA (min)", title="Festival day & city type"); ax[2].grid(axis="y", visible=False)
    fig.suptitle("How the ETA model's prediction changes with each factor (partial dependence; association, not causation)", fontweight="bold", color=INK)
    figs.append(save(fig, "23_eta_partial_dependence.png"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(decile["decile"].astype(str), 100 * decile["observed_late"], color=BLUE, width=0.65)
    ax.axhline(100 * c30["prevalence"], color=ORANGE, ls="--", lw=1.5)
    ax.text(9.6, 100 * c30["prevalence"] + 1.5, f"average {100 * c30['prevalence']:.1f}%", color=ORANGE, ha="right", fontsize=9)
    for i, v in enumerate(decile["observed_late"]):
        ax.text(i, 100 * v + 1, f"{100 * v:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set(xlabel="predicted-risk decile (1 = riskiest 10% of orders)", ylabel="% actually late (> 30 min)",
           title="Late-risk model: observed late rate by predicted-risk decile (test period)")
    figs.append(save(fig, "24_late_risk_deciles.png"))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ranges = [("Across 22 geographic zones", M["geography"]["zone_late30_range_pct"]),
              ("Across order-time bands", M["geography"]["time_band_late30_range_pct"]),
              (f"Festival No vs Yes (Yes = {next(r['orders'] for r in obs['festival'] if r['festival'] == 'Yes'):,} orders)", sorted(r["late30_pct"] for r in obs["festival"] if r["festival"] in ("No", "Yes"))),
              (f"Across city types (Semi-Urban = {next(r['orders'] for r in obs['city_type'] if r['city_type'] == 'Semi-Urban'):,} orders)", [min(r["late30_pct"] for r in obs["city_type"] if r["city_type"] != "(missing)"),
                                     max(r["late30_pct"] for r in obs["city_type"] if r["city_type"] != "(missing)")])]
    for i, (lbl, (lo, hi)) in enumerate(ranges):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=6, solid_capstyle="round")
        ax.text(hi + 1.5, i, f"{lo:.1f}% – {hi:.1f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(ranges)), [r[0] for r in ranges]); ax.invert_yaxis()
    ax.set_xlim(0, 105); ax.set_xlabel("% of orders late (> 30 min), observed")
    ax.set_title("Where late-rate differences are large vs small (observed, all orders)\n"
                 "100% rates are deterministic patterns typical of synthetic data"); ax.grid(axis="y", visible=False)
    figs.append(save(fig, "25_late_rate_ranges.png"))
    # ---- analyses required by the official challenge guide (gap fill) ----
    from src import challenge_gaps
    figs += challenge_gaps.compute(fe, train, test, Xtr, ytr, Xte, yte, p30, late[30]["decision_threshold"], km, M,
                                   plt, (BLUE, ORANGE, AQUA, GRAY, INK, INK2), save)
    M["figures"] = figs

    # ======================= write & validate =======================
    hashes_after = {p.name: sha(p) for p in (RAW, CLEAN, FE)}
    check("raw, cleaned and engineered datasets unchanged after Stage 5", hashes_after == hashes_before == EXPECTED_HASHES)
    check("explainability computed on training rows only", len(Xtr) == mm["split"]["train_rows"], f"{len(Xtr):,} rows")
    from src.insights_report import write_reports
    n_rec = write_reports(M)
    rec_text = (REP / "business_recommendations.md").read_text(encoding="utf-8")
    check("exactly three prioritised recommendations", n_rec == 3 and rec_text.count("\n## Priority ") == 3, f"{n_rec}")
    docs = ["business_insights_report.md", "business_report.md", "challenge_compliance.md"]
    body = "".join((REP / x).read_text(encoding="utf-8") for x in docs) + rec_text
    missing_refs = [f for f in __import__("re").findall(r"figures/\d\d_[\w\-]+\.png", body) if not (ROOT / f).exists()]
    check("every figure referenced in the Stage 5 documents exists", not missing_refs, missing_refs or "all present")
    check("challenge guide PDF found and used for the compliance check",
          "Hackathon_Challenge_Guide.pdf" in M["challenge_pdf_found_in_project"], M["challenge_pdf_found_in_project"])
    wc = M["report_word_counts"]
    check("Business Report within the guide's 4–6 pages (estimated 1,800–3,300 words)",
          1800 <= wc["business_report"] <= 3300, f"{wc['business_report']:,} words")
    check("Business Solution Document at least 5 pages (estimated ≥ 2,500 words)", wc["solution_doc"] >= 2500,
          f"{wc['solution_doc']:,} words")
    # 'placeholder' alone is a legitimate domain term here (placeholder rider records), so look for stub markers instead.
    stub = __import__("re").search(r"Placeholder;|placeholder\n|lorem ipsum|\bTODO\b|\bTBD\b|\bXXX\b", body)
    small = [x for x in docs + ["business_recommendations.md"] if len((REP / x).read_text(encoding="utf-8")) < 3000]
    check("Stage 5 documents are complete (no stub markers; every document > 3,000 characters)",
          stub is None and not small, f"stub: {stub.group(0) if stub else 'none'}; short files: {small or 'none'}")
    check("rider profiles labelled as order-level, not rider segmentation",
          "NOT rider segmentation" in body and "not individual riders" in body)
    check("30-minute benchmark never described as an official SLA",
          "official SLA" not in body.replace("not an official SLA", "").replace("no official SLA", "").replace("No official SLA", "")
          .replace("not an official quick-commerce SLA", "").replace("no official quick-commerce SLA", ""))
    OUT_JSON.write_text(json.dumps(M, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_reports(M)    # re-render so the validation table includes the final checks
    failed = [c["check"] for c in M["checks"] if not c["passed"]]
    print(f"checks: {len(M['checks']) - len(failed)}/{len(M['checks'])} passed")
    for c in M["checks"]:
        print(("  PASS " if c["passed"] else "  FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    if failed:
        raise SystemExit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()
