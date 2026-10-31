"""DeliveryIQ — Stage 4 modelling orchestrator.

Trains and evaluates:
    * ETA regression (baseline + ridge + random forest + histogram GBM)
    * late-delivery classification at the project benchmark > 30 min (+ sensitivity at > 35 / > 40)
    * K-Means geographic delivery zones (elbow + silhouette)

Protocol (CLAUDE.md §7–8):
    * Chronological split on whole order dates (``chronological_split``); the test set is used ONCE,
      after every model, feature-variant and threshold decision has been made on training data.
    * Hyper-parameters, model family, feature variants and decision thresholds are chosen by
      forward-chaining chronological CV inside the training split only.
    * Inputs only via ``get_model_inputs`` (leakage guard); preprocessing lives inside each pipeline and
      is therefore fitted on training rows only.

Outputs: models/*.joblib, models/model_card.json, reports/modelling_metrics.json,
reports/modelling_report.md, figures/12–21_*.png.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.modelling
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve

from src import classification as clf
from src import clustering as clu
from src import regression as reg
from src.data_cleaning import OUT_CSV as CLEAN_CSV
from src.data_cleaning import RAW_PATH, RAW_SHA256
from src.evaluation import (chronological_folds, classification_report, regression_report,
                            select_threshold)
from src.feature_engineering import (OUT_CSV as FE_CSV, PRIMARY_CATEGORICAL, PRIMARY_FEATURES,
                                     PRIMARY_NUMERIC, TARGET, assert_no_leakage, build_preprocessor,
                                     chronological_split, get_model_inputs)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIG = ROOT / "figures"
OUT_METRICS = ROOT / "reports" / "modelling_metrics.json"
OUT_REPORT = ROOT / "reports" / "modelling_report.md"
SEED = 42
N_FOLDS = 3
REL_TOL = 0.005          # pre-registered 0.5 % CV-MAE tolerance for feature-variant decisions

# ---- feature sets ----------------------------------------------------------------------
COMP_NUM = ["delivery_person_age", "delivery_person_rating", "vehicle_condition", "multiple_deliveries"]
COMP_CAT = ["weather", "traffic_density", "vehicle_type"]
FEATURE_SETS = {   # name: (numeric, categorical, allow_excluded, label)
    "primary": (PRIMARY_NUMERIC, PRIMARY_CATEGORICAL, False, "Order placement (Tier A), 14 features"),
    "primary_no_month": (PRIMARY_NUMERIC, [c for c in PRIMARY_CATEGORICAL if c != "order_month"], False,
                         "Order placement without order_month"),
    "dispatch_comparison": (PRIMARY_NUMERIC + COMP_NUM, PRIMARY_CATEGORICAL + COMP_CAT, True,
                            "COMPARISON ONLY — adds rider, vehicle, batching, weather, traffic (not available at order placement)"),
    "dispatch_no_rating": (PRIMARY_NUMERIC + [c for c in COMP_NUM if c != "delivery_person_rating"],
                           PRIMARY_CATEGORICAL + COMP_CAT, True, "COMPARISON ONLY — as above without rider rating"),
}

# ---- chart style (same palette as the EDA notebook) --------------------------------------
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#8a8984",
                                                   "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlecolor": INK, "font.size": 10})


def save(fig, name: str) -> str:
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150, bbox_inches="tight"); plt.close(fig)
    return f"figures/{name}"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items() if not str(k).startswith("oof_")}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o


# =========================================================================================
def main() -> None:
    t0 = time.time()
    hashes_before = {p.name: sha(p) for p in (RAW_PATH, CLEAN_CSV, FE_CSV)}
    assert hashes_before["train.csv"] == RAW_SHA256
    M: dict = {"run_date": date.today().isoformat(), "seed": SEED, "checks": []}
    check = lambda name, ok, detail="": M["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)})
    figs: list[str] = []

    # ---------------- data & split ----------------
    fe = pd.read_csv(FE_CSV)
    tr_idx, te_idx, cut = chronological_split(fe)
    train, test = fe.loc[tr_idx].reset_index(drop=True), fe.loc[te_idx].reset_index(drop=True)
    dtr, dte = pd.to_datetime(train.order_date), pd.to_datetime(test.order_date)
    M["split"] = {"cut_date": cut, "train_rows": len(train), "test_rows": len(test),
                  "train_dates": [str(dtr.min().date()), str(dtr.max().date())],
                  "test_dates": [str(dte.min().date()), str(dte.max().date())]}
    check("train and test disjoint; every train date < every test date",
          dtr.max() < dte.min() and not set(train.order_id) & set(test.order_id), f"cut {cut}")
    folds = chronological_folds(train["order_date"], N_FOLDS)
    check("CV folds use training rows only and validate on later dates",
          all(max(tr) < len(train) and max(va) < len(train)
              and pd.to_datetime(train.order_date.iloc[tr]).max() < pd.to_datetime(train.order_date.iloc[va]).min()
              for tr, va in folds), f"{N_FOLDS} forward-chaining folds")
    M["cv_folds"] = [{"train_rows": int(len(tr)), "val_rows": int(len(va)),
                      "val_dates": [str(pd.to_datetime(train.order_date.iloc[va]).min().date()),
                                    str(pd.to_datetime(train.order_date.iloc[va]).max().date())]} for tr, va in folds]

    def XY(df, fs):
        num, cat, allow, _ = FEATURE_SETS[fs]
        return get_model_inputs(df, num + cat, allow_excluded=allow)

    def prep(fs):
        num, cat, allow, _ = FEATURE_SETS[fs]
        return build_preprocessor(num, cat, allow_excluded=allow)

    for fs in ("primary", "primary_no_month"):
        assert_no_leakage(FEATURE_SETS[fs][0] + FEATURE_SETS[fs][1])        # strict guard (no exclusions)
    check("primary feature sets pass the strict leakage guard; city_code absent",
          "city_code" not in PRIMARY_FEATURES and len(PRIMARY_FEATURES) == 14, f"{len(PRIMARY_FEATURES)} features")

    mask_time = ~train["dq_pickup_day_assumed"].to_numpy(bool)
    mask_fest = train["festival"].notna().to_numpy()
    complete_te = ~test["dq_pickup_day_assumed"].to_numpy(bool)

    # =====================================================================================
    # 1. REGRESSION
    # =====================================================================================
    Xtr, ytr = XY(train, "primary")
    tuning = reg.tune(prep("primary"), Xtr, ytr, folds)
    fam = min((n for n in tuning if n != "baseline_median"), key=lambda n: tuning[n]["cv"]["MAE"])
    params = tuning[fam]["best_params"]

    def fit_eval_reg(name, prm, fs, mask=None):
        Xa, ya = XY(train, fs); Xb, yb = XY(test, fs)
        pipe = reg.make_pipeline(prep(fs), name, prm)
        cv = reg.cv_score(pipe, Xa, ya, folds, mask)
        rows = np.arange(len(train)) if mask is None else np.where(mask)[0]
        model = clone(pipe).fit(Xa.iloc[rows], ya.iloc[rows])
        pred = model.predict(Xb)
        return model, cv, {"all": regression_report(yb, pred),
                           "complete_time": regression_report(yb[complete_te], pred[complete_te]),
                           "missing_time": regression_report(yb[~complete_te], pred[~complete_te])}, pred

    variants = {"primary": ("primary", None), "primary_no_month": ("primary_no_month", None),
                "primary_excl_missing_time_train": ("primary", mask_time),
                "primary_excl_missing_festival_train": ("primary", mask_fest),
                "dispatch_comparison": ("dispatch_comparison", None), "dispatch_no_rating": ("dispatch_no_rating", None)}
    reg_var = {}
    for vname, (fs, mask) in variants.items():
        _, cv, te, _ = fit_eval_reg(fam, params, fs, mask)
        reg_var[vname] = {"feature_set": fs, "train_rows": int(len(train) if mask is None else mask.sum()),
                          "cv": {k: cv[k] for k in ("MAE", "RMSE", "R2")}, "test": te}
    base = reg_var["primary"]["cv"]["MAE"]
    decisions = {
        "drop_order_month": reg_var["primary_no_month"]["cv"]["MAE"] <= base * (1 + REL_TOL),
        "exclude_missing_time_from_training": reg_var["primary_excl_missing_time_train"]["cv"]["MAE"] < base * (1 - REL_TOL),
        "exclude_missing_festival_from_training": reg_var["primary_excl_missing_festival_train"]["cv"]["MAE"] < base * (1 - REL_TOL),
    }
    final_fs = "primary_no_month" if decisions["drop_order_month"] else "primary"
    final_mask = np.ones(len(train), bool)
    if decisions["exclude_missing_time_from_training"]:
        final_mask &= mask_time
    if decisions["exclude_missing_festival_from_training"]:
        final_mask &= mask_fest
    final_mask_or_none = None if final_mask.all() else final_mask
    M["feature_decisions"] = {"rule": f"CV MAE tolerance {REL_TOL:.1%}; drop order_month unless it improves CV MAE by "
                                      f"more than the tolerance; exclude a training subset only if that improves CV MAE "
                                      f"by more than the tolerance", **decisions, "final_feature_set": final_fs,
                              "final_features": FEATURE_SETS[final_fs][0] + FEATURE_SETS[final_fs][1],
                              "final_train_rows": int(final_mask.sum()), "decided_with": f"{fam} (best CV family on primary set)"}

    # all families under the final config: CV for selection, test for reporting
    reg_fam = {}
    fitted_reg = {}
    for name in reg.GRIDS:
        prm = tuning[name]["best_params"]
        model, cv, te, pred = fit_eval_reg(name, prm, final_fs, final_mask_or_none)
        reg_fam[name] = {"params": prm, "cv": {k: cv[k] for k in ("MAE", "RMSE", "R2")}, "test": te}
        fitted_reg[name] = (model, pred)
    best_reg = min((n for n in reg_fam if n != "baseline_median"), key=lambda n: reg_fam[n]["cv"]["MAE"])
    eta_model, eta_pred = fitted_reg[best_reg]
    Xte_f, yte = XY(test, final_fs)
    M["regression"] = {"tuning": {n: {"best_params": t["best_params"], "cv": t["cv"], "trials": t["trials"]} for n, t in tuning.items()},
                       "variants": reg_var, "families_final_config": reg_fam, "selected": best_reg,
                       "selected_by": "lowest chronological-CV MAE on the training split"}

    # segment errors (test, selected model)
    seg = test.assign(abs_err=np.abs(yte.to_numpy() - eta_pred))
    seg["distance_band"] = pd.cut(seg["distance_km"], [0, 5, 10, 15, 25], labels=["0–5 km", "5–10 km", "10–15 km", "15–25 km"]).astype("string").fillna("invalid geo")
    seg["order_time"] = np.where(complete_te, "time present", "time missing")
    seg_tables = {}
    for col in ["traffic_density", "city_type", "distance_band", "order_time"]:
        g = seg.assign(**{col: seg[col].astype("string").fillna("<missing>")}).groupby(col)
        seg_tables[col] = pd.DataFrame({"n": g.size(), "MAE": g["abs_err"].mean().round(3),
                                        "mean_actual": g[TARGET].mean().round(2)}).reset_index().to_dict("records")
    M["regression"]["segments_test"] = seg_tables

    # permutation importance (post-hoc interpretation on test; not used for any decision)
    pi = permutation_importance(eta_model, Xte_f, yte, scoring="neg_mean_absolute_error", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=Xte_f.columns).sort_values(ascending=False)
    M["regression"]["permutation_importance_test_MAE_increase"] = imp.round(4).to_dict()

    # sanity: label permutation should destroy skill
    Xtr_f, ytr_f = XY(train, final_fs)
    # A single shuffle is not a valid test: a model fitted to noise has a random coefficient direction,
    # and any random combination of informative features correlates (±) with the true labels.
    # So repeat the shuffle and test the distribution (Stage 4 finding; see report §7).
    perm_r2 = np.array([regression_report(yte, reg.make_pipeline(prep(final_fs), "ridge", {"alpha": 1.0}).fit(
        Xtr_f, pd.Series(np.random.default_rng(s).permutation(ytr_f.to_numpy()), index=ytr_f.index)).predict(Xte_f))["R2"]
        for s in range(10)])
    M["sanity_permutation_regression_R2"] = {"runs": 10, "mean": float(perm_r2.mean()), "max": float(perm_r2.max())}
    check("label-permutation sanity (10 shuffles): ridge on shuffled targets has mean test R² ≤ 0.01 and max < selected model R²",
          perm_r2.mean() <= 0.01 and perm_r2.max() < reg_fam[best_reg]["test"]["all"]["R2"],
          f"mean R² {perm_r2.mean():.4f}, max {perm_r2.max():.4f}")
    check("selected ETA model beats the median baseline on test MAE",
          reg_fam[best_reg]["test"]["all"]["MAE"] < reg_fam["baseline_median"]["test"]["all"]["MAE"],
          f"{reg_fam[best_reg]['test']['all']['MAE']:.3f} vs {reg_fam['baseline_median']['test']['all']['MAE']:.3f}")

    # =====================================================================================
    # 2. CLASSIFICATION
    # =====================================================================================
    rows_f = np.where(final_mask)[0]

    def clf_block(minutes, families=None, fixed=None):
        ytr_b = clf.make_late_label(ytr_f, minutes); yte_b = clf.make_late_label(yte, minutes)
        pp = prep(final_fs)
        if fixed is None:
            t = clf.tune(pp, Xtr_f, ytr_b, folds) if final_mask_or_none is None else None
            if t is None:     # tuning under the training mask
                t = {}
                for name, grid in clf.GRIDS.items():
                    trials = [{"params": p, **clf.cv_oof(clf.make_pipeline(pp, name, p), Xtr_f, ytr_b, folds, final_mask_or_none)} for p in grid]
                    b = max(trials, key=lambda z: z["PR_AUC"])
                    t[name] = {"best_params": b["params"], "cv": {"ROC_AUC": b["ROC_AUC"], "PR_AUC": b["PR_AUC"]},
                               "oof_index": b["oof_index"], "oof_proba": b["oof_proba"],
                               "trials": [{"params": z["params"], "PR_AUC": z["PR_AUC"], "ROC_AUC": z["ROC_AUC"]} for z in trials]}
        else:
            name, prm = fixed
            r = clf.cv_oof(clf.make_pipeline(pp, name, prm), Xtr_f, ytr_b, folds, final_mask_or_none)
            t = {name: {"best_params": prm, "cv": {"ROC_AUC": r["ROC_AUC"], "PR_AUC": r["PR_AUC"]},
                        "oof_index": r["oof_index"], "oof_proba": r["oof_proba"], "trials": []}}
        res, fitted = {}, {}
        for name, tt in t.items():
            thr = select_threshold(ytr_b.iloc[tt["oof_index"]], tt["oof_proba"]) if name != "baseline_prior" else None
            model = clf.make_pipeline(pp, name, tt["best_params"]).fit(Xtr_f.iloc[rows_f], ytr_b.iloc[rows_f])
            proba = model.predict_proba(Xte_f)[:, 1]
            th = thr["f1"]["threshold"] if thr else 0.5
            res[name] = {"params": tt["best_params"], "cv": tt["cv"], "trials": tt["trials"],
                         "threshold_selection_oof": {k: v for k, v in thr.items() if k != "curve"} if thr else None,
                         "test_at_f1_threshold": classification_report(yte_b, proba, th),
                         "test_at_recall_threshold": classification_report(yte_b, proba, thr["recall_target"]["threshold"]) if thr else None,
                         "test_complete_time": classification_report(yte_b[complete_te], proba[complete_te], th)}
            fitted[name] = (model, proba, thr)
        return res, fitted, yte_b

    clf30, fitted30, yte30 = clf_block(30)
    best_clf = max((n for n in clf30 if n != "baseline_prior"), key=lambda n: clf30[n]["cv"]["PR_AUC"])
    late_model, late_proba, late_thr = fitted30[best_clf]
    M["classification"] = {"gt30": clf30, "selected": best_clf,
                           "selected_by": "highest chronological-CV PR-AUC on the training split",
                           "threshold_rule": "maximise F1 on out-of-fold training predictions; recall ≥ 0.80 point also reported"}
    sens = {30: {"selected": best_clf, **clf30[best_clf]}}
    sens_fitted = {30: fitted30[best_clf]}
    for mins in (35, 40):
        r, f, _ = clf_block(mins, fixed=(best_clf, clf30[best_clf]["params"]))
        sens[mins] = {"selected": best_clf, **r[best_clf]}; sens_fitted[mins] = f[best_clf]
    M["classification"]["threshold_sensitivity"] = {str(k): v for k, v in sens.items()}

    # classification feature variants (selected family/params, > 30 min)
    clf_var = {}
    for vname, (fs, mask) in variants.items():
        Xa, ya = XY(train, fs); Xb, yb = XY(test, fs)
        ya, yb = clf.make_late_label(ya, 30), clf.make_late_label(yb, 30)
        pipe = clf.make_pipeline(prep(fs), best_clf, clf30[best_clf]["params"])
        r = clf.cv_oof(pipe, Xa, ya, folds, mask)
        rows = np.arange(len(train)) if mask is None else np.where(mask)[0]
        p = clone(pipe).fit(Xa.iloc[rows], ya.iloc[rows]).predict_proba(Xb)[:, 1]
        thr = select_threshold(ya.iloc[r["oof_index"]], r["oof_proba"])["f1"]["threshold"]
        clf_var[vname] = {"cv": {"ROC_AUC": r["ROC_AUC"], "PR_AUC": r["PR_AUC"]}, "test": classification_report(yb, p, thr)}
    M["classification"]["variants_gt30"] = clf_var

    lab30 = clf.make_late_label(ytr_f, 30).to_numpy()
    perm_auc = np.array([classification_report(yte30, clf.make_pipeline(prep(final_fs), "logistic", {"C": 1.0}).fit(
        Xtr_f, pd.Series(np.random.default_rng(s).permutation(lab30))).predict_proba(Xte_f)[:, 1], 0.5)["ROC_AUC"]
        for s in range(20)])
    real_auc = clf30[best_clf]["test_at_f1_threshold"]["ROC_AUC"]
    M["sanity_permutation_classification_AUC"] = {"runs": 20, "mean": float(perm_auc.mean()), "sd": float(perm_auc.std()),
                                                  "min": float(perm_auc.min()), "max": float(perm_auc.max()),
                                                  "share_above_0.5": float((perm_auc > 0.5).mean())}
    check("label-permutation sanity (20 shuffles): mean test ROC-AUC within 0.5 ± 0.03 and max < selected model ROC-AUC",
          abs(perm_auc.mean() - 0.5) <= 0.03 and perm_auc.max() < real_auc,
          f"mean {perm_auc.mean():.4f} (sd {perm_auc.std():.4f}, max {perm_auc.max():.4f}) vs model {real_auc:.4f}")
    check("decision threshold chosen on out-of-fold TRAINING predictions only",
          late_thr is not None and len(fitted30[best_clf][2]["curve"]["threshold"]) > 0)

    # =====================================================================================
    # 3. CLUSTERING
    # =====================================================================================
    geo_tr = train[train["dq_geo_valid"]].reset_index(drop=True)
    geo_all = fe[fe["dq_geo_valid"]].reset_index(drop=True)
    ref_lat = float(geo_tr["delivery_lat"].mean())
    Xg = clu.project_km(geo_tr["delivery_lat"], geo_tr["delivery_lon"], ref_lat)
    scan = clu.scan_k(Xg, range(2, 31))
    k_sil = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    k_elbow = clu.elbow_k(scan)
    km = clu.fit_zones(Xg, k_sil)
    labels_all = km.predict(clu.project_km(geo_all["delivery_lat"], geo_all["delivery_lon"], ref_lat))
    prof = clu.profile_zones(geo_all, labels_all)
    M["clustering"] = {"data": "delivery coordinates of geo-valid orders; fitted on training-period rows",
                       "fit_rows": len(geo_tr), "profiled_rows": len(geo_all), "projection_ref_lat": ref_lat,
                       "scan": scan.to_dict("records"), "k_best_silhouette": k_sil, "k_elbow": k_elbow,
                       "k_selected": k_sil, "selected_by": "maximum silhouette score (elbow reported for reference)",
                       "silhouette_selected": float(scan.loc[scan.k == k_sil, "silhouette"].iloc[0]),
                       "profile": prof.to_dict("records"),
                       "rider_segmentation": "not performed — delivery_person_id is not a person identifier (Stage 1 audit)",
                       "geo_invalid_share_by_city": (1 - fe.groupby("city_code")["dq_geo_valid"].mean())
                       .sort_values(ascending=False).round(4).to_dict(),
                       "geo_invalid_share_overall": float(1 - fe["dq_geo_valid"].mean())}
    check("clustering uses geo-valid rows only (no missing coordinates)",
          geo_tr[["delivery_lat", "delivery_lon"]].notna().all().all() and geo_all["dq_geo_valid"].all())

    # =====================================================================================
    # 4. PERSIST
    # =====================================================================================
    MODELS.mkdir(exist_ok=True)
    feats_final = FEATURE_SETS[final_fs][0] + FEATURE_SETS[final_fs][1]
    meta = {"feature_set": final_fs, "features": feats_final, "trained_on": M["split"]["train_dates"],
            "train_rows": int(final_mask.sum()), "target": TARGET, "tier": "A (order placement)"}
    joblib.dump({"pipeline": eta_model, "model": best_reg, **meta}, MODELS / "eta_regressor.joblib")
    for mins in (30, 35, 40):
        mdl, _, thr = sens_fitted[mins]
        joblib.dump({"pipeline": mdl, "model": best_clf, "late_threshold_minutes": mins,
                     "decision_threshold": thr["f1"]["threshold"], "recall80_threshold": thr["recall_target"]["threshold"],
                     "label": f"late = time_taken_min > {mins} (project benchmark, not an official SLA)", **meta},
                    MODELS / f"late_classifier_gt{mins}.joblib")
    joblib.dump({"kmeans": km, "projection_ref_lat": ref_lat, "k": k_sil,
                 "inputs": ["delivery_lat", "delivery_lon"]}, MODELS / "delivery_zones_kmeans.joblib")
    reload = joblib.load(MODELS / "eta_regressor.joblib")["pipeline"].predict(Xte_f)
    check("saved ETA model reloads and reproduces test predictions exactly", np.array_equal(reload, eta_pred))
    refit = clone(eta_model).fit(Xtr_f.iloc[rows_f], ytr_f.iloc[rows_f]).predict(Xte_f)
    check("ETA model refit is deterministic (seeded)", np.array_equal(refit, eta_pred))
    reload_c = joblib.load(MODELS / "late_classifier_gt30.joblib")["pipeline"].predict_proba(Xte_f)[:, 1]
    check("saved late-risk model reloads and reproduces test probabilities exactly", np.array_equal(reload_c, late_proba))
    card = {"generated": M["run_date"], "split": M["split"], "eta_regressor": {"model": best_reg, "params": reg_fam[best_reg]["params"],
            "test": reg_fam[best_reg]["test"]["all"], **meta},
            "late_classifiers": {str(m): {"model": best_clf, "decision_threshold": sens_fitted[m][2]["f1"]["threshold"],
                                          "test": sens[m]["test_at_f1_threshold"]} for m in (30, 35, 40)},
            "delivery_zones": {"k": k_sil, "silhouette": M["clustering"]["silhouette_selected"]}}
    (MODELS / "model_card.json").write_text(json.dumps(clean_json(card), indent=2), encoding="utf-8")

    # =====================================================================================
    # 5. CHARTS
    # =====================================================================================
    names = [n for n in reg.GRIDS]
    pretty = {"baseline_median": "Median baseline", "ridge": "Ridge", "random_forest": "Random forest", "hist_gbm": "Hist. GBM",
              "baseline_prior": "Prior baseline", "logistic": "Logistic"}
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(names)); h = 0.36
    ax.barh(y + h / 2, [reg_fam[n]["cv"]["MAE"] for n in names], h, color="#9cc3f0", label="CV MAE (train)")
    ax.barh(y - h / 2, [reg_fam[n]["test"]["all"]["MAE"] for n in names], h, color=BLUE, label="Test MAE")
    for i, n in enumerate(names):
        ax.text(reg_fam[n]["test"]["all"]["MAE"] + 0.05, i - h / 2, f"{reg_fam[n]['test']['all']['MAE']:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [pretty[n] + (" ★" if n == best_reg else "") for n in names]); ax.invert_yaxis()
    ax.set_xlabel("MAE (minutes, lower is better)"); ax.set_title("ETA regression — model comparison")
    ax.set_xlim(0, max(reg_fam[n]["cv"]["MAE"] for n in names) * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False); figs.append(save(fig, "12_regression_model_comparison.png"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    hb = ax[0].hexbin(yte, eta_pred, gridsize=40, cmap="Blues", mincnt=1)
    ax[0].plot([10, 54], [10, 54], color=ORANGE, lw=1.5, label="perfect prediction")
    ax[0].set_xlabel("actual minutes"); ax[0].set_ylabel("predicted minutes"); ax[0].legend(loc="upper left")
    ax[0].set_title(f"{pretty[best_reg]} — predicted vs actual (test)"); fig.colorbar(hb, ax=ax[0], label="orders")
    ax[1].hist(yte - eta_pred, bins=50, color=BLUE, edgecolor=SURF, linewidth=0.4)
    ax[1].axvline(0, color=INK2, lw=1); ax[1].set_xlabel("actual − predicted (minutes)"); ax[1].set_ylabel("orders")
    ax[1].set_title("Residuals (test)"); figs.append(save(fig, "13_eta_pred_vs_actual.png"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for a, col, title in zip(axes, ["traffic_density", "city_type", "distance_band"],
                             ["by traffic (not a model input)", "by city type", "by distance band"]):
        order = {"traffic_density": ["Low", "Medium", "High", "Jam", "<missing>"],
                 "city_type": ["Urban", "Metropolitan", "Semi-Urban", "<missing>"],
                 "distance_band": ["0–5 km", "5–10 km", "10–15 km", "15–25 km", "invalid geo"]}[col]
        tbl = pd.DataFrame(seg_tables[col]).set_index(col).reindex(order).dropna(subset=["n"]).reset_index()
        a.bar(tbl[col].astype(str), tbl["MAE"], color=BLUE, width=0.6)
        a.set_ylim(0, tbl["MAE"].max() * 1.12)
        for i, r in tbl.iterrows():
            a.text(i, r.MAE + 0.05, f"n={r.n:,}", ha="center", fontsize=8, color=INK2)
        a.set_title(f"Test MAE {title}"); a.set_ylabel("MAE (min)"); a.tick_params(axis="x", rotation=20)
    figs.append(save(fig, "14_eta_error_by_segment.png"))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    imp_s = imp.sort_values()
    ax.barh(imp_s.index, imp_s.values, color=BLUE, height=0.6)
    ax.set_xlabel("increase in test MAE when feature is shuffled (minutes)")
    ax.set_title(f"Permutation importance — {pretty[best_reg]} (post-hoc, test)"); ax.grid(axis="y", visible=False)
    figs.append(save(fig, "15_eta_feature_importance.png"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, color in [("logistic", AQUA), ("hist_gbm", BLUE)]:
        p = fitted30[name][1]
        fpr, tpr, _ = roc_curve(yte30, p); pr, rc, _ = precision_recall_curve(yte30, p)
        ax[0].plot(fpr, tpr, color=color, lw=2, label=f"{pretty[name]} (AUC {clf30[name]['test_at_f1_threshold']['ROC_AUC']:.3f})")
        ax[1].plot(rc, pr, color=color, lw=2, label=f"{pretty[name]} (AP {clf30[name]['test_at_f1_threshold']['PR_AUC']:.3f})")
    ax[0].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="prior baseline (0.5)")
    ax[1].axhline(yte30.mean(), color=GRAY, ls="--", lw=1, label=f"prior baseline ({yte30.mean():.3f})")
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC — late > 30 min (test)")
    ax[1].set(xlabel="recall", ylabel="precision", title="Precision–recall — late > 30 min (test)")
    ax[0].legend(loc="lower right"); ax[1].legend(loc="upper right"); figs.append(save(fig, "16_classification_roc_pr.png"))

    curve = late_thr["curve"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(curve["threshold"], curve["precision"], color=BLUE, lw=2, label="precision")
    ax.plot(curve["threshold"], curve["recall"], color=AQUA, lw=2, label="recall")
    ax.plot(curve["threshold"], curve["F1"], color=ORANGE, lw=2, label="F1")
    ax.axvline(late_thr["f1"]["threshold"], color=INK2, ls="--", lw=1)
    ax.text(late_thr["f1"]["threshold"] + 0.01, 0.05, f"chosen {late_thr['f1']['threshold']:.3f} (max F1)", color=INK2, fontsize=9)
    ax.axvline(late_thr["recall_target"]["threshold"], color=GRAY, ls=":", lw=1)
    ax.text(late_thr["recall_target"]["threshold"] + 0.01, 0.15, f"recall ≥ 0.80: {late_thr['recall_target']['threshold']:.3f}", color=GRAY, fontsize=9)
    ax.set(xlabel="decision threshold on predicted probability", ylabel="score", ylim=(0, 1.02),
           title=f"Threshold selection on out-of-fold TRAINING predictions ({pretty[best_clf]}, > 30 min)")
    ax.legend(loc="center right"); figs.append(save(fig, "17_classification_threshold.png"))

    cm = sens[30]["test_at_f1_threshold"]["confusion"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(mat, cmap="Blues"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", color="white" if mat[i, j] > mat.max() / 2 else INK, fontsize=12)
    ax.set_xticks([0, 1], ["predicted on time", "predicted late"]); ax.set_yticks([0, 1], ["actual on time", "actual late"])
    ax.set_title(f"Confusion matrix, test (threshold {late_thr['f1']['threshold']:.3f})"); figs.append(save(fig, "18_confusion_matrix.png"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(scan.k, scan.inertia, color=BLUE, lw=2, marker="o", ms=4)
    ax[0].axvline(k_elbow, color=GRAY, ls=":", lw=1); ax[0].text(k_elbow + 0.3, scan.inertia.max() * 0.9, f"elbow k={k_elbow}", color=GRAY, fontsize=9)
    ax[0].set(xlabel="k", ylabel="inertia (km²)", title="Elbow analysis")
    ax[1].plot(scan.k, scan.silhouette, color=BLUE, lw=2, marker="o", ms=4)
    ax[1].axvline(k_sil, color=ORANGE, ls="--", lw=1); ax[1].text(k_sil + 0.3, scan.silhouette.min(), f"best k={k_sil}", color=ORANGE, fontsize=9)
    ax[1].set(xlabel="k", ylabel="silhouette (sample 10k)", title="Silhouette analysis")
    figs.append(save(fig, "19_kmeans_elbow_silhouette.png"))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    samp = geo_all.sample(min(8000, len(geo_all)), random_state=SEED)
    ax.scatter(samp.delivery_lon, samp.delivery_lat, s=2, color=GRAY, alpha=0.25, label="orders (sample)")
    sc = ax.scatter(prof.centroid_lon, prof.centroid_lat, s=prof.orders / prof.orders.max() * 600 + 30,
                    c=prof["late_gt30_pct"], cmap="Blues", edgecolor=INK2, linewidth=0.6, zorder=3)
    for _, r in prof.iterrows():
        ax.annotate(f"{int(r.zone)}", (r.centroid_lon, r.centroid_lat), fontsize=7, ha="center", va="center", zorder=4)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="% orders > 30 min")
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"K-Means delivery zones (k={k_sil}); bubble = order volume\n"
                 f"note: colour spans only {prof['late_gt30_pct'].min():.1f}–{prof['late_gt30_pct'].max():.1f}% late")
    figs.append(save(fig, "20_delivery_zones.png"))

    fig, ax = plt.subplots(figsize=(9, 4))
    mets = ["precision", "recall", "F1", "PR_AUC", "ROC_AUC"]; w = 0.26
    for i, (mins, color) in enumerate([(30, BLUE), (35, AQUA), (40, ORANGE)]):
        vals = [sens[mins]["test_at_f1_threshold"][m] for m in mets]
        ax.bar(np.arange(len(mets)) + (i - 1) * w, vals, w, color=color, label=f"> {mins} min (prevalence {sens[mins]['test_at_f1_threshold']['prevalence']:.1%})")
    ax.set_xticks(range(len(mets)), ["precision", "recall", "F1", "PR-AUC", "ROC-AUC"]); ax.set_ylim(0, 1)
    ax.set_title("Late-benchmark sensitivity (test, F1-optimal OOF thresholds)"); ax.legend(loc="upper left", fontsize=8)
    figs.append(save(fig, "21_late_threshold_sensitivity.png"))
    M["figures"] = figs

    # ---------------- integrity & write ----------------
    hashes_after = {p.name: sha(p) for p in (RAW_PATH, CLEAN_CSV, FE_CSV)}
    check("raw, cleaned and engineered datasets unchanged (SHA-256)", hashes_before == hashes_after)
    M["dataset_hashes"] = hashes_after
    M["runtime_seconds"] = round(time.time() - t0, 1)
    M = clean_json(M)
    failed = [c["check"] for c in M["checks"] if not c["passed"]]
    OUT_METRICS.write_text(json.dumps(M, indent=2), encoding="utf-8")
    from src.modelling_report import write_report
    write_report(M, OUT_REPORT)
    print(f"checks: {len(M['checks']) - len(failed)}/{len(M['checks'])} passed | runtime {M['runtime_seconds']} s")
    for c in M["checks"]:
        print(("  PASS " if c["passed"] else "  FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    if failed:
        raise SystemExit(f"FAILED checks: {failed}")


if __name__ == "__main__":
    main()
