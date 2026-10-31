"""DeliveryIQ — Stage 5 analyses required by the official challenge guide (Hackathon_Challenge_Guide.pdf).

Fills the gaps found when Stages 1–5 were checked against the PDF, without retraining the primary models:
    * late rate by hour of day and day of week (PS3, visual set)
    * weather × traffic late-rate matrix — "does weather add delay beyond traffic?" (§8)
    * delivery time explained by distance alone (§8)                  [additional baseline, train-fitted]
    * mean-delivery-time baseline (PS4 names the average as the naive baseline)
    * rider experience / rating vs delay on similar routes (PS3, §8)  [stratified, descriptive]
    * order-level rider-attribute profiles with K-Means (PS7 alternative — NOT individual rider segments)
    * zone vs network benchmark table with per-zone significance tests (PS6, §8)
    * PR / ROC curves with the chosen threshold marked (visual set)

All observed comparisons are associations, not causal effects. Test data is used only to *report*
fixed models, never to choose anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve, roc_curve, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clustering import project_km
from src.evaluation import regression_report

SEED = 42
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]
WEATHER = ["Sunny", "Windy", "Stormy", "Sandstorms", "Cloudy", "Fog"]


def _late(df):
    return (df["time_taken_min"] > 30).astype(int)


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.assign(late=_late(df)).groupby(col)
    return pd.DataFrame({"orders": g.size(), "mean_eta": g["time_taken_min"].mean(),
                         "late_pct": 100 * g["late"].mean()}).reset_index()


def stratified_diff(df: pd.DataFrame, group_col: str, hi: str, lo: str, strata: list[str]) -> dict:
    """Mean-ETA and late-rate difference hi − lo inside each stratum, weighted by the smaller group size.
    Compares like with like ('similar routes'); still an association, not a causal effect."""
    d = df.assign(late=_late(df)).dropna(subset=[group_col] + strata)
    rows, w_tot, eta_sum, late_sum = [], 0, 0.0, 0.0
    for key, s in d.groupby(strata):
        a, b = s[s[group_col] == hi], s[s[group_col] == lo]
        if len(a) >= 30 and len(b) >= 30:
            w = min(len(a), len(b))
            de, dl = a["time_taken_min"].mean() - b["time_taken_min"].mean(), 100 * (a["late"].mean() - b["late"].mean())
            rows.append(w); w_tot += w; eta_sum += w * de; late_sum += w * dl
    return {"strata_used": len(rows), "weight": int(w_tot),
            "eta_diff_min": eta_sum / w_tot if w_tot else float("nan"),
            "late_diff_pp": late_sum / w_tot if w_tot else float("nan")}


def compute(fe, train, test, Xtr, ytr, Xte, yte, p30, threshold, km_art, M, plt, colors, save) -> list[str]:
    """Run all gap analyses; write results into ``M['challenge']``; return the list of figures produced."""
    BLUE, ORANGE, AQUA, GRAY, INK, INK2 = colors
    C: dict = {}
    figs: list[str] = []
    net_late = float(100 * _late(fe).mean())
    net_eta = float(fe["time_taken_min"].mean())
    C["network"] = {"late_pct": net_late, "mean_eta": net_eta, "orders": len(fe)}

    # ---- 1. hour of day & day of week ---------------------------------------------------
    timed = fe[fe["order_hour"].notna()].assign(order_hour=lambda d: d["order_hour"].astype(int))
    hr = rate_table(timed, "order_hour")
    hr["gap_vs_network_pp"] = hr["late_pct"] - net_late
    C["by_hour"] = hr.to_dict("records")
    top = hr.sort_values("late_pct", ascending=False).head(3)
    C["peak_hours"] = top["order_hour"].astype(int).tolist()
    C["hours_above_network"] = hr.loc[hr["late_pct"] > net_late, "order_hour"].astype(int).tolist()
    dw = rate_table(fe, "order_day_of_week")
    dw["day"] = [DOW[int(i)] for i in dw["order_day_of_week"]]
    C["by_day_of_week"] = dw.to_dict("records")
    dpw = fe.groupby("order_day_of_week")["order_date"].nunique()
    C["dates_per_weekday"] = [int(dpw.min()), int(dpw.max())]
    C["day_of_week_chi2_p"] = float(stats.chi2_contingency(pd.crosstab(fe["order_day_of_week"], _late(fe)))[1])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax[0].bar(hr["order_hour"].astype(str), hr["orders"], color=GRAY, width=0.7)
    ax[0].set(ylabel="orders", title="Order volume and late rate by hour of order (hours 01–07 have no orders)")
    cols = [ORANGE if h in C["peak_hours"] else BLUE for h in hr["order_hour"]]
    ax[1].bar(hr["order_hour"].astype(str), hr["late_pct"], color=cols, width=0.7)
    ax[1].axhline(net_late, color=INK2, ls="--", lw=1)
    ax[1].text(len(hr) - 0.5, net_late + 1.5, f"network {net_late:.1f}%", ha="right", fontsize=9, color=INK2)
    for i, v in enumerate(hr["late_pct"]):
        ax[1].text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax[1].set(xlabel="hour of order", ylabel="% late (> 30 min)")
    figs.append(save(fig, "26_late_rate_by_hour.png"))

    # ---- 2. weather × traffic --------------------------------------------------------------
    wt = fe.dropna(subset=["weather", "traffic_density"]).assign(late=lambda d: _late(d))
    piv = wt.pivot_table(index="weather", columns="traffic_density", values="late", aggfunc="mean").reindex(index=WEATHER, columns=TRAFFIC) * 100
    cnt = wt.pivot_table(index="weather", columns="traffic_density", values="late", aggfunc="size").reindex(index=WEATHER, columns=TRAFFIC)
    spread = {t: float(piv[t].max() - piv[t].min()) for t in TRAFFIC}
    traffic_spread = {w: float(piv.loc[w].max() - piv.loc[w].min()) for w in WEATHER}
    sunny_fog = {t: float(piv.loc["Fog", t] - piv.loc["Sunny", t]) for t in TRAFFIC}
    C["weather_traffic"] = {"late_pct": piv.round(2).to_dict(), "orders": cnt.to_dict(),
                            "weather_spread_within_traffic_pp": spread, "traffic_spread_within_weather_pp": traffic_spread,
                            "fog_minus_sunny_within_traffic_pp": sunny_fog}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(WEATHER)):
        for j in range(len(TRAFFIC)):
            v, n = piv.values[i, j], cnt.values[i, j]
            ax.text(j, i, f"{v:.0f}%\nn={int(n):,}", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(len(TRAFFIC)), TRAFFIC); ax.set_yticks(range(len(WEATHER)), WEATHER); ax.grid(False)
    ax.set(xlabel="traffic density", ylabel="weather", title="Late rate (> 30 min) by weather × traffic")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% late")
    figs.append(save(fig, "27_weather_traffic_late_rate.png"))

    # ---- 3. distance alone & mean baseline (train-fitted, test-reported) --------------------
    dist_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("lr", LinearRegression())])
    dist_pipe.fit(Xtr[["distance_km"]], ytr)
    C["distance_only_model"] = {**regression_report(yte, dist_pipe.predict(Xte[["distance_km"]])),
                                "coef_min_per_km": float(dist_pipe.named_steps["lr"].coef_[0]),
                                "note": "additional analysis: linear regression on distance only, fitted on training rows"}
    v = fe.dropna(subset=["distance_km"])
    C["distance_pearson_r2_all_valid"] = float(np.corrcoef(v["distance_km"], v["time_taken_min"])[0, 1] ** 2)
    C["mean_baseline"] = {**regression_report(yte, np.full(len(yte), float(ytr.mean()))), "train_mean": float(ytr.mean())}

    # ---- 4. rider experience / rating vs delay ------------------------------------------------
    r = fe.copy()
    r["age_band"] = pd.cut(r["delivery_person_age"], [19, 24, 29, 34, 39], labels=["20–24", "25–29", "30–34", "35–39"]).astype("string")
    r["rating_band"] = pd.cut(r["delivery_person_rating"], [0, 4.45, 4.75, 5.0], labels=["< 4.5", "4.5–4.7", "4.8–5.0"]).astype("string")
    r["distance_band"] = pd.cut(r["distance_km"], [0, 5, 10, 15, 25], labels=["0–5", "5–10", "10–15", "15–25"]).astype("string")
    C["rider_age_band"] = rate_table(r.dropna(subset=["age_band"]), "age_band").to_dict("records")
    C["rider_rating_band"] = rate_table(r.dropna(subset=["rating_band"]), "rating_band").to_dict("records")
    strata = ["distance_band", "traffic_density"]
    C["rating_hi_vs_lo_similar_routes"] = stratified_diff(r, "rating_band", "4.8–5.0", "< 4.5", strata)
    C["age_old_vs_young_similar_routes"] = stratified_diff(r, "age_band", "35–39", "20–24", strata)
    C["rider_note"] = ("age is the only experience proxy (no tenure field); rating may include feedback on the same order; "
                       "rows are orders, not riders; Tier B data, excluded from the placement models")

    # ---- 5. order-level rider-attribute profiles (PS7 alternative) ----------------------------
    prof_df = fe.dropna(subset=["delivery_person_age", "delivery_person_rating"]).reset_index(drop=True)
    Z = StandardScaler().fit_transform(prof_df[["delivery_person_age", "delivery_person_rating"]])
    scan = []
    for k in range(2, 9):
        km_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        scan.append({"k": k, "inertia": float(km_k.inertia_),
                     "silhouette": float(silhouette_score(Z, km_k.labels_, sample_size=10_000, random_state=SEED))})
    scan = pd.DataFrame(scan)
    k_best = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    km_r = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit(Z)
    prof_df = prof_df.assign(profile=km_r.labels_, late=_late(prof_df))
    g = prof_df.groupby("profile")
    pr = pd.DataFrame({"orders": g.size(), "mean_age": g["delivery_person_age"].mean(), "mean_rating": g["delivery_person_rating"].mean(),
                       "mean_eta": g["time_taken_min"].mean(), "late_pct": 100 * g["late"].mean(),
                       "multi_delivery_mean": g["multiple_deliveries"].mean(),
                       "motorcycle_pct": 100 * g["vehicle_type"].apply(lambda s: (s == "motorcycle").mean())}).reset_index()
    a_mid, r_mid = prof_df["delivery_person_age"].mean(), prof_df["delivery_person_rating"].mean()
    names = []
    for _, row in pr.iterrows():
        nm = f"{'Older' if row.mean_age >= a_mid else 'Younger'}, {'higher' if row.mean_rating >= r_mid else 'lower'}-rated"
        names.append(nm if nm not in names else f"{nm} ({int(row.profile)})")
    pr["name"] = names
    C["rider_profiles"] = {"label": "order-level rider-attribute profiles — NOT individual rider segments",
                           "features": ["delivery_person_age", "delivery_person_rating"], "rows": len(prof_df),
                           "scan": scan.to_dict("records"), "k_selected": k_best,
                           "silhouette": float(scan.loc[scan.k == k_best, "silhouette"].iloc[0]),
                           "profiles": pr.to_dict("records")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax[0].plot(scan.k, scan.silhouette, color=BLUE, lw=2, marker="o", ms=5)
    ax[0].axvline(k_best, color=ORANGE, ls="--", lw=1)
    ax[0].set(xlabel="k", ylabel="silhouette (sample 10k)", title=f"Rider-attribute profiles: k = {k_best} chosen")
    sc = ax[1].scatter(pr.mean_rating, pr.mean_age, s=pr.orders / pr.orders.max() * 2200 + 80, c=pr.late_pct, cmap="Blues",
                       vmin=0, vmax=max(60, pr.late_pct.max()), edgecolor=INK2)
    for _, row in pr.iterrows():
        ax[1].annotate(f"{row['name']}\n{row.late_pct:.0f}% late · {row.mean_eta:.1f} min · {row.orders:,} orders",
                       (row.mean_rating, row.mean_age), xytext=(0, -34), textcoords="offset points",
                       fontsize=8, ha="center", va="top", color=INK)
    ax[1].margins(x=0.25)
    ax[1].set_ylim(pr.mean_age.min() - 5, pr.mean_age.max() + 2.5)
    ax[1].set(xlabel="mean rider rating", ylabel="mean rider age (experience proxy)",
              title="Profiles of ORDERS by rider attributes (not individual riders)")
    fig.colorbar(sc, ax=ax[1], shrink=0.8, label="% late (> 30 min)")
    figs.append(save(fig, "28_rider_attribute_profiles.png"))

    # ---- 6. zone vs network -----------------------------------------------------------------
    geo = fe[fe["dq_geo_valid"]].reset_index(drop=True)
    zone = km_art["kmeans"].predict(project_km(geo["delivery_lat"], geo["delivery_lon"], km_art["projection_ref_lat"]))
    geo = geo.assign(zone=zone, late=_late(geo))
    gl, ge = float(100 * geo["late"].mean()), float(geo["time_taken_min"].mean())
    rows = []
    for z, s in geo.groupby("zone"):
        rest = geo[geo["zone"] != z]
        p = stats.chi2_contingency(pd.crosstab(geo["zone"] == z, geo["late"]))[1]
        t = stats.ttest_ind(s["time_taken_min"], rest["time_taken_min"], equal_var=False).pvalue
        rows.append({"zone": int(z), "city": s["city_code"].mode().iloc[0], "orders": len(s), "demand_share_pct": 100 * len(s) / len(geo),
                     "mean_eta": s["time_taken_min"].mean(), "eta_vs_network_min": s["time_taken_min"].mean() - ge,
                     "late_pct": 100 * s["late"].mean(), "late_vs_network_pp": 100 * s["late"].mean() - gl,
                     "late_p": float(p), "eta_p": float(t)})
    zt = pd.DataFrame(rows)
    m = len(zt)
    zt["late_sig_bonferroni"] = zt["late_p"] < 0.05 / m
    zt["eta_sig_bonferroni"] = zt["eta_p"] < 0.05 / m
    med = zt["orders"].median()
    zt["high_demand"] = zt["orders"] >= med
    zt["above_benchmark"] = (zt["eta_vs_network_min"] > 0) & (zt["late_vs_network_pp"] > 0)
    zt["candidate_flag"] = zt["high_demand"] & zt["above_benchmark"]
    C["zones_vs_network"] = {"network_valid_mean_eta": ge, "network_valid_late_pct": gl, "median_zone_orders": float(med),
                             "table": zt.sort_values("late_vs_network_pp", ascending=False).to_dict("records"),
                             "n_significant_late_bonferroni": int(zt["late_sig_bonferroni"].sum()),
                             "n_significant_eta_bonferroni": int(zt["eta_sig_bonferroni"].sum()),
                             "candidates": zt.loc[zt["candidate_flag"], "city"].tolist(),
                             "candidates_significant": zt.loc[zt["candidate_flag"] & (zt["late_sig_bonferroni"] | zt["eta_sig_bonferroni"]), "city"].tolist()}

    # ---- 7. PR / ROC with the chosen threshold marked --------------------------------------
    yb = (yte > 30).astype(int).to_numpy()
    pr_p, pr_r, pr_t = precision_recall_curve(yb, p30)
    fpr, tpr, roc_t = roc_curve(yb, p30)
    pred = p30 >= threshold
    pt_p, pt_r = float(yb[pred].mean()), float(pred[yb == 1].mean())
    pt_fpr = float(pred[yb == 0].mean())
    C["threshold_point_test"] = {"threshold": float(threshold), "precision": pt_p, "recall": pt_r, "fpr": pt_fpr}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(pr_r, pr_p, color=BLUE, lw=2, label="gradient boosting (test)")
    ax[0].axhline(yb.mean(), color=GRAY, ls="--", lw=1, label=f"no-skill ({yb.mean():.3f})")
    ax[0].scatter([pt_r], [pt_p], s=90, color=ORANGE, zorder=3, label=f"chosen threshold {threshold:.3f}")
    ax[0].annotate(f"recall {pt_r:.3f}\nprecision {pt_p:.3f}", (pt_r, pt_p), xytext=(-150, 45), textcoords="offset points",
                   fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2))
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision–recall, late > 30 min (test)", ylim=(0, 1.02))
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(fpr, tpr, color=BLUE, lw=2, label="gradient boosting (test)")
    ax[1].plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="no-skill")
    ax[1].scatter([pt_fpr], [pt_r], s=90, color=ORANGE, zorder=3, label=f"chosen threshold {threshold:.3f}")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC, late > 30 min (test)")
    ax[1].legend(loc="lower right", fontsize=8)
    figs.append(save(fig, "29_threshold_on_pr_roc.png"))

    # ---- 8. class balance (for the imbalance statement) ---------------------------------------
    C["class_balance"] = {"train_late_pct": float(100 * (ytr > 30).mean()), "test_late_pct": float(100 * yb.mean())}
    M["challenge"] = C
    return figs
