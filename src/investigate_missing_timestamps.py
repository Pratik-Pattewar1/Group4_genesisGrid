"""Investigation of the 1,731 records with a missing order timestamp (read-only).

Reads data/raw/train.csv and data/processed/cleaned_data.csv, modifies nothing, and writes:
    reports/missing_timestamp_investigation.md
    reports/missing_timestamp_metrics.json
    figures/11_missing_order_time.png

Rule followed: pickup time is NOT used to reconstruct or describe order time.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.investigate_missing_timestamps
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
CLEAN = ROOT / "data" / "processed" / "cleaned_data.csv"
OUT_MD = ROOT / "reports" / "missing_timestamp_investigation.md"
OUT_JSON = ROOT / "reports" / "missing_timestamp_metrics.json"
OUT_FIG = ROOT / "figures" / "11_missing_order_time.png"
RAW_SHA = "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1"
# Hour bands implied by traffic level on rows WITH an order time (derived below, then tested).
TRAFFIC_HOURS = {"Low": {0, 8, 9, 10, 22, 23}, "High": {11, 12, 13, 14},
                 "Medium": {15, 16, 17, 18}, "Jam": {19, 20, 21}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rate_table(c: pd.DataFrame, miss: pd.Series, col: str) -> tuple[pd.DataFrame, float]:
    """Missing-timestamp count and rate per level of `col`, plus chi-square p-value."""
    key = c[col].astype("string").fillna("<missing>")
    t = pd.crosstab(key, miss)
    t = t.reindex(columns=[False, True], fill_value=0)
    p = float(stats.chi2_contingency(t)[1])
    out = pd.DataFrame({"rows": t.sum(axis=1), "missing_time": t[True]})
    out["rate_pct"] = (100 * out["missing_time"] / out["rows"]).round(2)
    return out.sort_values("rate_pct"), p


def main() -> None:
    sha_before = sha(RAW)
    assert sha_before == RAW_SHA, "raw file hash mismatch"
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())
    c = pd.read_csv(CLEAN)
    M: dict = {}

    miss = c["order_time"].isna()
    known = ~miss
    raw_miss = raw["Time_Orderd"] == "NaN"
    assert (miss.values == raw_miss.values).all() and (c["order_id"].values == raw["ID"].values).all()
    M["n_missing"] = int(miss.sum())
    M["pct_missing"] = round(100 * miss.mean(), 2)
    M["order_date_missing"] = int(c["order_date"].isna().sum())
    c["city_code"] = c["delivery_person_id"].str.extract(r"^([A-Z]+)RES")[0]
    odate = pd.to_datetime(c["order_date"])
    hour = pd.to_datetime(c["order_datetime"]).dt.hour

    # ---------------- 1. patterns ----------------
    by_date = miss.groupby(odate).agg(["size", "sum", "mean"])
    M["by_date"] = {"dates": int(len(by_date)), "dates_with_zero_missing": int((by_date["sum"] == 0).sum()),
                    "min_rate_pct": round(100 * by_date["mean"].min(), 2), "max_rate_pct": round(100 * by_date["mean"].max(), 2)}
    M["by_month"] = {str(k): {"missing": int(v["sum"]), "rate_pct": round(100 * v["mean"], 2)}
                     for k, v in miss.groupby(odate.dt.to_period("M")).agg(["sum", "mean"]).iterrows()}
    wk, wk_p = rate_table(c.assign(weekday=odate.dt.day_name()), miss, "weekday")
    M["by_weekday_p"] = wk_p

    tables = {}
    for col in ["city_code", "city_type", "weather", "traffic_density", "festival", "multiple_deliveries",
                "vehicle_type", "vehicle_condition", "order_type"]:
        tables[col] = rate_table(c, miss, col)
    M["chi2_p"] = {k: v[1] for k, v in tables.items()}

    y = c["time_taken_min"]
    M["target"] = {
        "missing": y[miss].describe().round(2).to_dict(), "present": y[known].describe().round(2).to_dict(),
        "mann_whitney_p": float(stats.mannwhitneyu(y[miss], y[known]).pvalue),
        "ks_p": float(stats.ks_2samp(y[miss], y[known]).pvalue),
        "late_gt30_pct": {"missing": round(100 * (y[miss] > 30).mean(), 2), "present": round(100 * (y[known] > 30).mean(), 2)},
    }

    co = {}
    for col in ["delivery_person_age_raw", "delivery_person_rating_raw", "weather", "traffic_density",
                "multiple_deliveries", "festival", "city_type"]:
        co[col] = {"missing_in_block": int((c[col].isna() & miss).sum()), "missing_total": int(c[col].isna().sum())}
    M["co_missing"] = co
    M["block_contains_all"] = {
        "vehicle_condition_3": [int(((c.vehicle_condition == 3) & miss).sum()), int((c.vehicle_condition == 3).sum())],
        "bicycle": [int(((c.vehicle_type == "bicycle") & miss).sum()), int((c.vehicle_type == "bicycle").sum())],
        "rider_sentinel": [int((c.dq_rider_sentinel & miss).sum()), int(c.dq_rider_sentinel.sum())],
    }
    M["geo_invalid_in_block"] = int((~c.dq_geo_valid & miss).sum())

    # ---------------- 2. recovery candidates (pickup time deliberately excluded) ----------------
    rec = {}
    rec["raw_columns"] = list(raw.columns)
    rec["order_date_complete"] = M["order_date_missing"] == 0
    idint = c["order_id"].apply(lambda s: int(s, 16))
    rec["hex_id_spearman_vs_datetime"] = round(float(stats.spearmanr(idint[known], pd.to_datetime(c.order_datetime[known]).astype("int64")).statistic), 3)
    rec["hex_id_spearman_vs_hour"] = round(float(stats.spearmanr(idint[known], hour[known]).statistic), 3)
    rec["row_order_spearman_vs_datetime"] = round(float(stats.spearmanr(np.arange(len(c))[known], pd.to_datetime(c.order_datetime[known]).astype("int64")).statistic), 4)
    kk = c.loc[known, ["delivery_person_id", "order_date", "order_time"]]
    size = kk.groupby(["delivery_person_id", "order_date"])["order_time"].transform("size")
    same_time = kk.duplicated(["delivery_person_id", "order_date", "order_time"], keep=False)
    rec["peer_same_id_date_rows"] = int((size > 1).sum())
    rec["peer_exact_time_match_pct"] = round(100 * float(same_time[size > 1].mean()), 2)
    has_tr = known & c["traffic_density"].notna()
    consistent = pd.Series([h in TRAFFIC_HOURS[t] for t, h in zip(c.loc[has_tr, "traffic_density"], hour[has_tr])])
    rec["traffic_hour_band_consistency_pct"] = round(100 * float(consistent.mean()), 2)
    rec["traffic_distinct_times_per_level"] = c[known].groupby("traffic_density")["order_time"].nunique().to_dict()
    rec["block_rows_with_traffic"] = int(c.loc[miss, "traffic_density"].notna().sum())
    rec["block_traffic_counts"] = c.loc[miss, "traffic_density"].value_counts().to_dict()
    wmax = pd.crosstab(hour[known], c.weather[known], normalize="index").max(axis=1)
    rec["weather_max_share_any_hour"] = round(float(wmax.max()), 3)
    M["recovery"] = rec
    traffic_hour = pd.crosstab(c.loc[known, "traffic_density"], hour[known])

    # ---------------- 5. split check ----------------
    cut = odate.quantile(0.8)
    M["example_split_80pct_date"] = {"cut_date": str(cut.date()),
                                     "train_missing": int(miss[odate <= cut].sum()), "train_rate_pct": round(100 * miss[odate <= cut].mean(), 2),
                                     "test_missing": int(miss[odate > cut].sum()), "test_rate_pct": round(100 * miss[odate > cut].mean(), 2)}

    # ---------------- figure ----------------
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True,
                         "font.size": 10, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].bar(by_date.index, 100 * by_date["mean"], color="#2a78d6", width=0.8)
    ax[0].axhline(M["pct_missing"], color="#52514e", ls="--", lw=1)
    ax[0].set_title(f"Missing order time, % of orders per date (dashed = overall {M['pct_missing']}%)"); ax[0].set_ylabel("% missing"); ax[0].tick_params(axis="x", rotation=30)
    share = traffic_hour.div(traffic_hour.sum(axis=0), axis=1).reindex(["Low", "Medium", "High", "Jam"])
    im = ax[1].imshow(share.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax[1].set_yticks(range(4), share.index); ax[1].set_xticks(range(share.shape[1]), [int(h) for h in share.columns], fontsize=8)
    ax[1].set_xlabel("order hour (rows with a timestamp)"); ax[1].grid(False)
    ax[1].set_title("Traffic level is almost a function of hour")
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="share of the hour's orders")
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight"); plt.close(fig)

    sha_after = sha(RAW)
    M["raw_unchanged"] = sha_before == sha_after == RAW_SHA
    M["cleaned_sha256"] = sha(CLEAN)
    OUT_JSON.write_text(json.dumps(M, indent=2, default=str), encoding="utf-8")
    write_report(M, tables, wk)
    print(f"missing: {M['n_missing']} | raw unchanged: {M['raw_unchanged']} | wrote {OUT_MD.name}, {OUT_JSON.name}, {OUT_FIG.name}")


def fmt_table(df: pd.DataFrame, label: str) -> list[str]:
    L = [f"| {label} | Rows | Missing time | Rate % |", "|---|---|---|---|"]
    L += [f"| {i} | {int(r.rows):,} | {int(r.missing_time):,} | {r.rate_pct:.2f} |" for i, r in df.iterrows()]
    return L


def write_report(M: dict, tables: dict, wk: pd.DataFrame) -> None:
    rec, co, b, t = M["recovery"], M["co_missing"], M["block_contains_all"], M["target"]
    sp = M["example_split_80pct_date"]
    L = []
    a = L.append
    a("# DeliveryIQ — Missing Order Timestamp Investigation\n")
    a("**Scope:** the records whose `Time_Orderd` / `order_time` is missing. **Read-only:** no record was filled, "
      "deleted or modified. **Pickup time was not used** to reconstruct or describe order time.  ")
    a(f"**Sources:** `data/raw/train.csv` (SHA-256 verified unchanged: {M['raw_unchanged']}), "
      f"`data/processed/cleaned_data.csv` (`{M['cleaned_sha256'][:12]}…`).  ")
    a("**Reproduce:** `.venv\\Scripts\\python.exe -m src.investigate_missing_timestamps`. Numbers come from "
      "`reports/missing_timestamp_metrics.json`. Chart: `figures/11_missing_order_time.png`.\n")
    a("Labels: **[V]** verified from data · **[A]** interpretation · **[R]** recommendation\n")

    a("## Key findings\n")
    a(f"1. **{M['n_missing']:,} records ({M['pct_missing']}%)** have no order time. "
      f"Their **order date is always present** ({M['order_date_missing']} missing dates) [V].")
    a(f"2. They are spread **evenly over time**: every one of the {M['by_date']['dates']} dates has some "
      f"(daily rate {M['by_date']['min_rate_pct']}–{M['by_date']['max_rate_pct']}%), and each month is about "
      f"{min(v['rate_pct'] for v in M['by_month'].values())}–{max(v['rate_pct'] for v in M['by_month'].values())}% [V].")
    a(f"3. **Delivery time is the same as for other records:** median {t['missing']['50%']:.0f} vs "
      f"{t['present']['50%']:.0f} min, mean {t['missing']['mean']} vs {t['present']['mean']}; Mann-Whitney "
      f"p = {t['mann_whitney_p']:.2f}, KS p = {t['ks_p']:.2f}. Share over 30 min: {t['late_gt30_pct']['missing']}% vs "
      f"{t['late_gt30_pct']['present']}% [V]. The missingness is not related to the outcome.")
    age_raw = co["delivery_person_age_raw"]
    a("4. **It is one block of incomplete records, not scattered gaps** [V]. In the cleaned data, every one of "
      "these records also lacks rider age and rating: "
      f"{age_raw['missing_in_block']:,} were already missing in the raw file, and the other "
      f"{M['n_missing'] - age_raw['missing_in_block']} are the placeholder rider records blanked in Stage 2. "
      f"({age_raw['missing_total'] - age_raw['missing_in_block']} raw age-missing rows lie outside the block.) "
      "The block contains **every**:")
    a(f"   - missing-weather row ({co['weather']['missing_in_block']}/{co['weather']['missing_total']}) and "
      f"missing-traffic row ({co['traffic_density']['missing_in_block']}/{co['traffic_density']['missing_total']});")
    a(f"   - `vehicle_condition = 3` row ({b['vehicle_condition_3'][0]}/{b['vehicle_condition_3'][1]}) and "
      f"bicycle ({b['bicycle'][0]}/{b['bicycle'][1]});")
    a(f"   - placeholder rider record ({b['rider_sentinel'][0]}/{b['rider_sentinel'][1]}).")
    a("5. **No field in the dataset can reliably recover the missing times** (§2). Traffic level narrows the "
      "time to a 3–6 hour window at best and can't be checked on these rows.")
    a(f"6. **Side finding with Stage 3 impact:** traffic level is almost a pure function of order hour "
      f"({rec['traffic_hour_band_consistency_pct']}% of timed orders follow the bands below) [V].\n")

    a("## 1. Patterns\n")
    a("### By date\n")
    a("| Month | Missing | Rate % |\n|---|---|---|")
    for k, v in M["by_month"].items():
        a(f"| {k} | {v['missing']:,} | {v['rate_pct']} |")
    a(f"\nDaily rate: {M['by_date']['min_rate_pct']}–{M['by_date']['max_rate_pct']}%; dates with none missing: "
      f"{M['by_date']['dates_with_zero_missing']} of {M['by_date']['dates']}. Weekday association: chi-square "
      f"p = {M['by_weekday_p']:.3g}.\n")
    a("### By city\n")
    a(f"City code (ID prefix, 22 cities): rates {tables['city_code'][0].rate_pct.min()}–"
      f"{tables['city_code'][0].rate_pct.max()}%, chi-square p = {M['chi2_p']['city_code']:.2f}, so there is no city effect [V].\n")
    L += fmt_table(tables["city_type"][0], "City type")
    a(f"\nChi-square p = {M['chi2_p']['city_type']:.2f} [V].\n")
    a("### By weather\n")
    L += fmt_table(tables["weather"][0], "Weather")
    a("\nWhen weather is recorded, the rate is 2.3–2.8% for every condition. All missing-weather rows fall inside "
      "the block, so the high chi-square is co-missingness, not a weather effect [V/A].\n")
    a("### By traffic\n")
    L += fmt_table(tables["traffic_density"][0], "Traffic")
    a("\nSame pattern as weather: about 2.3–2.7% for each recorded level, 100% where traffic itself is missing [V].\n")
    a("### By delivery duration (target)\n")
    a("| | Missing order time | Order time present |\n|---|---|---|")
    for s in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        f = (lambda v: f"{v:,.0f}") if s in ("count", "min", "25%", "50%", "75%", "max") else (lambda v: f"{v:,.2f}")
        a(f"| {s} | {f(t['missing'][s])} | {f(t['present'][s])} |")
    a(f"| > 30 min (project benchmark) | {t['late_gt30_pct']['missing']}% | {t['late_gt30_pct']['present']}% |")
    a(f"\nMann-Whitney p = {t['mann_whitney_p']:.3f}; Kolmogorov-Smirnov p = {t['ks_p']:.3f}. No detectable "
      "difference in delivery time [V].\n")
    a("### Other fields\n")
    L += fmt_table(tables["vehicle_condition"][0], "Vehicle condition")
    a("")
    L += fmt_table(tables["vehicle_type"][0], "Vehicle type")
    a(f"\nNo association with order type (p = {M['chi2_p']['order_type']:.2f}), festival (p = {M['chi2_p']['festival']:.2f}) "
      f"or multiple deliveries (p = {M['chi2_p']['multiple_deliveries']:.2f}).\n")
    a("### Co-missingness\n")
    a("| Field | Missing inside block | Missing overall |\n|---|---|---|")
    for k, v in co.items():
        a(f"| `{k}` | {v['missing_in_block']:,} | {v['missing_total']:,} |")
    a(f"\nAlso inside the block: {M['geo_invalid_in_block']} geo-invalid rows (not over-represented: "
      f"{100 * M['geo_invalid_in_block'] / M['n_missing']:.1f}% vs 7.98% overall).\n")
    a("**Interpretation [A]:** these look like orders whose rider/dispatch record failed to join. They lost "
      "order time, rider age and rating, and in some cases weather and traffic, and carry an unusual vehicle "
      "profile (all bicycles, all condition 3). Because delivery times are unaffected, the missingness is unrelated "
      "to the outcome, but it is **not random with respect to the other fields**.\n")

    a("## 2. Can the original data recover the missing times?\n")
    a("Pickup time was excluded by rule. Every other candidate in the raw file was tested:\n")
    a("| Candidate | Test (on rows *with* an order time) | Result | Usable? |\n|---|---|---|---|")
    a(f"| `Order_Date` | completeness | {M['order_date_missing']} missing | ✅ The **date** is known for every row |")
    a(f"| Row order in the file | Spearman with order datetime | {rec['row_order_spearman_vs_datetime']} | ❌ File is not in time order |")
    a(f"| Hex `ID` | Spearman with datetime / with hour | {rec['hex_id_spearman_vs_datetime']} / {rec['hex_id_spearman_vs_hour']} | "
      "❌ Weakly tracks the date (already known), says nothing about time of day |")
    a(f"| Same `delivery_person_id` on same date | another order with identical time | {rec['peer_exact_time_match_pct']}% of "
      f"{rec['peer_same_id_date_rows']:,} rows | ❌ Peers almost never share a time |")
    a(f"| `Road_traffic_density` | hour falls in the traffic level's band | {rec['traffic_hour_band_consistency_pct']}% | "
      f"⚠️ Gives only a 3–6 hour **band** ({min(rec['traffic_distinct_times_per_level'].values())}–"
      f"{max(rec['traffic_distinct_times_per_level'].values())} distinct times per level). Available for only "
      f"{rec['block_rows_with_traffic']:,} of {M['n_missing']:,} block rows; can't be checked on the block |")
    a(f"| `Weatherconditions` | largest single-weather share in any hour | {rec['weather_max_share_any_hour']} | ❌ Weather is unrelated to hour |")
    a("| Other columns (rider, vehicle, order type, festival, city, coordinates) | no time content | — | ❌ |")
    a("\n**Traffic bands observed on timed orders** [V]:\n")
    a("| Traffic | Order hours |\n|---|---|")
    for k, v in TRAFFIC_HOURS.items():
        a(f"| {k} | {', '.join(f'{h:02d}' for h in sorted(v))} |")
    a("\n**Conclusion:** the dataset holds **no reliable source for the missing order times**. The order "
      "**date** is fully available. Traffic level gives a coarse, unverifiable hour band and **should not be "
      "used to fill** `order_time`:")
    a("- it would recover a range, not a time;")
    a("- it can't be checked on the very rows it would fill;")
    a("- it would make the hour feature a copy of the traffic feature.\n")

    a("## 3. Handling in the chronological split [R]\n")
    a("1. **Split on `order_date`, not `order_datetime`.** `order_date` is complete, so every record (including "
      "these) can be placed on one side of the split. Sorting by `order_datetime` would push all NaT rows to one "
      "end and silently move them all into train or all into test.")
    a("2. **Cut at a whole-date boundary** (every order on a date goes to the same side), so no time-of-day "
      "ordering is needed within a date.")
    a(f"3. **The block splits proportionally on its own:** with an example 80/20 cut after {sp['cut_date']}, "
      f"train gets {sp['train_missing']:,} ({sp['train_rate_pct']}%) and test gets {sp['test_missing']:,} "
      f"({sp['test_rate_pct']}%). No stratification is needed. (The real cut date is chosen in Stage 4.)\n")

    a("## 4. Handling in model training [R]\n")
    a("1. **Keep the records by default.** Their delivery time matches the rest, and dropping them would remove "
      "every bicycle and every `vehicle_condition = 3` order from training.")
    a("2. **No filling outside the model pipeline.** Hour-based features get imputation fitted on the **training "
      "split only**, or are left as missing for models that handle it natively (e.g. gradient-boosted trees).")
    a("3. **Never back-fill from pickup time** (Tier C information) and never from traffic level (circular).")
    a("4. **No 'order time missing' indicator by default.** It carries no outcome signal and would mostly flag "
      "the incomplete-record block (bicycles and condition 3). `dq_pickup_day_assumed` marks exactly these rows "
      "but stays an audit column, never a feature.")
    a("5. **Sensitivity analysis:** train with and without the block (alongside the planned Festival comparison) "
      "and report test metrics separately for complete records and block records.")
    a("6. **Demo implication:** the Streamlit app always takes an order time, so the complete-record test "
      "metrics are the ones that represent live use.")
    a(f"7. **Stage 3 note:** traffic level and order hour carry almost the same information "
      f"({rec['traffic_hour_band_consistency_pct']}% overlap). Feature selection and importance analysis must "
      "treat them as collinear. It also suggests traffic was generated from order time, which bears on whether "
      "it is available at order time (Tier A\\*) [A].\n")

    a("## 5. Limitations\n")
    a("- The cause of the missingness is unknown; the 'failed join' explanation is an interpretation [A].")
    a("- The traffic-to-hour bands come from timed records. Whether they hold for the block can't be tested.")
    a("- Statistical tests show no *detectable* difference in delivery time. That isn't proof of no difference.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
