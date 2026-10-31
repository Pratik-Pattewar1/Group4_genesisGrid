"""Independent final audit of data/processed/cleaned_data.csv against data/raw/train.csv.

Deliberately does NOT import src.data_cleaning: every expectation is recomputed from the
raw file with plain pandas, so a bug in the cleaning pipeline cannot hide itself.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.audit_cleaned_data

Writes reports/final_data_audit.md and exits non-zero if any check FAILs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train.csv"
TEST = ROOT / "data" / "raw" / "test.csv"
CLEAN = ROOT / "data" / "processed" / "cleaned_data.csv"
SCHEMA = ROOT / "data" / "processed" / "cleaned_data_schema.json"
CLEAN_REPORT = ROOT / "reports" / "data_cleaning_report.md"
OUT = ROOT / "reports" / "final_data_audit.md"
EXPECTED_SHA = "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1"
N = 45_593

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, evidence: str) -> None:
    assert status in {"PASS", "REVIEW", "FAIL"}
    RESULTS.append((name, status, evidence))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hav(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def same(a: pd.Series, b: pd.Series) -> bool:
    """Element-wise equality treating NaN == NaN, exact for floats read from text."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return bool(((a == b) | (a.isna() & b.isna())).all())


def main() -> int:
    sha_before = sha(RAW)

    # ---------- raw, independently parsed ----------
    r = pd.read_csv(RAW, dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())
    num = lambda s: pd.to_numeric(s.replace("NaN", np.nan))
    rid = r["ID"]
    rlat, rlon = num(r["Restaurant_latitude"]), num(r["Restaurant_longitude"])
    dlat, dlon = num(r["Delivery_location_latitude"]), num(r["Delivery_location_longitude"])
    rage, rrat = num(r["Delivery_person_Age"]), num(r["Delivery_person_Ratings"])
    rtarget = r["Time_taken(min)"].str.replace("(min)", "", regex=False).str.strip().astype(int)

    # ---------- cleaned, read straight from disk (no project loader) ----------
    blank = pd.read_csv(CLEAN, dtype=str, keep_default_na=False)       # for literal empty-cell counts
    c = pd.read_csv(CLEAN)                                               # default NA parsing
    for f in [x for x in c.columns if x.startswith("dq_")]:
        c[f] = c[f].map({True: True, False: False, "True": True, "False": False})

    # 1 ---- records & order -------------------------------------------------------------
    ok = len(c) == len(r) == N and c["order_id"].is_unique and (c["order_id"].values == rid.values).all()
    record("1. All 45,593 records present, order_id matches raw ID in the same order",
           "PASS" if ok else "FAIL",
           f"raw {len(r):,} rows, cleaned {len(c):,} rows; order_id unique={c['order_id'].is_unique}; "
           f"position-by-position match={(c['order_id'].values == rid.values).all()}")

    # 2 ---- 431 sign corrections ---------------------------------------------------------
    zero = (rlat == 0) & (rlon == 0)
    neg = ((rlat < 0) | (rlon < 0)) & ~zero
    pos = ~neg & ~zero
    alat, alon = rlat.abs(), rlon.abs()
    known = set(zip(r.loc[pos, "Delivery_person_ID"], rlat[pos].round(6), rlon[pos].round(6)))
    tA = pd.Series([(i, a, b) in known for i, a, b in zip(r["Delivery_person_ID"], alat.round(6), alon.round(6))])
    band = lambda s: s.round(6).between(0.0099, 0.1401)
    tB = band(dlat - alat) & band(dlon - alon)
    tC = alat.between(6, 37.5) & alon.between(68, 97.5) & dlat.between(6, 37.5) & dlon.between(68, 97.5)
    tD = pd.Series(hav(alat, alon, dlat, dlon) <= 25)
    expected_fix = neg & tA & tB & tC & tD
    flag = c["dq_restaurant_sign_corrected"]
    checks2 = {
        "raw negative rows": int(neg.sum()),
        "pass all 4 rules (recomputed)": int(expected_fix.sum()),
        "flag == recomputed set": bool((flag.values == expected_fix.values).all()),
        "cleaned lat/lon == |raw|": same(c.loc[neg, "restaurant_lat"], alat[neg]) and same(c.loc[neg, "restaurant_lon"], alon[neg]),
        "*_raw == signed raw": same(c.loc[neg, "restaurant_lat_raw"], rlat[neg]) and same(c.loc[neg, "restaurant_lon_raw"], rlon[neg]),
        "positive-sign rows untouched": same(c.loc[pos, "restaurant_lat"], rlat[pos]) and same(c.loc[pos, "restaurant_lon"], rlon[pos]),
    }
    ok = neg.sum() == 431 and expected_fix.sum() == 431 and all(v for k, v in checks2.items() if isinstance(v, bool))
    record("2. 431 coordinate corrections match raw values and the approved validation rules",
           "PASS" if ok else "FAIL", "; ".join(f"{k}: {v}" for k, v in checks2.items()))

    # 3 ---- 3,640 invalid geography ------------------------------------------------------
    geo_cols = ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon"]
    inv = ~c["dq_geo_valid"]
    checks3 = {
        "raw (0,0) rows": int(zero.sum()),
        "dq_geo_valid False": int(inv.sum()),
        "flag set == raw (0,0) set": bool((inv.values == zero.values).all()),
        "all 4 cleaned coords blank": bool(c.loc[inv, geo_cols].isna().all().all()),
        "originals preserved in *_raw": same(c.loc[inv, "restaurant_lat_raw"], rlat[zero]) and same(c.loc[inv, "delivery_lat_raw"], dlat[zero])
                                         and same(c.loc[inv, "restaurant_lon_raw"], rlon[zero]) and same(c.loc[inv, "delivery_lon_raw"], dlon[zero]),
        "no blank coords on valid rows": bool(c.loc[~inv, geo_cols].notna().all().all()),
        "not sign-corrected": bool((~c.loc[inv, "dq_restaurant_sign_corrected"]).all()),
    }
    ok = zero.sum() == 3640 and all(v for v in checks3.values() if isinstance(v, bool))
    record("3. 3,640 invalid geographic records have blank cleaned coordinates and flags",
           "PASS" if ok else "FAIL", "; ".join(f"{k}: {v}" for k, v in checks3.items()))

    # 4 ---- 91 rider sentinel records ----------------------------------------------------
    sent = ((rage == 15) & (rrat == 1.0)) | ((rage == 50) & (rrat == 6.0))
    s = c["dq_rider_sentinel"]
    other = ~sent
    checks4 = {
        "raw sentinel rows (age15+r1 / age50+r6)": f"{int(sent.sum())} ({int(((rage == 15) & sent).sum())} / {int(((rage == 50) & sent).sum())})",
        "flag == raw set": bool((s.values == sent.values).all()),
        "cleaned age & rating blank": bool(c.loc[s, ["delivery_person_age", "delivery_person_rating"]].isna().all().all()),
        "originals preserved": same(c.loc[s, "delivery_person_age_raw"], rage[sent]) and same(c.loc[s, "delivery_person_rating_raw"], rrat[sent]),
        "all other rows' age/rating == raw": same(c.loc[other, "delivery_person_age"], rage[other]) and same(c.loc[other, "delivery_person_rating"], rrat[other]),
    }
    ok = sent.sum() == 91 and all(v for v in checks4.values() if isinstance(v, bool))
    record("4. 91 suspicious rider records: age & rating blank, originals preserved",
           "PASS" if ok else "FAIL", "; ".join(f"{k}: {v}" for k, v in checks4.items()))

    # 5 ---- missing values straight from the saved CSV -------------------------------------
    raw_map = {"Delivery_person_Age": "delivery_person_age", "Delivery_person_Ratings": "delivery_person_rating",
               "Restaurant_latitude": "restaurant_lat", "Restaurant_longitude": "restaurant_lon",
               "Delivery_location_latitude": "delivery_lat", "Delivery_location_longitude": "delivery_lon",
               "Time_Orderd": "order_time", "Weatherconditions": "weather", "Road_traffic_density": "traffic_density",
               "multiple_deliveries": "multiple_deliveries", "Festival": "festival", "City": "city_type",
               "Time_taken(min)": "time_taken_min", "ID": "order_id", "Delivery_person_ID": "delivery_person_id",
               "Order_Date": "order_date", "Time_Order_picked": "pickup_time", "Vehicle_condition": "vehicle_condition",
               "Type_of_order": "order_type", "Type_of_vehicle": "vehicle_type"}
    raw_missing = {raw_map[k]: int(r[k].isin(["NaN", "conditions NaN"]).sum()) for k in raw_map}
    extra_expected = {"delivery_person_age": int(sent.sum()), "delivery_person_rating": int(sent.sum()),
                      **{g: int(zero.sum()) for g in geo_cols}}
    rows5, ok5 = [], True
    for col in c.columns:
        empty = int((blank[col] == "").sum())
        na = int(c[col].isna().sum())
        exp = raw_missing.get(col, None)
        exp = None if exp is None else exp + extra_expected.get(col, 0)
        good = (empty == na) and (exp is None or empty == exp)
        ok5 &= good
        rows5.append((col, empty, na, "—" if exp is None else f"{exp:,}", "ok" if good else "MISMATCH"))
    leftover = int(blank.isin(["NaN", "conditions NaN", "nan", "None"]).sum().sum())
    ok5 &= leftover == 0
    record("5. Missing-value counts recomputed from the saved CSV",
           "PASS" if ok5 else "FAIL",
           f"every column: empty cells == pandas NA, and == raw missing tokens + documented blanks; "
           f"literal 'NaN'/'nan'/'None' strings left: {leftover} (table in §Missing values)")

    # 6 ---- 1,731 missing order timestamps -------------------------------------------------
    m_raw = r["Time_Orderd"] == "NaN"
    report_txt = CLEAN_REPORT.read_text(encoding="utf-8")
    checks6 = {
        "raw 'NaN' Time_Orderd": int(m_raw.sum()),
        "cleaned order_time blank on same rows": bool((c["order_time"].isna().values == m_raw.values).all()),
        "order_datetime blank on same rows": bool((c["order_datetime"].isna().values == m_raw.values).all()),
        "dq_pickup_day_assumed on same rows": bool((c["dq_pickup_day_assumed"].values == m_raw.values).all()),
        "pickup_time still present": bool(c.loc[m_raw.values, "pickup_time"].notna().all()),
        "documented in cleaning report": "1,731" in report_txt,
    }
    ok = m_raw.sum() == 1731 and all(v for v in checks6.values() if isinstance(v, bool))
    record("6. 1,731 missing order timestamps identified & documented",
           "PASS" if ok else "FAIL", "; ".join(f"{k}: {v}" for k, v in checks6.items()))

    # 7 ---- dq_* / *_raw excluded from features --------------------------------------------
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    audit_cols = [x for x in c.columns if x.startswith("dq_") or x.endswith("_raw")]
    tagged = all("never a feature" in schema[x]["role"] for x in audit_cols)
    target_tagged = "never a feature" in schema["time_taken_min"]["role"]
    pickup_excl = all("EXCLUDED" in schema[x]["role"] for x in ["pickup_time", "pickup_datetime"])
    record("7. dq_* and *_raw columns excluded from future model features",
           "REVIEW" if (tagged and target_tagged and pickup_excl) else "FAIL",
           f"{len(audit_cols)} audit columns all tagged 'never a feature' in schema: {tagged}; target tagged: "
           f"{target_tagged}; pickup_* tagged EXCLUDED: {pickup_excl}. Documented in CLAUDE.md and the cleaning "
           "report, but no feature code exists yet, so exclusion is not yet *enforced*. Stage 3 must assert it in code.")

    # 8 ---- no target leakage introduced -----------------------------------------------------
    tgt_same = (c["time_taken_min"].values == rtarget.values).all()
    rollover = (pd.to_timedelta(r["Time_Order_picked"]) < pd.to_timedelta(r["Time_Orderd"].replace("NaN", None)))
    flags_from_inputs = (
        (c["dq_pickup_midnight_rollover"].values == rollover.fillna(False).values).all()
        and (c["dq_restaurant_sign_corrected"].values == expected_fix.values).all()
        and (inv.values == zero.values).all() and (s.values == sent.values).all()
        and (c["dq_pickup_day_assumed"].values == m_raw.values).all())
    # each non-target feature column reproducible from its raw (non-target) source
    cat_ok = (same(c["weather"], r["Weatherconditions"].str.replace("conditions ", "", regex=False).replace("NaN", np.nan))
              and same(c["traffic_density"], r["Road_traffic_density"].replace("NaN", np.nan))
              and same(c["festival"], r["Festival"].replace("NaN", np.nan))
              and same(c["city_type"], r["City"].replace({"NaN": np.nan, "Metropolitian": "Metropolitan"}))
              and same(c["order_type"], r["Type_of_order"]) and same(c["vehicle_type"], r["Type_of_vehicle"])
              and same(c["vehicle_condition"], num(r["Vehicle_condition"]))
              and same(c["multiple_deliveries"], num(r["multiple_deliveries"])))
    numeric = c.select_dtypes("number").drop(columns=["time_taken_min"])
    max_corr = numeric.corrwith(c["time_taken_min"], method="spearman").abs().max()
    target_named = [x for x in c.columns if ("time_taken" in x or "target" in x) and x != "time_taken_min"]
    ok = tgt_same and flags_from_inputs and cat_ok and not target_named and max_corr < 0.9
    record("8. No target leakage introduced",
           "PASS" if ok else "FAIL",
           f"target identical to raw: {tgt_same}; all 5 dq flags recomputed from non-target raw columns: "
           f"{flags_from_inputs}; categorical/int columns reproduce raw sources: {cat_ok}; extra target-like "
           f"columns: {target_named or 'none'}; max |Spearman| of any numeric column with target: {max_corr:.3f}. "
           "Note: the raw dataset's pre-existing Festival-missing artifact (target 10–14 min) was kept as blanks, "
           "not created by cleaning; no indicator column was added.")

    # 9 ---- raw SHA-256 ---------------------------------------------------------------------
    sha_after = sha(RAW)
    record("9. Raw file SHA-256 unchanged", "PASS" if sha_before == sha_after == EXPECTED_SHA else "FAIL",
           f"expected = before = after = {EXPECTED_SHA}" if sha_before == sha_after == EXPECTED_SHA
           else f"before {sha_before} after {sha_after} expected {EXPECTED_SHA}")

    # 10 ---- test.csv -----------------------------------------------------------------------
    others = sorted(p.name for p in (ROOT / "data" / "raw").iterdir())
    record("10. data/raw/test.csv exists?", "REVIEW" if not TEST.exists() else "PASS",
           f"test.csv {'FOUND' if TEST.exists() else 'NOT FOUND'}; data/raw contains: {others}. "
           + ("" if TEST.exists() else "No external test set: Stage 4 must hold out a chronological test split "
              "from train.csv (already approved), and no test-set results can be reported against an official test file."))

    # 11 ---- city spelling count -----------------------------------------------------------
    mis = int((r["City"] == "Metropolitian").sum())
    correct_raw = int((r["City"] == "Metropolitan").sum())
    metrics = json.loads((ROOT / "reports" / "cleaning_metrics.json").read_text(encoding="utf-8"))
    reported = metrics["values_changed"]["city_type"][0]
    record("11. City spelling-correction count = originally misspelled values only",
           "PASS" if reported == mis else "FAIL",
           f"raw 'Metropolitian': {mis:,}; raw already-correct 'Metropolitan': {correct_raw}; "
           f"reported count: {reported:,}; count now taken from raw values (step R5), not from cleaned rows")

    # ---------- write report ----------
    n_fail = sum(st == "FAIL" for _, st, _ in RESULTS)
    L = ["# DeliveryIQ — Final Data Audit (cleaned vs. raw)\n",
         "Independent check of `data/processed/cleaned_data.csv` against `data/raw/train.csv`. "
         "Produced by `src/audit_cleaned_data.py`, which does **not** import the cleaning pipeline.\n",
         f"**Cleaned file SHA-256:** `{sha(CLEAN)}` · **Raw file SHA-256:** `{sha_after}`\n",
         "| Check | Result | Evidence |", "|---|---|---|"]
    L += [f"| {n} | **{st}** | {ev.replace('|', '/')} |" for n, st, ev in RESULTS]
    L += ["\n## Missing values (computed from the saved CSV)\n",
          "| Column | Empty cells | pandas NA | Expected (raw tokens + documented blanks) | Status |", "|---|---|---|---|---|"]
    L += [f"| `{a}` | {b:,} | {c_:,} | {d} | {e} |" for a, b, c_, d, e in rows5 if b or d not in ("—", "0")]
    L += ["\nColumns not listed have no missing values.\n",
          f"**Summary:** {sum(st == 'PASS' for _, st, _ in RESULTS)} PASS · "
          f"{sum(st == 'REVIEW' for _, st, _ in RESULTS)} REVIEW · {n_fail} FAIL\n"]
    OUT.write_text("\n".join(L), encoding="utf-8")

    for n, st, ev in RESULTS:
        print(f"[{st:6s}] {n}")
    print("missing (col, empty, na, expected):", [(a, b, d) for a, b, _, d, e in rows5 if b])
    print("wrote", OUT)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
