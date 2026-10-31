"""DeliveryIQ — Stage 2 data cleaning and validation.

Reads the immutable raw file ``data/raw/train.csv``, applies the owner-approved
cleaning rules (see ``reports/data_cleaning_report.md``), validates the result and writes:

* ``data/processed/cleaned_data.csv``         — all 45,593 rows, snake_case columns
* ``data/processed/cleaned_data_schema.json`` — dtype + role of every column
* ``reports/cleaning_metrics.json``           — every count quoted in the report
* ``reports/data_cleaning_report.md``         — human-readable cleaning report

Run from the project root::

    .venv\\Scripts\\python.exe -m src.data_cleaning

Design rules (CLAUDE.md):
* The raw file is never written to; its SHA-256 is verified before and after the run.
* No rows are deleted. Problems are recorded in ``dq_*`` flag columns, and every changed
  value keeps its original in a ``*_raw`` column.
* No imputation, no feature engineering, no model training happens here.
* ``dq_*`` and ``*_raw`` columns are audit columns and must never be used as model features.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
OUT_SCHEMA = PROJECT_ROOT / "data" / "processed" / "cleaned_data_schema.json"
OUT_METRICS = PROJECT_ROOT / "reports" / "cleaning_metrics.json"
OUT_REPORT = PROJECT_ROOT / "reports" / "data_cleaning_report.md"

RAW_SHA256 = "d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1"
EXPECTED_ROWS = 45_593

MISSING_TOKEN = "NaN"                      # after whitespace stripping
WEATHER_PREFIX = "conditions "             # "conditions Sunny" -> "Sunny"; "conditions NaN" -> null
TARGET_PREFIX = "(min)"                    # "(min) 24" -> 24

# Generous mainland-India bounding box (assumption, also used in the Stage 1 audit).
INDIA_LAT = (6.0, 37.5)
INDIA_LON = (68.0, 97.5)
DELIVERY_OFFSET_BAND = (0.0099, 0.1401)    # observed delivery-minus-restaurant offset band (degrees)
MAX_VALID_DISTANCE_KM = 25.0

# Owner-approved rider rules (2026-10-31).
SENTINEL_AGE_RATING = {15.0: 1.0, 50.0: 6.0}   # age -> the only rating that co-occurs with it
RATING_RANGE = (1.0, 5.0)                      # ASSUMED scale; not officially documented

COLUMN_MAP: dict[str, str] = {
    "ID": "order_id",
    "Delivery_person_ID": "delivery_person_id",
    "Delivery_person_Age": "delivery_person_age",
    "Delivery_person_Ratings": "delivery_person_rating",
    "Restaurant_latitude": "restaurant_lat",
    "Restaurant_longitude": "restaurant_lon",
    "Delivery_location_latitude": "delivery_lat",
    "Delivery_location_longitude": "delivery_lon",
    "Order_Date": "order_date",
    "Time_Orderd": "order_time",
    "Time_Order_picked": "pickup_time",
    "Weatherconditions": "weather",
    "Road_traffic_density": "traffic_density",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle_type",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city_type",
    "Time_taken(min)": "time_taken_min",
}

ALLOWED_CATEGORIES: dict[str, set[str]] = {
    "weather": {"Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"},
    "traffic_density": {"Low", "Medium", "High", "Jam"},
    "order_type": {"Snack", "Meal", "Drinks", "Buffet"},
    "vehicle_type": {"motorcycle", "scooter", "electric_scooter", "bicycle"},
    "festival": {"Yes", "No"},
    "city_type": {"Metropolitan", "Urban", "Semi-Urban"},
}
CATEGORY_RENAMES = {"city_type": {"Metropolitian": "Metropolitan"}}

# Role of every output column (availability tiers from CLAUDE.md §7).
COLUMN_ROLES: dict[str, str] = {
    "order_id": "row_identifier (never a feature)",
    "delivery_person_id": "identifier (raw value never a feature; city prefix is Tier A)",
    "delivery_person_age": "feature Tier B (rider assignment)",
    "delivery_person_rating": "feature Tier B (rider assignment; leakage risk)",
    "restaurant_lat": "feature Tier A (order placement)",
    "restaurant_lon": "feature Tier A (order placement)",
    "delivery_lat": "feature Tier A (order placement)",
    "delivery_lon": "feature Tier A (order placement)",
    "order_date": "feature Tier A (order placement)",
    "order_time": "feature Tier A (order placement)",
    "order_datetime": "feature Tier A (order placement); chronological split key",
    "pickup_time": "Tier C (pickup) — EXCLUDED from order-placement models",
    "pickup_datetime": "Tier C (pickup) — EXCLUDED from order-placement models",
    "weather": "feature Tier A* (placement only if from a live feed)",
    "traffic_density": "feature Tier A* (placement only if from a live feed)",
    "vehicle_condition": "feature Tier B (rider assignment)",
    "order_type": "feature Tier A (order placement)",
    "vehicle_type": "feature Tier B (rider assignment)",
    "multiple_deliveries": "feature Tier B (dispatch; leakage risk)",
    "festival": "feature Tier A (order placement)",
    "city_type": "feature Tier A (order placement)",
    "time_taken_min": "TARGET (never a feature)",
    "restaurant_lat_raw": "audit: original value (never a feature)",
    "restaurant_lon_raw": "audit: original value (never a feature)",
    "delivery_lat_raw": "audit: original value (never a feature)",
    "delivery_lon_raw": "audit: original value (never a feature)",
    "delivery_person_age_raw": "audit: original value (never a feature)",
    "delivery_person_rating_raw": "audit: original value (never a feature)",
    "dq_restaurant_sign_corrected": "data-quality flag (never a feature)",
    "dq_geo_valid": "data-quality flag (never a feature); filter for geo work",
    "dq_rider_sentinel": "data-quality flag (never a feature)",
    "dq_pickup_midnight_rollover": "data-quality flag (never a feature)",
    "dq_pickup_day_assumed": "data-quality flag (never a feature)",
}
OUTPUT_COLUMNS = list(COLUMN_ROLES)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (used for validation only, not saved as a feature)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def _in_india(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return lat.between(*INDIA_LAT) & lon.between(*INDIA_LON)


def _in_band(s: pd.Series) -> pd.Series:
    return s.round(6).between(*DELIVERY_OFFSET_BAND)


@dataclass
class CleaningLog:
    """Counts and evidence collected while cleaning (serialised to cleaning_metrics.json)."""

    steps: dict = field(default_factory=dict)

    def add(self, key: str, value) -> None:
        self.steps[key] = value


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def load_raw(path: Path = RAW_PATH, verify_hash: bool = True) -> pd.DataFrame:
    """Read the raw CSV read-only with every value as text (no implicit NA conversion)."""
    if verify_hash:
        actual = sha256_file(path)
        if actual != RAW_SHA256:
            raise ValueError(f"Raw file hash mismatch: {actual} != {RAW_SHA256}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != list(COLUMN_MAP):
        raise ValueError(f"Unexpected raw columns: {list(df.columns)}")
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply the approved cleaning rules. Returns a new DataFrame; the input is not mutated."""
    log = CleaningLog()
    df = raw.copy()

    # ---- Rule 1: whitespace -----------------------------------------------------------
    ws = {c: int((df[c] != df[c].str.strip()).sum()) for c in df.columns}
    log.add("r1_whitespace_stripped_values", {k: v for k, v in ws.items() if v})
    df = df.apply(lambda s: s.str.strip())

    # ---- Rule 2: missing-value tokens --------------------------------------------------
    weather_nan = int((df["Weatherconditions"] == WEATHER_PREFIX + MISSING_TOKEN).sum())
    df["Weatherconditions"] = df["Weatherconditions"].str.removeprefix(WEATHER_PREFIX)
    tokens = {c: int((df[c] == MISSING_TOKEN).sum()) for c in df.columns}
    df = df.replace({MISSING_TOKEN: np.nan})
    log.add("r2_missing_tokens_to_null", {k: v for k, v in tokens.items() if v})
    log.add("r2_weather_conditions_nan_tokens", weather_nan)

    # ---- Rule 3: rename columns -------------------------------------------------------
    df = df.rename(columns=COLUMN_MAP)

    # ---- Rule 4: types ----------------------------------------------------------------
    target = df["time_taken_min"].str.removeprefix(TARGET_PREFIX).str.strip()
    if not target.str.fullmatch(r"\d+").all():
        raise ValueError("Target contains values not matching '(min) <int>'")
    df["time_taken_min"] = target.astype("int64")

    for c in ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon",
              "delivery_person_age", "delivery_person_rating"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    for c in ["vehicle_condition", "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("Int64")

    # ---- Rule 5: categorical normalisation ---------------------------------------------
    for col, mapping in CATEGORY_RENAMES.items():
        log.add(f"r5_{col}_renamed_values", {k: int((df[col] == k).sum()) for k in mapping})
        df[col] = df[col].replace(mapping)

    # ---- Rule 6: dates & times, midnight roll-over ------------------------------------
    order_date = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="raise")
    for c in ["order_time", "pickup_time"]:
        bad = df[c].notna() & ~df[c].str.fullmatch(r"\d{2}:\d{2}:\d{2}", na=False)
        if bad.any():
            raise ValueError(f"{c}: {int(bad.sum())} values not in HH:MM:SS format")
    order_td = pd.to_timedelta(df["order_time"])          # NaT where missing
    pickup_td = pd.to_timedelta(df["pickup_time"])
    order_dt = order_date + order_td
    pickup_dt = order_date + pickup_td
    rollover = order_dt.notna() & (pickup_dt < order_dt)
    pickup_dt = pickup_dt.where(~rollover, pickup_dt + pd.Timedelta(days=1))
    df["order_date"] = order_date.dt.date.astype(str)
    df["order_datetime"] = order_dt
    df["pickup_datetime"] = pickup_dt
    df["dq_pickup_midnight_rollover"] = rollover
    df["dq_pickup_day_assumed"] = order_dt.isna()   # no order time -> same-day pickup assumed
    log.add("r6_pickup_midnight_rollover_rows", int(rollover.sum()))
    log.add("r6_pickup_day_assumed_rows", int(order_dt.isna().sum()))
    log.add("r6_pickup_day_assumed_with_pickup_hour_0",
            int((order_dt.isna() & (pickup_td < pd.Timedelta(hours=1))).sum()))

    # ---- Rule 7: preserve originals before changing values ----------------------------
    for c in ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon",
              "delivery_person_age", "delivery_person_rating"]:
        df[f"{c}_raw"] = df[c]

    # ---- Rule 8: restaurant sign correction (only when geographically validated) -------
    rlat, rlon = df["restaurant_lat_raw"], df["restaurant_lon_raw"]
    dlat, dlon = df["delivery_lat_raw"], df["delivery_lon_raw"]
    zero = (rlat == 0) & (rlon == 0)
    negative = ((rlat < 0) | (rlon < 0)) & ~zero
    pos = ~negative & ~zero
    alat, alon = rlat.abs(), rlon.abs()

    # Evidence test A: |location| already recorded with positive sign for the same delivery_person_id
    known = set(zip(df.loc[pos, "delivery_person_id"], rlat[pos].round(6), rlon[pos].round(6)))
    test_a = pd.Series([(i, la, lo) in known for i, la, lo in
                        zip(df["delivery_person_id"], alat.round(6), alon.round(6))], index=df.index)
    # Evidence test B: delivery offset from |location| lies in the dataset's normal band
    test_b = _in_band(dlat - alat) & _in_band(dlon - alon)
    # Evidence test C: |location| and delivery point inside India
    test_c = _in_india(alat, alon) & _in_india(dlat, dlon)
    # Evidence test D: distance after correction within the plausible delivery radius
    test_d = pd.Series(haversine_km(alat, alon, dlat, dlon) <= MAX_VALID_DISTANCE_KM, index=df.index)

    correct = negative & test_a & test_b & test_c & test_d
    df.loc[correct, "restaurant_lat"] = alat[correct]
    df.loc[correct, "restaurant_lon"] = alon[correct]
    df["dq_restaurant_sign_corrected"] = correct
    log.add("r8_negative_sign_rows", int(negative.sum()))
    log.add("r8_negative_lat_only", int(((rlat < 0) & (rlon >= 0)).sum()))
    log.add("r8_negative_lat_and_lon", int(((rlat < 0) & (rlon < 0)).sum()))
    log.add("r8_negative_lon_only", int(((rlat >= 0) & (rlon < 0)).sum()))
    log.add("r8_evidence_pass_counts_on_negative_rows", {
        "A_same_id_location_seen_positive": int((negative & test_a).sum()),
        "B_delivery_offset_in_normal_band": int((negative & test_b).sum()),
        "C_inside_india": int((negative & test_c).sum()),
        "D_distance_le_25km": int((negative & test_d).sum()),
        "all_four": int(correct.sum()),
    })
    log.add("r8_sign_corrected_rows", int(correct.sum()))
    log.add("r8_negative_rows_failing_validation", int((negative & ~correct).sum()))

    # ---- Rule 9: invalid geography -> flag and blank (never imputed) -------------------
    geo_valid = (pos | correct)
    geo_valid &= _in_india(df["restaurant_lat"], df["restaurant_lon"]) & _in_india(dlat, dlon)
    geo_valid &= pd.Series(haversine_km(df["restaurant_lat"], df["restaurant_lon"], dlat, dlon)
                           <= MAX_VALID_DISTANCE_KM, index=df.index).fillna(False)
    for c in ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon"]:
        df.loc[~geo_valid, c] = np.nan
    df["dq_geo_valid"] = geo_valid
    log.add("r9_restaurant_zero_coordinate_rows", int(zero.sum()))
    log.add("r9_geo_invalid_rows", int((~geo_valid).sum()))
    log.add("r9_geo_invalid_rows_that_are_zero_coordinate", int((~geo_valid & zero).sum()))
    log.add("r9_zero_rows_whose_id_has_a_valid_location_elsewhere",
            int(df.loc[zero, "delivery_person_id"].isin(set(df.loc[pos, "delivery_person_id"])).sum()))

    # ---- Rule 10: rider sentinel block (age 15 / rating 1.0, age 50 / rating 6.0) -----
    age, rating = df["delivery_person_age_raw"], df["delivery_person_rating_raw"]
    sentinel = pd.Series(False, index=df.index)
    for a, r in SENTINEL_AGE_RATING.items():
        sentinel |= (age == a) & (rating == r)
    df.loc[sentinel, ["delivery_person_age", "delivery_person_rating"]] = np.nan
    df["dq_rider_sentinel"] = sentinel
    log.add("r10_rider_sentinel_rows", int(sentinel.sum()))
    log.add("r10_rider_sentinel_by_age", {str(int(a)): int(((age == a) & sentinel).sum())
                                          for a in SENTINEL_AGE_RATING})
    log.add("r10_age_15_or_50_not_matching_sentinel_rating",
            int((age.isin(list(SENTINEL_AGE_RATING)) & ~sentinel).sum()))
    log.add("r10_rating_outside_assumed_range_after_cleaning",
            int((~df["delivery_person_rating"].between(*RATING_RANGE) & df["delivery_person_rating"].notna()).sum()))

    out = df[OUTPUT_COLUMNS].copy()
    log.add("missing_after_cleaning", {c: int(v) for c, v in out.isna().sum().items() if v})
    return out, log


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate(raw: pd.DataFrame, clean_df: pd.DataFrame) -> list[dict]:
    """Run all post-cleaning checks. Returns a list of {check, passed, detail} records."""
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    c = clean_df
    check("row count preserved", len(c) == len(raw) == EXPECTED_ROWS, f"{len(raw)} -> {len(c)}")
    check("order_id unique", c["order_id"].is_unique)
    check("order_id 1:1 and same order as raw", (c["order_id"].values == raw["ID"].str.strip().values).all())
    check("output columns match schema", list(c.columns) == OUTPUT_COLUMNS)

    text = c.select_dtypes(include=["object", "string"])
    check("no 'NaN' text tokens remain", not text.isin([MISSING_TOKEN, WEATHER_PREFIX + MISSING_TOKEN]).any().any())
    check("no leading/trailing whitespace remains",
          all((s.dropna() == s.dropna().str.strip()).all() for _, s in text.items()))
    for col, allowed in ALLOWED_CATEGORIES.items():
        extra = set(c[col].dropna().unique()) - allowed
        check(f"{col} values in allowed set", not extra, f"unexpected: {sorted(extra)}" if extra else "")

    y = c["time_taken_min"]
    check("target is complete integer in [10, 54]",
          y.notna().all() and pd.api.types.is_integer_dtype(y) and y.between(10, 54).all(),
          f"min={y.min()} max={y.max()}")
    check("target unchanged vs raw",
          (y.values == raw["Time_taken(min)"].str.removeprefix("(min) ").astype(int).values).all())

    g = c["dq_geo_valid"]
    check("geo-valid rows: restaurant & delivery inside India",
          (_in_india(c.loc[g, "restaurant_lat"], c.loc[g, "restaurant_lon"])
           & _in_india(c.loc[g, "delivery_lat"], c.loc[g, "delivery_lon"])).all())
    dist = haversine_km(c.loc[g, "restaurant_lat"], c.loc[g, "restaurant_lon"],
                        c.loc[g, "delivery_lat"], c.loc[g, "delivery_lon"])
    check(f"geo-valid rows: distance <= {MAX_VALID_DISTANCE_KM:g} km", (dist <= MAX_VALID_DISTANCE_KM).all(),
          f"min={dist.min():.2f} max={dist.max():.2f} km")
    check("geo-invalid rows: all coordinates blank (not imputed)",
          c.loc[~g, ["restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon"]].isna().all().all())
    check("no negative coordinates remain", not (c[["restaurant_lat", "restaurant_lon"]] < 0).any().any())
    sc = c["dq_restaurant_sign_corrected"]
    check("sign-corrected rows equal abs(raw) values",
          np.allclose(c.loc[sc, "restaurant_lat"], c.loc[sc, "restaurant_lat_raw"].abs())
          and np.allclose(c.loc[sc, "restaurant_lon"], c.loc[sc, "restaurant_lon_raw"].abs()))
    untouched = g & ~sc
    check("non-corrected valid coordinates unchanged",
          (c.loc[untouched, "restaurant_lat"] == c.loc[untouched, "restaurant_lat_raw"]).all()
          and (c.loc[untouched, "restaurant_lon"] == c.loc[untouched, "restaurant_lon_raw"]).all())

    age, rating = c["delivery_person_age"], c["delivery_person_rating"]
    check("age within [20, 39] or missing", age.dropna().between(20, 39).all(),
          f"min={age.min()} max={age.max()}")
    check("rating within assumed [1, 5] or missing", rating.dropna().between(*RATING_RANGE).all(),
          f"min={rating.min()} max={rating.max()}")
    s = c["dq_rider_sentinel"]
    check("sentinel rows: age & rating blank, originals preserved",
          c.loc[s, ["delivery_person_age", "delivery_person_rating"]].isna().all().all()
          and c.loc[s, ["delivery_person_age_raw", "delivery_person_rating_raw"]].notna().all().all())

    has_order = c["order_datetime"].notna()
    gap = (c.loc[has_order, "pickup_datetime"] - c.loc[has_order, "order_datetime"]).dt.total_seconds() / 60
    check("pickup - order gap in {5, 10, 15} minutes", gap.isin([5, 10, 15]).all(),
          f"values: {sorted(gap.unique().tolist())}")
    check("pickup never before order", (gap >= 0).all())
    check("pickup_datetime present for every row", c["pickup_datetime"].notna().all())
    return checks


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise deterministically (fixed line endings and datetime format)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S")
    return buf.getvalue().encode("utf-8")


def build_schema(df: pd.DataFrame) -> dict:
    """Column dtype + role, used to reload the CSV with correct types."""
    return {col: {"dtype": str(df[col].dtype), "role": COLUMN_ROLES[col],
                  "missing": int(df[col].isna().sum())} for col in df.columns}


def load_cleaned(path: Path = OUT_CSV) -> pd.DataFrame:
    """Reload cleaned_data.csv with the dtypes recorded in the schema file."""
    schema = json.loads(OUT_SCHEMA.read_text(encoding="utf-8"))
    df = pd.read_csv(path, keep_default_na=True)
    for col, meta in schema.items():
        dt = meta["dtype"]
        if dt.startswith("datetime"):
            df[col] = pd.to_datetime(df[col])
        elif dt in ("Int64", "int64", "float64", "bool"):
            df[col] = df[col].astype(dt)
        else:
            df[col] = df[col].astype("string")
    return df


def write_report(metrics: dict, path: Path = OUT_REPORT) -> None:
    """Render reports/data_cleaning_report.md entirely from the computed metrics."""
    m, s = metrics, metrics["steps"]
    ev = s["r8_evidence_pass_counts_on_negative_rows"]
    n = m["rows_out"]
    flags = m["flag_counts"]
    L: list[str] = []
    add = L.append

    add("# DeliveryIQ — Stage 2 Data Cleaning Report\n")
    add(f"**Generated by:** `src/data_cleaning.py` on {m['run_date']} · **Status:** awaiting owner review  ")
    add("**Every number below is written by the pipeline from `reports/cleaning_metrics.json`.**  ")
    add(f"**Raw file:** `data/raw/train.csv` — SHA-256 before `{m['raw_sha256_before'][:12]}…`, "
        f"after `{m['raw_sha256_after'][:12]}…` → **{'unchanged' if m['raw_unchanged'] else 'CHANGED'}**\n")
    add("Labels: **[V]** verified from data · **[A]** assumption · **[D]** owner decision (2026-10-31)\n")

    add("## 1. Summary\n")
    add("| Item | Count |\n|---|---|")
    add(f"| Raw rows | {m['rows_in']:,} |")
    add(f"| Rows retained in `cleaned_data.csv` | **{n:,} (100%)** |")
    add(f"| Rows deleted | **{m['rows_in'] - n}** |")
    add(f"| Rows with ≥ 1 data-quality flag | {m['rows_any_flag']:,} ({100 * m['rows_any_flag'] / n:.2f}%) |")
    add(f"| Substantive values changed (coordinates, age, rating, pickup date) | {m['values_changed_substantive']:,} |")
    add(f"| Cosmetic spelling corrections (raw `Metropolitian` values only) | {m['values_changed']['city_type'][0]:,} |")
    add(f"| Validation checks passed | {m['checks_passed']} / {m['checks_total']} |\n")
    add("No rows were deleted. Use-case exclusions are applied later as filters on flag columns:\n")
    add(f"- **Geographic work (clustering, distance):** use `dq_geo_valid == True` → "
        f"{n - s['r9_geo_invalid_rows']:,} rows usable, "
        f"{s['r9_geo_invalid_rows']:,} excluded (no valid restaurant location).")
    add("- **Missing-Festival rows (228):** kept; Stage 4 compares models with vs. without them [D].\n")

    add("## 2. Column mapping (old → new) [D]\n")
    add("| # | Raw column | Cleaned column | Notes |\n|---|---|---|---|")
    notes = {"City": "holds a city *type*, renamed `city_type`; `Metropolitian` → `Metropolitan`",
             "Time_taken(min)": "TARGET; `(min) 24` → `24` (int)",
             "Weatherconditions": "`conditions ` prefix removed",
             "Time_Order_picked": "Tier C — excluded from order-placement features [D]",
             "Delivery_person_Ratings": "singular `rating`",
             "Time_Orderd": "source typo `Orderd` fixed"}
    for i, (old, new) in enumerate(COLUMN_MAP.items(), 1):
        add(f"| {i} | `{old}` | `{new}` | {notes.get(old, '')} |")
    add("\n**Columns added:**\n")
    add("| Column | Purpose |\n|---|---|")
    for col in OUTPUT_COLUMNS:
        if col not in COLUMN_MAP.values():
            add(f"| `{col}` | {COLUMN_ROLES[col]} |")
    add("")

    add("## 3. Cleaning rules applied\n")
    add("### R1 — Whitespace stripped [V]")
    add("| Column | Values stripped |\n|---|---|")
    for k, v in s["r1_whitespace_stripped_values"].items():
        add(f"| `{k}` | {v:,} |")
    add("\n### R2 — Missing-value tokens converted to nulls [V]")
    add("`\"NaN \"` (→ `NaN` after stripping) and `\"conditions NaN\"` became real nulls. **No imputation** "
        "is done here; it will happen inside the model pipeline, fitted on the training split only.\n")
    add("| Raw column | Tokens → null |\n|---|---|")
    for k, v in s["r2_missing_tokens_to_null"].items():
        add(f"| `{k}` | {v:,} |")
    add(f"\n(`Weatherconditions` tokens were `conditions NaN`: {s['r2_weather_conditions_nan_tokens']:,}.)\n")
    add("### R3 — Snake_case renaming — see §2 [D]\n")
    add("### R4 — Type parsing [V]")
    add("Target parsed to `int64` (all rows matched `(min) <int>`); coordinates, age and rating to `float64`; "
        "`vehicle_condition` and `multiple_deliveries` to nullable `Int64`.\n")
    add("### R5 — Category spelling [D]")
    add(f"`Metropolitian` → `Metropolitan`: {s['r5_city_type_renamed_values']['Metropolitian']:,} values.\n")
    add("### R6 — Dates, times and midnight roll-over [V]")
    add(f"- `order_datetime` = `order_date` + `order_time` (blank when `order_time` is missing: "
        f"{s['r6_pickup_day_assumed_rows']:,} rows).")
    add(f"- `pickup_datetime`: +1 day when pickup clock time < order clock time → "
        f"**{s['r6_pickup_midnight_rollover_rows']:,} rows** (`dq_pickup_midnight_rollover`).")
    add(f"- Where `order_time` is missing, the pickup is assumed to be on `order_date` "
        f"(`dq_pickup_day_assumed`, {s['r6_pickup_day_assumed_rows']:,} rows; "
        f"{s['r6_pickup_day_assumed_with_pickup_hour_0']} of them have a 00:xx pickup and could belong to "
        "the next day) [A].")
    add("- Missing `order_time` is **not** back-filled from pickup time, because that would copy Tier C "
        "information into a Tier A field.\n")
    add("### R7 — Originals preserved [V]")
    add("Before any value change, `restaurant_lat/lon`, `delivery_lat/lon`, `delivery_person_age` and "
        "`delivery_person_rating` are copied to `*_raw` columns.\n")
    add("### R8 — Restaurant sign correction (validated) [D][V]")
    add(f"Negative restaurant coordinates: **{s['r8_negative_sign_rows']} rows** "
        f"(latitude only: {s['r8_negative_lat_only']}, latitude and longitude: {s['r8_negative_lat_and_lon']}, "
        f"longitude only: {s['r8_negative_lon_only']}). A row is corrected **only if all four** tests pass on "
        "the absolute-value location:\n")
    add("| Test | Rows passing |\n|---|---|")
    add(f"| A. Same `delivery_person_id` already has this location with positive signs | {ev['A_same_id_location_seen_positive']} / {s['r8_negative_sign_rows']} |")
    add(f"| B. Delivery offset from the location is in the normal 0.01–0.14° band | {ev['B_delivery_offset_in_normal_band']} / {s['r8_negative_sign_rows']} |")
    add(f"| C. Location and delivery point inside India | {ev['C_inside_india']} / {s['r8_negative_sign_rows']} |")
    add(f"| D. Restaurant-to-delivery distance ≤ 25 km | {ev['D_distance_le_25km']} / {s['r8_negative_sign_rows']} |")
    add(f"| **All four → corrected** | **{ev['all_four']}** |\n")
    add(f"Rows failing validation (left uncorrected and flagged geo-invalid): {s['r8_negative_rows_failing_validation']}.\n")
    add("### R9 — Invalid geography flagged, never imputed [D][V]")
    add(f"- Restaurant recorded as (0, 0): **{s['r9_restaurant_zero_coordinate_rows']:,} rows**. Their delivery "
        "coordinates (0.01–0.14°) are offsets, not real positions.")
    add(f"- None of these rows' `delivery_person_id`s has a valid location elsewhere "
        f"({s['r9_zero_rows_whose_id_has_a_valid_location_elsewhere']} matches), so nothing is recoverable.")
    add(f"- All four cleaned coordinates are set to blank on **{s['r9_geo_invalid_rows']:,} rows** "
        f"(`dq_geo_valid = False`); originals stay in `*_raw`. The rows are kept for non-geographic models.\n")
    add("### R10 — Rider sentinel block [D][V][A]")
    add(f"Rows with age 15 + rating 1.0 or age 50 + rating 6.0: **{s['r10_rider_sentinel_rows']} rows** "
        f"(age 15: {s['r10_rider_sentinel_by_age']['15']}, age 50: {s['r10_rider_sentinel_by_age']['50']}). "
        f"Age 15/50 rows without the paired rating: {s['r10_age_15_or_50_not_matching_sentinel_rating']}.")
    add("- Age and rating set to blank, originals kept in `*_raw`, flagged `dq_rider_sentinel`.")
    add("- **[A] Assumption:** ratings run from 1 to 5. This is a working assumption, not an officially "
        "documented rule. Rating 6.0 is impossible under it.")
    add(f"- Ratings outside [1, 5] remaining after cleaning: {s['r10_rating_outside_assumed_range_after_cleaning']}. "
        "Ratings 2.5–3.9 are kept as plausible.\n")

    add("## 4. Values changed\n")
    add("| Column | Values changed | Reason |\n|---|---|---|")
    for col, (cnt, why) in m["values_changed"].items():
        add(f"| `{col}` | {cnt:,} | {why} |")
    add("")

    add("## 5. Data-quality flags\n")
    add("| Flag | Rows flagged | Meaning |\n|---|---|---|")
    meaning = {"dq_restaurant_sign_corrected": "restaurant sign fixed (R8)",
               "dq_geo_valid": "count shown = rows where **False**: coordinates invalid and blanked (R9)",
               "dq_rider_sentinel": "age/rating sentinel blanked (R10)",
               "dq_pickup_midnight_rollover": "pickup moved to next day (R6)",
               "dq_pickup_day_assumed": "no order time; pickup assumed same day (R6)"}
    for k, v in flags.items():
        add(f"| `{k}` | {v:,} | {meaning[k]} |")
    add(f"\nRows with at least one flag: **{m['rows_any_flag']:,}**. "
        "`dq_*` and `*_raw` columns are audit columns and **must never be used as model features**.\n")

    add("## 6. Missing values after cleaning\n")
    add("Blank by design: the model pipeline will impute or filter them later.\n")
    add("| Column | Missing | % |\n|---|---|---|")
    for k, v in m["steps"]["missing_after_cleaning"].items():
        if not k.endswith("_raw"):
            add(f"| `{k}` | {v:,} | {100 * v / n:.2f} |")
    add("")

    add("## 7. Validation checks\n")
    add("| # | Check | Result | Detail |\n|---|---|---|---|")
    for i, ch in enumerate(m["checks"], 1):
        add(f"| {i} | {ch['check']} | {'✅ pass' if ch['passed'] else '❌ FAIL'} | {ch['detail']} |")
    add(f"| {len(m['checks']) + 1} | deterministic output (two independent runs, identical bytes) | "
        f"{'✅ pass' if m['deterministic'] else '❌ FAIL'} | sha256 `{m['output_sha256'][:12]}…` |")
    add(f"| {len(m['checks']) + 2} | written CSV reloads identically via `load_cleaned()` | "
        f"{'✅ pass' if m['reload_roundtrip'] else '❌ FAIL'} | |")
    add(f"| {len(m['checks']) + 3} | raw file unchanged (SHA-256 before = after = expected) | "
        f"{'✅ pass' if m['raw_unchanged'] else '❌ FAIL'} | |\n")

    add("## 8. Leakage & usage rules carried forward [D]\n")
    add("- **Target:** `time_taken_min`. Never an input, never used to derive inputs.")
    add("- **`pickup_time` / `pickup_datetime`** (Tier C) are **excluded from order-placement models**.")
    add("- **Never features:** `order_id`, raw `delivery_person_id`, all `*_raw` and `dq_*` columns. "
        "No missing-Festival indicator.")
    add("- **Late benchmark (project-defined, not an official SLA):** `time_taken_min > 30`; sensitivity at "
        "> 35 and > 40. Not materialised in this dataset.")
    add("- **Split:** chronological by `order_datetime`/`order_date` in Stage 4.")
    add("- The column-role table is in `data/processed/cleaned_data_schema.json`.\n")

    add("## 9. Assumptions & limitations\n")
    add("- [A] Ratings run from 1 to 5. This is an assumption, not documented in the dataset.")
    add("- [A] The India bounding box (lat 6–37.5, lon 68–97.5) and the 25 km radius are validation heuristics.")
    add(f"- [A] Pickup is assumed to be on `order_date` when `order_time` is missing ({s['r6_pickup_day_assumed_rows']:,} rows).")
    add("- [V] Delivery coordinates remain synthetic offsets (Stage 1 §6), so geographic insight is mostly city-level.")
    add("- [A] The age 15/50 blocks are treated as placeholders. Other values in those rows (vehicle, weather) are kept unchanged.\n")

    add("## 10. Outputs & reproduction\n")
    add("| File | Content |\n|---|---|")
    add(f"| `data/processed/cleaned_data.csv` | {n:,} rows × {m['cols_out']} columns, sha256 `{m['output_sha256'][:12]}…` |")
    add("| `data/processed/cleaned_data_schema.json` | dtype, role and missing count per column |")
    add("| `reports/cleaning_metrics.json` | all counts in this report |")
    add("| `reports/data_cleaning_report.md` | this report |\n")
    add("```powershell\ncd D:\\Project1\n.venv\\Scripts\\python.exe -m src.data_cleaning\n```")
    add("Load in later stages with `from src.data_cleaning import load_cleaned`.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def _values_changed(c: pd.DataFrame, log: CleaningLog) -> dict:
    """Count changed values per column. The city count comes from the raw misspelled values
    (recorded in step R5), not from the number of cleaned 'Metropolitan' rows."""
    def diff(a, b):
        return int(((a != b) & ~(a.isna() & b.isna())).sum())
    return {
        "restaurant_lat": (diff(c.restaurant_lat, c.restaurant_lat_raw), "sign fix (R8) or blanked (R9)"),
        "restaurant_lon": (diff(c.restaurant_lon, c.restaurant_lon_raw), "sign fix (R8) or blanked (R9)"),
        "delivery_lat": (diff(c.delivery_lat, c.delivery_lat_raw), "blanked, geo-invalid (R9)"),
        "delivery_lon": (diff(c.delivery_lon, c.delivery_lon_raw), "blanked, geo-invalid (R9)"),
        "delivery_person_age": (diff(c.delivery_person_age, c.delivery_person_age_raw), "sentinel blanked (R10)"),
        "delivery_person_rating": (diff(c.delivery_person_rating, c.delivery_person_rating_raw), "sentinel blanked (R10)"),
        "city_type": (int(log.steps["r5_city_type_renamed_values"]["Metropolitian"]),
                      "raw `Metropolitian` values respelled `Metropolitan` (R5)"),
        "pickup date": (int(c.dq_pickup_midnight_rollover.sum()), "midnight roll-over +1 day (R6)"),
    }


def run(raw_path: Path = RAW_PATH, out_csv: Path = OUT_CSV) -> dict:
    """Execute load → clean → validate → write; raise if any validation check fails."""
    sha_before = sha256_file(raw_path)
    raw = load_raw(raw_path)
    cleaned, log = clean(raw)
    checks = validate(raw, cleaned)

    payload = to_csv_bytes(cleaned)
    payload_2 = to_csv_bytes(clean(load_raw(raw_path))[0])     # independent second run
    deterministic = payload == payload_2

    failed = [c["check"] for c in checks if not c["passed"]]
    if failed or not deterministic:
        raise RuntimeError(f"Validation failed: {failed or ['non-deterministic output']}. Nothing written.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Skip the write when the existing file is byte-identical (avoids needless rewrites and
    # failures when the CSV is open read-only in another program, e.g. Excel on Windows).
    if not (out_csv.exists() and out_csv.read_bytes() == payload):
        out_csv.write_bytes(payload)
    schema = build_schema(cleaned)
    OUT_SCHEMA.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    reloaded = load_cleaned(out_csv)
    reload_ok = (to_csv_bytes(reloaded) == payload) and all(
        str(reloaded[col].dtype) == schema[col]["dtype"] or schema[col]["dtype"] in ("object", "str", "string")
        for col in reloaded.columns)
    sha_after = sha256_file(raw_path)

    flag_counts = {
        "dq_restaurant_sign_corrected": int(cleaned.dq_restaurant_sign_corrected.sum()),
        "dq_geo_valid": int((~cleaned.dq_geo_valid).sum()),
        "dq_rider_sentinel": int(cleaned.dq_rider_sentinel.sum()),
        "dq_pickup_midnight_rollover": int(cleaned.dq_pickup_midnight_rollover.sum()),
        "dq_pickup_day_assumed": int(cleaned.dq_pickup_day_assumed.sum()),
    }
    any_flag = (cleaned.dq_restaurant_sign_corrected | ~cleaned.dq_geo_valid | cleaned.dq_rider_sentinel
                | cleaned.dq_pickup_midnight_rollover | cleaned.dq_pickup_day_assumed)
    changed = _values_changed(cleaned, log)
    metrics = {
        "run_date": date.today().isoformat(),
        "raw_sha256_before": sha_before, "raw_sha256_after": sha_after,
        "raw_unchanged": sha_before == sha_after == RAW_SHA256,
        "rows_in": len(raw), "rows_out": len(cleaned), "cols_out": cleaned.shape[1],
        "rows_any_flag": int(any_flag.sum()),
        "flag_counts": flag_counts,
        "values_changed": changed,
        "values_changed_substantive": int(sum(v[0] for k, v in changed.items() if k != "city_type")),
        "steps": log.steps,
        "checks": checks,
        "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
        "deterministic": deterministic, "reload_roundtrip": bool(reload_ok),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if not (metrics["raw_unchanged"] and reload_ok):
        raise RuntimeError("Post-write check failed (raw hash or reload round-trip).")
    OUT_METRICS.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(metrics)
    return metrics


def main() -> None:
    """CLI entry point: ``python -m src.data_cleaning``."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    m = run()
    print(f"rows in/out: {m['rows_in']:,} / {m['rows_out']:,} | flagged rows: {m['rows_any_flag']:,}")
    print(f"validation: {m['checks_passed']}/{m['checks_total']} passed | deterministic: {m['deterministic']} "
          f"| reload round-trip: {m['reload_roundtrip']} | raw unchanged: {m['raw_unchanged']}")
    for c in m["checks"]:
        print(("  PASS " if c["passed"] else "  FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    print("wrote:", OUT_CSV.relative_to(PROJECT_ROOT), "|", OUT_SCHEMA.relative_to(PROJECT_ROOT), "|",
          OUT_METRICS.relative_to(PROJECT_ROOT), "|", OUT_REPORT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
