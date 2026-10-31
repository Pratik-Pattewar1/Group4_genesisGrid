"""DeliveryIQ — Stage 3 feature engineering.

Builds order-placement (Tier A) features for the primary ETA / late-risk models from
``data/processed/cleaned_data.csv``, validates them and writes:

* ``data/processed/feature_engineered_data.csv`` — all 45,593 rows, missing values kept
* ``reports/feature_dictionary.md``               — every column, its meaning and availability
* ``reports/feature_engineering_report.md``       — decisions, checks, limitations
* ``reports/feature_engineering_metrics.json``    — numbers quoted in the reports

Run from the project root::

    .venv\\Scripts\\python.exe -m src.feature_engineering

Guarantees:
* Raw and cleaned files are read-only (SHA-256 checked before and after).
* No imputation in the saved dataset: missing stays missing (no zeros, no guesses).
* Model inputs only come through :func:`get_model_inputs`, which calls :func:`assert_no_leakage`.
* Preprocessing (:func:`build_preprocessor`) is an sklearn pipeline that must be fitted on the
  training split only; :func:`chronological_split` produces that split on whole dates.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_cleaning import OUT_CSV as CLEAN_CSV
from src.data_cleaning import RAW_PATH, RAW_SHA256, load_cleaned

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "feature_engineered_data.csv"
OUT_DICT = PROJECT_ROOT / "reports" / "feature_dictionary.md"
OUT_REPORT = PROJECT_ROOT / "reports" / "feature_engineering_report.md"
OUT_METRICS = PROJECT_ROOT / "reports" / "feature_engineering_metrics.json"
EXPECTED_ROWS = 45_593
SEED = 42
TARGET = "time_taken_min"

# --------------------------------------------------------------------------------------
# Feature sets (owner decisions, 2026-10-31)
# --------------------------------------------------------------------------------------
PRIMARY_NUMERIC = ["distance_km", "restaurant_lat", "restaurant_lon",
                   "order_hour", "order_hour_sin", "order_hour_cos",
                   "order_day_of_month", "order_is_weekend"]
PRIMARY_CATEGORICAL = ["city_type", "order_type", "festival",
                       "order_time_band", "order_day_of_week", "order_month"]
PRIMARY_FEATURES = PRIMARY_NUMERIC + PRIMARY_CATEGORICAL

# Kept in the dataset for later, clearly-labelled comparison models — NOT primary inputs.
EXCLUDED_BY_DECISION = ["delivery_person_age", "delivery_person_rating", "vehicle_condition", "vehicle_type",
                        "multiple_deliveries", "weather", "traffic_density",
                        # Stage 4 review (2026-10-31): no independent order-time source for the city.
                        # Valid rows: implied by restaurant coordinates (already features).
                        # Geo-invalid rows: only from the rider-ID prefix (rider assignment).
                        "city_code"]
# Never model inputs under any feature set.
FORBIDDEN_EXACT = {TARGET, "order_id", "delivery_person_id", "pickup_time", "pickup_datetime",
                   "order_date", "order_datetime", "order_time"}
FORBIDDEN_PATTERNS = [r"_raw$", r"^dq_", r"pickup", r"time_taken", r"(^|_)late($|_)", r"target"]

# Order-hour bands (descriptive time-of-day, from order_time only).
TIME_BANDS = [(0, 8, "night_00_07"), (8, 11, "morning_08_10"), (11, 15, "midday_11_14"),
              (15, 19, "afternoon_15_18"), (19, 24, "evening_19_23")]

# Columns carried from the cleaned data into the engineered dataset.
CARRIED = ["order_id", "order_date", "order_datetime", TARGET,
           "restaurant_lat", "restaurant_lon", "delivery_lat", "delivery_lon",
           "city_type", "order_type", "festival",
           *[c for c in EXCLUDED_BY_DECISION if c != "city_code"],  # city_code is derived (NEW)
           "dq_geo_valid", "dq_restaurant_sign_corrected", "dq_rider_sentinel", "dq_pickup_day_assumed"]
NEW = ["distance_km", "city_code", "order_hour", "order_hour_sin", "order_hour_cos", "order_time_band",
       "order_day_of_week", "order_day_of_month", "order_month", "order_is_weekend"]
OUTPUT_COLUMNS = CARRIED[:4] + NEW + CARRIED[4:]

DICTIONARY: dict[str, tuple[str, str, str]] = {
    # column: (description, availability, primary-model status)
    "order_id": ("Unique order key", "—", "Forbidden (identifier)"),
    "order_date": ("Order calendar date (complete for all rows)", "Order placement", "Forbidden as input; chronological split key"),
    "order_datetime": ("Order date + time; missing for 1,731 rows", "Order placement", "Forbidden as input; reference only"),
    TARGET: ("Delivery time in minutes (10–54)", "After delivery", "TARGET — forbidden as input"),
    "distance_km": ("Haversine restaurant→delivery distance; missing where `dq_geo_valid` is False", "Order placement", "Primary (numeric)"),
    "city_code": ("City code from the `delivery_person_id` prefix (22 cities)", "Rider assignment (ID prefix); no independent order-time source", "Excluded by decision (Stage 4 review): redundant with restaurant coordinates on valid rows, rider-ID-only on geo-invalid rows"),
    "order_hour": ("Hour of order (0–23); missing where order time is missing", "Order placement", "Primary (numeric)"),
    "order_hour_sin": ("sin(2π·hour/24), cyclical hour; missing with hour", "Order placement", "Primary (numeric)"),
    "order_hour_cos": ("cos(2π·hour/24), cyclical hour; missing with hour", "Order placement", "Primary (numeric)"),
    "order_time_band": ("Order-hour band: night 00–07, morning 08–10, midday 11–14, afternoon 15–18, evening 19–23; missing with hour", "Order placement", "Primary (categorical)"),
    "order_day_of_week": ("Day of week 0=Mon … 6=Sun (from order_date, complete)", "Order placement", "Primary (categorical)"),
    "order_day_of_month": ("Day of month 1–31 (from order_date, complete)", "Order placement", "Primary (numeric)"),
    "order_month": ("Month 2–4 (from order_date, complete)", "Order placement", "Primary (categorical) — see limitation"),
    "order_is_weekend": ("1 if Saturday/Sunday else 0 (from order_date, complete)", "Order placement", "Primary (numeric)"),
    "restaurant_lat": ("Restaurant latitude (cleaned); missing where geo-invalid", "Order placement", "Primary (numeric)"),
    "restaurant_lon": ("Restaurant longitude (cleaned); missing where geo-invalid", "Order placement", "Primary (numeric)"),
    "delivery_lat": ("Delivery latitude (cleaned); missing where geo-invalid", "Order placement", "Not selected: synthetic offset, information carried by distance_km; kept for clustering"),
    "delivery_lon": ("Delivery longitude (cleaned); missing where geo-invalid", "Order placement", "Not selected: synthetic offset, information carried by distance_km; kept for clustering"),
    "city_type": ("Metropolitan / Urban / Semi-Urban; missing kept", "Order placement", "Primary (categorical)"),
    "order_type": ("Snack / Meal / Drinks / Buffet", "Order placement", "Primary (categorical)"),
    "festival": ("Festival day Yes/No; missing kept (no missing-indicator by rule)", "Order placement", "Primary (categorical)"),
    "delivery_person_age": ("Rider age; missing kept", "Rider assignment (Tier B)", "Excluded by decision"),
    "delivery_person_rating": ("Rider rating; missing kept", "Rider assignment (Tier B)", "Excluded by decision (leakage risk)"),
    "vehicle_condition": ("Vehicle condition 0–3", "Rider assignment (Tier B)", "Excluded by decision"),
    "vehicle_type": ("Vehicle type", "Rider assignment (Tier B)", "Excluded by decision"),
    "multiple_deliveries": ("Orders batched with this one", "Dispatch (Tier B)", "Excluded by decision (leakage risk)"),
    "weather": ("Weather condition", "Unknown recording time (Tier A*)", "Excluded by decision"),
    "traffic_density": ("Traffic level (≈ function of order hour)", "Unknown recording time (Tier A*)", "Excluded by decision"),
    "dq_geo_valid": ("False where restaurant was recorded as (0,0)", "—", "Forbidden (audit flag); filter for geo work"),
    "dq_restaurant_sign_corrected": ("Restaurant sign fixed in Stage 2", "—", "Forbidden (audit flag)"),
    "dq_rider_sentinel": ("Age/rating placeholder blanked in Stage 2", "—", "Forbidden (audit flag)"),
    "dq_pickup_day_assumed": ("Order time missing (the 1,731-row block)", "—", "Forbidden (audit flag); sensitivity filter"),
}


# --------------------------------------------------------------------------------------
# Leakage guard & model inputs
# --------------------------------------------------------------------------------------
class LeakageError(ValueError):
    """Raised when a forbidden or excluded column is requested as a model input."""


def assert_no_leakage(features: list[str], allow_excluded: bool = False) -> None:
    """Raise LeakageError if any feature is the target, an identifier, a pickup/post-outcome
    column, an audit column (``*_raw`` / ``dq_*``) or — unless ``allow_excluded`` — a column
    excluded from the primary model by owner decision."""
    bad = []
    for f in features:
        if f in FORBIDDEN_EXACT:
            bad.append(f"{f} (forbidden)")
        elif any(re.search(p, f) for p in FORBIDDEN_PATTERNS):
            bad.append(f"{f} (forbidden pattern)")
        elif f in EXCLUDED_BY_DECISION and not allow_excluded:
            bad.append(f"{f} (excluded from primary model)")
    if len(set(features)) != len(features):
        bad.append("duplicate feature names")
    if bad:
        raise LeakageError("Disallowed model inputs: " + ", ".join(bad))


def get_model_inputs(df: pd.DataFrame, features: list[str] | None = None,
                     allow_excluded: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). The only sanctioned way to build model inputs.

    Defaults to the primary (order-placement) feature set. ``allow_excluded=True`` admits the
    owner-excluded dispatch-time columns for clearly labelled *comparison* models only; forbidden
    columns (target, identifiers, pickup, ``*_raw``, ``dq_*``) are rejected regardless."""
    features = list(PRIMARY_FEATURES if features is None else features)
    assert_no_leakage(features, allow_excluded=allow_excluded)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Features not in dataframe: {missing}")
    X = df[features].copy()
    # sklearn needs np.nan rather than pd.NA: numeric -> float64, categorical -> object strings.
    # This changes the missing-value *marker* only; no value is filled.
    for c in features:
        if c in PRIMARY_CATEGORICAL or not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object").where(X[c].notna(), np.nan).map(lambda v: v if v is np.nan else str(v))
        else:
            X[c] = X[c].astype("float64")
    return X, df[TARGET].copy()


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.Index, pd.Index, str]:
    """Split on whole order dates: the latest dates holding ~test_frac of rows form the test set.
    Uses ``order_date`` (complete for every row), never ``order_datetime``. Returns (train_idx, test_idx, cut)."""
    d = pd.to_datetime(df["order_date"])
    per_date = d.value_counts().sort_index(ascending=False).cumsum()
    test_dates = per_date[per_date <= test_frac * len(df)].index
    cut = test_dates.min()
    test = d >= cut
    return df.index[~test], df.index[test], str(cut.date())


def build_preprocessor(numeric: list[str] | None = None, categorical: list[str] | None = None,
                       allow_excluded: bool = False) -> ColumnTransformer:
    """Unfitted preprocessing pipeline. Fit on the TRAINING split only.

    Numeric: median imputation (statistics learned from train) + standard scaling.
    Categorical: most-frequent imputation (from train) + one-hot (unknown levels ignored).
    No missing-indicator columns are added (avoids artifact indicators such as Festival-missing).
    """
    numeric = PRIMARY_NUMERIC if numeric is None else numeric
    categorical = PRIMARY_CATEGORICAL if categorical is None else categorical
    assert_no_leakage(list(numeric) + list(categorical), allow_excluded=allow_excluded)
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], verbose_feature_names_out=False)


# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; NaN in → NaN out."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Derive Tier A features from the cleaned data. Uses no target information and fills nothing."""
    df = clean[[c for c in CARRIED]].copy()

    geo = clean["dq_geo_valid"].astype(bool)
    dist = haversine_km(clean["restaurant_lat"], clean["restaurant_lon"], clean["delivery_lat"], clean["delivery_lon"])
    df["distance_km"] = pd.Series(dist, index=clean.index).where(geo).round(4)

    code = clean["delivery_person_id"].astype("string").str.extract(r"^([A-Z]+)RES\d+DEL\d+$")[0]
    if code.isna().any():
        raise ValueError("delivery_person_id prefix could not be parsed for some rows")
    df["city_code"] = code

    odt = pd.to_datetime(clean["order_datetime"])          # NaT on the 1,731 rows
    hour = odt.dt.hour.astype("Int64")
    df["order_hour"] = hour
    rad = 2 * np.pi * hour.astype("float64") / 24
    df["order_hour_sin"] = np.sin(rad).round(6)
    df["order_hour_cos"] = np.cos(rad).round(6)
    band = pd.Series(pd.NA, index=clean.index, dtype="string")
    for lo, hi, name in TIME_BANDS:
        band[(hour >= lo) & (hour < hi)] = name
    df["order_time_band"] = band

    d = pd.to_datetime(clean["order_date"])
    df["order_day_of_week"] = d.dt.day_of_week.astype("int64")
    df["order_day_of_month"] = d.dt.day.astype("int64")
    df["order_month"] = d.dt.month.astype("int64")
    df["order_is_weekend"] = (d.dt.day_of_week >= 5).astype("int64")
    return df[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate(clean: pd.DataFrame, fe: pd.DataFrame) -> list[dict]:
    """Automated checks on the engineered dataset, the leakage guard and train-only fitting."""
    out: list[dict] = []

    def check(name, ok, detail=""):
        out.append({"check": name, "passed": bool(ok), "detail": detail})

    check("row count preserved", len(fe) == len(clean) == EXPECTED_ROWS, f"{len(fe):,}")
    check("order_id identical and in the same order", (fe["order_id"].values == clean["order_id"].values).all())
    check("target carried unchanged", (fe[TARGET].values == clean[TARGET].values).all())
    check("every output column documented in feature dictionary", set(fe.columns) == set(DICTIONARY),
          str(sorted(set(fe.columns) ^ set(DICTIONARY))))

    # leakage guard: accepts primary set, rejects every forbidden/excluded column
    try:
        assert_no_leakage(PRIMARY_FEATURES); ok = True
    except LeakageError:
        ok = False
    check("primary feature set passes leakage guard", ok, f"{len(PRIMARY_FEATURES)} features")
    candidates = (sorted(FORBIDDEN_EXACT) + EXCLUDED_BY_DECISION
                  + [c for c in clean.columns if c.endswith("_raw") or c.startswith("dq_")]
                  + ["late_30", "is_late", "target_encoded_city"])
    escaped = []
    for c in candidates:
        try:
            assert_no_leakage(PRIMARY_FEATURES + [c]); escaped.append(c)
        except LeakageError:
            pass
    check("guard rejects every forbidden / excluded column", not escaped,
          f"{len(candidates)} tested; escaped: {escaped or 'none'}")
    X, y = get_model_inputs(fe)
    check("get_model_inputs returns only primary features", list(X.columns) == PRIMARY_FEATURES and y.name == TARGET)
    check("no primary feature overlaps excluded/forbidden lists",
          not set(PRIMARY_FEATURES) & (set(EXCLUDED_BY_DECISION) | FORBIDDEN_EXACT))

    # target independence: rebuild with shuffled target -> identical features
    shuffled = clean.copy()
    shuffled[TARGET] = np.random.default_rng(SEED).permutation(shuffled[TARGET].values)
    fe2 = build_features(shuffled)
    check("features unchanged when target is shuffled (no target dependence)",
          fe2[PRIMARY_FEATURES].equals(fe[PRIMARY_FEATURES]))
    num_corr = fe[PRIMARY_NUMERIC].astype("float64").corrwith(fe[TARGET].astype("float64"), method="spearman").abs()
    check("no numeric feature is a near-copy of the target (|Spearman| < 0.9)", (num_corr < 0.9).all(),
          f"max {num_corr.max():.3f} ({num_corr.idxmax()})")

    # distance
    geo = fe["dq_geo_valid"].astype(bool)
    check("distance_km missing exactly where geo-invalid", (fe["distance_km"].isna() == ~geo).all(),
          f"missing {int(fe['distance_km'].isna().sum()):,}")
    dv = fe.loc[geo, "distance_km"]
    check("distance_km within (0, 25] km on valid rows", dv.gt(0).all() and dv.le(25).all(),
          f"min {dv.min():.2f}, max {dv.max():.2f}")

    # time features
    tmiss = clean["order_datetime"].isna()
    hour_cols = ["order_hour", "order_hour_sin", "order_hour_cos", "order_time_band"]
    check("hour features missing exactly on the 1,731 missing-timestamp rows",
          all((fe[c].isna() == tmiss).all() for c in hour_cols) and int(tmiss.sum()) == 1731,
          f"{int(tmiss.sum()):,} rows")
    h = fe["order_hour"].dropna()
    raw_h = clean["order_time"].dropna().str.slice(0, 2).astype(int)
    check("order_hour matches order_time (0 is a real hour, not a fill)", (h.values == raw_h.values).all(),
          f"hour-0 rows: {int((h == 0).sum())}")
    date_cols = ["order_day_of_week", "order_day_of_month", "order_month", "order_is_weekend"]
    check("date features complete for all rows", fe[date_cols].notna().all().all())
    check("weekend flag consistent with day of week",
          ((fe["order_day_of_week"] >= 5).astype(int) == fe["order_is_weekend"]).all())

    # no imputation of carried columns
    carried_missing = {c: (int(fe[c].isna().sum()), int(clean[c].isna().sum())) for c in CARRIED}
    check("missing counts of all carried columns unchanged (nothing filled)",
          all(a == b for a, b in carried_missing.values()))
    check("age, rating and coordinates still missing as in cleaned data",
          all(carried_missing[c][0] == carried_missing[c][1] > 0 for c in
              ["delivery_person_age", "delivery_person_rating", "restaurant_lat", "delivery_lat"]),
          ", ".join(f"{c} {carried_missing[c][0]:,}" for c in ["delivery_person_age", "delivery_person_rating", "restaurant_lat"]))
    check("no pickup-derived column present", not any("pickup_time" in c or "pickup_datetime" in c for c in fe.columns))
    check("city_code parsed for every row (22 codes)", fe["city_code"].notna().all() and fe["city_code"].nunique() == 22,
          f"{fe['city_code'].nunique()} codes")

    # train-only preprocessing
    tr, te, cut = chronological_split(fe)
    dtr, dte = pd.to_datetime(fe.loc[tr, "order_date"]), pd.to_datetime(fe.loc[te, "order_date"])
    check("chronological split on whole dates (all train dates < all test dates)",
          dtr.max() < dte.min() and len(tr) + len(te) == len(fe),
          f"cut {cut}; train {len(tr):,} / test {len(te):,}")
    pre = build_preprocessor()
    Xtr, _ = get_model_inputs(fe.loc[tr]); Xte, _ = get_model_inputs(fe.loc[te])
    pre.fit(Xtr)
    imp = pre.named_transformers_["num"].named_steps["impute"].statistics_
    train_med = Xtr[PRIMARY_NUMERIC].astype("float64").median().values
    full_med = fe[PRIMARY_NUMERIC].astype("float64").median().values
    check("imputer statistics equal TRAIN medians", np.allclose(imp, train_med),
          f"differs from full-data medians on {int((~np.isclose(train_med, full_med)).sum())} feature(s)")
    cats = pre.named_transformers_["cat"].named_steps["onehot"].categories_
    train_levels = all(set(map(str, lv)) <= set(map(str, Xtr[c].dropna().unique()))
                       for c, lv in zip(PRIMARY_CATEGORICAL, cats))
    check("one-hot categories learned from TRAIN levels only", train_levels)
    Zte = pre.transform(Xte)
    check("transform on test runs without NaN/inf (unseen levels ignored)",
          np.isfinite(Zte).all(), f"{Zte.shape[1]} encoded columns")
    unseen = {c: sorted(set(Xte[c].dropna().astype(str)) - set(Xtr[c].dropna().astype(str)))
              for c in PRIMARY_CATEGORICAL}
    unseen = {k: v for k, v in unseen.items() if v}
    out.append({"check": "_unseen_test_levels", "passed": True, "detail": json.dumps(unseen)})
    return out


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S")
    return buf.getvalue().encode("utf-8")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_dictionary(fe: pd.DataFrame, path: Path = OUT_DICT) -> None:
    L = ["# DeliveryIQ — Feature Dictionary\n",
         "Generated by `src/feature_engineering.py` for `data/processed/feature_engineered_data.csv`.\n",
         f"**Primary model inputs ({len(PRIMARY_FEATURES)}):** all available at **order placement**. "
         "Build them only with `get_model_inputs()`, which enforces the leakage guard.\n",
         "| Column | Type | Missing | Description | Available at | Primary model |", "|---|---|---|---|---|---|"]
    order = PRIMARY_NUMERIC + PRIMARY_CATEGORICAL + [c for c in fe.columns if c not in PRIMARY_FEATURES]
    for c in order:
        desc, avail, status = DICTIONARY[c]
        kind = "numeric" if c in PRIMARY_NUMERIC else "categorical" if c in PRIMARY_CATEGORICAL else str(fe[c].dtype)
        L.append(f"| `{c}` | {kind} | {int(fe[c].isna().sum()):,} | {desc} | {avail} | {status} |")
    L += ["\n## Leakage guard rules\n",
          f"- **Forbidden always:** {', '.join(f'`{c}`' for c in sorted(FORBIDDEN_EXACT))}; any column matching "
          f"{', '.join(f'`{p}`' for p in FORBIDDEN_PATTERNS)}.",
          f"- **Excluded from the primary model (owner decision):** {', '.join(f'`{c}`' for c in EXCLUDED_BY_DECISION)}.",
          "- Missing values are kept in the dataset. Imputation happens only inside `build_preprocessor()`, "
          "fitted on the training split.",
          "- No missing-indicator features (e.g. Festival-missing, order-time-missing) are created."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_report(m: dict, path: Path = OUT_REPORT) -> None:
    checks = [c for c in m["checks"] if not c["check"].startswith("_")]
    unseen = json.loads(next(c for c in m["checks"] if c["check"] == "_unseen_test_levels")["detail"])
    L = ["# DeliveryIQ — Stage 3 Feature Engineering Report\n",
         f"**Generated by:** `src/feature_engineering.py` on {m['run_date']} · **Status:** awaiting owner review  ",
         f"**Input:** `data/processed/cleaned_data.csv` (`{m['clean_sha256'][:12]}…`, unchanged: {m['clean_unchanged']}) · "
         f"raw file unchanged: {m['raw_unchanged']}\n",
         "## 1. Summary\n",
         "| Item | Value |\n|---|---|",
         f"| Rows in / out | {m['rows_in']:,} / {m['rows_out']:,} |",
         f"| Columns in engineered dataset | {m['cols_out']} |",
         f"| Primary model features | **{len(PRIMARY_FEATURES)}** ({len(PRIMARY_NUMERIC)} numeric, {len(PRIMARY_CATEGORICAL)} categorical) |",
         f"| Encoded width after one-hot (fitted on train) | {m['encoded_width']} |",
         f"| Validation checks passed | {m['checks_passed']} / {m['checks_total']} |",
         f"| Deterministic output | {m['deterministic']} |\n",
         "## 2. Decisions applied [owner decisions, 2026-10-31]\n",
         "- **Primary model = order-placement (Tier A) features only.**",
         f"- **Excluded from the primary model** (kept in the dataset for later, labelled comparisons): "
         f"{', '.join(f'`{c}`' for c in EXCLUDED_BY_DECISION)}.",
         "- **Never inputs:** pickup time/datetime, `order_id`, `delivery_person_id`, target, `*_raw`, `dq_*`. "
         "`*_raw` and pickup columns are not carried into this dataset at all.",
         "- **Missing stays missing:** no zeros, no guesses. Age, rating, coordinates and the 1,731 missing order "
         "times are untouched. Order time is not reconstructed from pickup time.",
         "- **Interpretation choices (open to veto):**",
         "  - The \"date\" feature is `order_day_of_month`. `order_date` itself is the split key, not an input.",
         "  - `city_code` (from the `delivery_person_id` prefix) is **kept in the dataset but excluded from the "
         "primary model** after the Stage 4 availability review: the data has no independent order-time city field. "
         "On geo-valid rows the city is fully implied by the restaurant coordinates (already features); on the 3,640 "
         "geo-invalid rows the rider-ID prefix is the only source.",
         "  - Delivery coordinates are kept but not selected: they are synthetic offsets whose information "
         "`distance_km` already carries.\n",
         "## 3. Features created\n",
         "| Feature | Rule | Missing |\n|---|---|---|"]
    rules = {"distance_km": "Haversine(restaurant, delivery) on `dq_geo_valid` rows only",
             "city_code": "regex `^([A-Z]+)RES\\d+DEL\\d+$` on `delivery_person_id`",
             "order_hour": "hour of `order_datetime`",
             "order_hour_sin": "sin(2π·hour/24)", "order_hour_cos": "cos(2π·hour/24)",
             "order_time_band": "night 00–07 · morning 08–10 · midday 11–14 · afternoon 15–18 · evening 19–23",
             "order_day_of_week": "`order_date` weekday, 0=Mon", "order_day_of_month": "`order_date` day",
             "order_month": "`order_date` month", "order_is_weekend": "Sat/Sun → 1"}
    for f in NEW:
        L.append(f"| `{f}` | {rules[f]} | {m['missing'][f]:,} |")
    L += ["\nFull column list with availability: `reports/feature_dictionary.md`.\n",
          "## 4. Train-only preprocessing\n",
          "`build_preprocessor()` returns an **unfitted** sklearn `ColumnTransformer`:",
          "- **Numeric:** median imputation + standard scaling.",
          "- **Categorical:** most-frequent imputation + one-hot, with unknown levels ignored.",
          "- No missing-indicator columns.\n",
          f"`chronological_split()` cuts on whole `order_date`s. In the validation run, the default 80/20 cut starts "
          f"the test set on **{m['split']['cut']}** (train {m['split']['train']:,}, test {m['split']['test']:,} rows). "
          "The checks prove that imputer statistics and one-hot levels come from the training rows only. "
          "The final cut is confirmed in Stage 4.\n",
          "## 5. Validation checks\n",
          "| # | Check | Result | Detail |\n|---|---|---|---|"]
    for i, c in enumerate(checks, 1):
        L.append(f"| {i} | {c['check']} | {'✅ pass' if c['passed'] else '❌ FAIL'} | {c['detail']} |")
    L += [f"| {len(checks) + 1} | raw and cleaned files unchanged (SHA-256) | "
          f"{'✅ pass' if m['raw_unchanged'] and m['clean_unchanged'] else '❌ FAIL'} | |",
          f"| {len(checks) + 2} | deterministic output (two runs, identical bytes) | "
          f"{'✅ pass' if m['deterministic'] else '❌ FAIL'} | sha256 `{m['output_sha256'][:12]}…` |\n",
          "## 6. Remaining limitations\n",
          f"- **Unseen levels in the example test split:** {unseen or 'none'}. Months are 2–4 and the test period is "
          "the latest dates, so `order_month` (and some `order_day_of_month` values) are partly unseen in training. "
          "One-hot ignores unseen levels, so the month feature adds little and may shift predictions. "
          "Stage 4 should test the model with and without `order_month`.",
          f"- **{m['missing']['distance_km']:,} rows ({100 * m['missing']['distance_km'] / m['rows_out']:.2f}%) have no distance** "
          "(geo-invalid) and **1,731 have no order hour**. A model-time median fill treats them as typical orders.",
          "- **Weak primary signal expected:** the strongest audit predictors (traffic, weather, multiple deliveries, "
          "rider fields) are excluded by design. The primary model trades accuracy for realistic order-time availability. "
          "Stage 4 can report a clearly-labelled comparison model that uses them.",
          "- **Synthetic geography:** distance takes discrete bands (delivery points are diagonal offsets), and "
          "restaurant coordinates may let models learn restaurant-level averages.",
          "- **Hour features overlap traffic:** they are near-duplicates of the excluded `traffic_density`. If traffic "
          "is later allowed, treat the two as collinear.",
          "- Missing-Festival (228) and missing-timestamp (1,731) rows are kept; with/without sensitivity runs belong to Stage 4.\n",
          "## 7. Reproduce\n",
          "```powershell\ncd D:\\Project1\n.venv\\Scripts\\python.exe -m src.feature_engineering\n```"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def run() -> dict:
    """Load → build → validate → write. Raises if any check fails (nothing is written)."""
    raw_before, clean_before = sha256_file(RAW_PATH), sha256_file(CLEAN_CSV)
    if raw_before != RAW_SHA256:
        raise ValueError("raw file hash mismatch")
    clean = load_cleaned()
    fe = build_features(clean)
    checks = validate(clean, fe)
    payload = to_csv_bytes(fe)
    deterministic = payload == to_csv_bytes(build_features(load_cleaned()))
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed or not deterministic:
        raise RuntimeError(f"Validation failed: {failed or ['non-deterministic']}. Nothing written.")

    if not (OUT_CSV.exists() and OUT_CSV.read_bytes() == payload):
        OUT_CSV.write_bytes(payload)
    tr, te, cut = chronological_split(fe)
    pre = build_preprocessor().fit(get_model_inputs(fe.loc[tr])[0])
    m = {
        "run_date": date.today().isoformat(),
        "rows_in": len(clean), "rows_out": len(fe), "cols_out": fe.shape[1],
        "primary_features": PRIMARY_FEATURES, "primary_numeric": PRIMARY_NUMERIC,
        "primary_categorical": PRIMARY_CATEGORICAL, "excluded_by_decision": EXCLUDED_BY_DECISION,
        "encoded_width": int(len(pre.get_feature_names_out())),
        "missing": {c: int(fe[c].isna().sum()) for c in fe.columns},
        "split": {"cut": cut, "train": len(tr), "test": len(te)},
        "checks": checks,
        "checks_passed": sum(c["passed"] for c in checks if not c["check"].startswith("_")),
        "checks_total": sum(1 for c in checks if not c["check"].startswith("_")),
        "deterministic": deterministic,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "clean_sha256": clean_before,
        "raw_unchanged": sha256_file(RAW_PATH) == raw_before == RAW_SHA256,
        "clean_unchanged": sha256_file(CLEAN_CSV) == clean_before,
    }
    if not (m["raw_unchanged"] and m["clean_unchanged"]):
        raise RuntimeError("Input file changed during the run.")
    OUT_METRICS.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    write_dictionary(fe)
    write_report(m)
    return m


def main() -> None:
    """CLI entry point: ``python -m src.feature_engineering``."""
    m = run()
    print(f"rows: {m['rows_out']:,} | columns: {m['cols_out']} | primary features: {len(m['primary_features'])} "
          f"(encoded width {m['encoded_width']})")
    print(f"validation: {m['checks_passed']}/{m['checks_total']} passed | deterministic: {m['deterministic']} | "
          f"raw unchanged: {m['raw_unchanged']} | cleaned unchanged: {m['clean_unchanged']}")
    for c in m["checks"]:
        if not c["check"].startswith("_"):
            print(("  PASS " if c["passed"] else "  FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] else ""))
    print("unseen test levels:", next(c for c in m["checks"] if c["check"] == "_unseen_test_levels")["detail"])


if __name__ == "__main__":
    main()
