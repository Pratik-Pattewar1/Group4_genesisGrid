"""DeliveryIQ — order-placement prediction service (used by the Streamlit demo).

Turns a user-entered order into the exact 13-feature row the saved Stage 4 models were trained on,
reusing the Stage 3 feature logic (Haversine distance, hour encodings, time bands) and the Stage 3
``get_model_inputs`` dtype conversion, so the app and the training pipeline cannot drift apart.

Only order-placement information is accepted. Pickup time, rider details, vehicle, batching, weather,
traffic and every audit / post-delivery column are rejected by the leakage guard.

No model is trained or modified here: the saved pipelines are loaded read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import (OUT_CSV as FE_CSV, TARGET, TIME_BANDS, assert_no_leakage,
                                     chronological_split, get_model_inputs, haversine_km)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
INDIA_LAT, INDIA_LON = (6.0, 37.5), (68.0, 97.5)
MAX_DISTANCE_KM = 25.0            # validation limit used since Stage 2
CITY_TYPES = ["Urban", "Metropolitan", "Semi-Urban"]
ORDER_TYPES = ["Snack", "Meal", "Drinks", "Buffet"]
FESTIVAL = ["No", "Yes"]
UNKNOWN = "Unknown"


@dataclass
class OrderInput:
    """An order as known at placement time. ``None`` means 'not known' (filled by the model's
    training-fitted imputer, never by a guessed value)."""
    restaurant_lat: float | None
    restaurant_lon: float | None
    order_date: date
    order_time: time | None
    city_type: str
    order_type: str
    festival: str
    customer_lat: float | None = None
    customer_lon: float | None = None
    distance_km: float | None = None          # used only when customer coordinates are not given


@dataclass
class Prediction:
    eta_min: float
    late_probability: float
    risk_level: str
    flagged: bool
    distance_km: float | None
    warnings: list[str] = field(default_factory=list)
    drivers: list[dict] = field(default_factory=list)
    features: dict = field(default_factory=dict)


class OrderValidationError(ValueError):
    """Raised when an input cannot be used safely (e.g. coordinates outside the data's area)."""


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load the saved primary models (read-only) and check they share the 13-feature contract."""
    eta = joblib.load(MODELS / "eta_regressor.joblib")
    late = joblib.load(MODELS / "late_classifier_gt30.joblib")
    if eta["features"] != late["features"]:
        raise RuntimeError("ETA and late-risk models disagree on the feature list")
    assert_no_leakage(eta["features"])                    # strict guard: order-placement only
    card = json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))
    return {"eta": eta, "late": late, "features": eta["features"], "card": card}


@lru_cache(maxsize=1)
def reference() -> dict:
    """Training-period reference data for the app: store locations, risk bands and a typical order.
    Computed from the TRAINING split only (never from test data)."""
    fe = pd.read_csv(FE_CSV)
    tr, _, _ = chronological_split(fe)
    train = fe.loc[tr]
    m = load_models()
    X, _ = get_model_inputs(train, m["features"])
    p_train = m["late"]["pipeline"].predict_proba(X)[:, 1]
    stores = (fe[fe["dq_geo_valid"]].groupby(["city_code", "restaurant_lat", "restaurant_lon"]).size()
              .rename("orders").reset_index().sort_values(["city_code", "orders"], ascending=[True, False]))
    typical = {
        "distance_km": float(train["distance_km"].median()), "order_hour": int(train["order_hour"].median()),
        "city_type": train["city_type"].mode().iloc[0], "festival": train["festival"].mode().iloc[0],
        "order_type": train["order_type"].mode().iloc[0], "order_day_of_week": int(train["order_day_of_week"].mode().iloc[0]),
        "order_day_of_month": int(train["order_day_of_month"].median()),
    }
    return {
        "stores": stores,
        "high_risk_cutoff": float(np.quantile(p_train, 0.90)),     # riskiest 10% of training orders
        "decision_threshold": float(m["late"]["decision_threshold"]),
        "typical": typical,
        "train_distance_range": [float(train["distance_km"].min()), float(train["distance_km"].max())],
        "train_hours": sorted(int(h) for h in train["order_hour"].dropna().unique()),
        "train_dates": m["eta"]["trained_on"],
    }


def _time_band(hour: int) -> str:
    return next(name for lo, hi, name in TIME_BANDS if lo <= hour < hi)


def _in_india(lat, lon) -> bool:
    return INDIA_LAT[0] <= lat <= INDIA_LAT[1] and INDIA_LON[0] <= lon <= INDIA_LON[1]


def build_feature_row(order: OrderInput) -> tuple[pd.DataFrame, list[str], float | None]:
    """Validate an order and build the 13-feature model input (same transformations as Stage 3).

    Returns (X, warnings, distance_km). Raises OrderValidationError for unusable inputs."""
    warn: list[str] = []
    ref = reference()

    # --- store location ---
    rlat, rlon = order.restaurant_lat, order.restaurant_lon
    if (rlat is None) != (rlon is None):
        raise OrderValidationError("Enter both store latitude and longitude, or neither.")
    if rlat is not None and not _in_india(rlat, rlon):
        raise OrderValidationError("Store coordinates are outside the area covered by the data (India).")
    if rlat is None:
        warn.append("Store location unknown: the model uses its training median location, and distance must be entered directly.")

    # --- distance (Haversine from store to customer, as in Stage 3) ---
    dist = None
    if order.customer_lat is not None or order.customer_lon is not None:
        if order.customer_lat is None or order.customer_lon is None:
            raise OrderValidationError("Enter both customer latitude and longitude, or neither.")
        if not _in_india(order.customer_lat, order.customer_lon):
            raise OrderValidationError("Customer coordinates are outside the area covered by the data (India).")
        if rlat is None:
            raise OrderValidationError("Customer coordinates need a store location to compute distance.")
        dist = float(haversine_km(rlat, rlon, order.customer_lat, order.customer_lon))
    elif order.distance_km is not None:
        dist = float(order.distance_km)
    if dist is not None:
        if not np.isfinite(dist) or dist <= 0:
            raise OrderValidationError("Delivery distance must be greater than 0 km.")
        if dist > MAX_DISTANCE_KM:
            raise OrderValidationError(f"Delivery distance {dist:.1f} km exceeds the {MAX_DISTANCE_KM:.0f} km limit of the data.")
        lo, hi = ref["train_distance_range"]
        if dist < lo or dist > hi:
            warn.append(f"Distance {dist:.1f} km is outside the training range ({lo:.1f}–{hi:.1f} km); the prediction is an extrapolation.")
    else:
        warn.append("Delivery distance unknown: the model uses its training median distance.")

    # --- categorical inputs ---
    def cat(value, allowed, name):
        if value in (None, "", UNKNOWN):
            warn.append(f"{name} unknown: the model uses its most frequent training value.")
            return np.nan
        if value not in allowed:
            raise OrderValidationError(f"Invalid {name}: {value!r}. Allowed: {', '.join(allowed)}.")
        return value
    city_type = cat(order.city_type, CITY_TYPES, "City type")
    order_type = cat(order.order_type, ORDER_TYPES, "Order type")
    festival = cat(order.festival, FESTIVAL, "Festival")
    if city_type == "Semi-Urban":
        warn.append("Semi-Urban orders are rare in the data (164 orders, all late): treat this prediction with caution.")
    if festival == "Yes":
        warn.append("Every festival-day order in the data was late (a synthetic-data pattern): treat this prediction with caution.")

    # --- time & calendar (as Stage 3) ---
    if not isinstance(order.order_date, date):
        raise OrderValidationError("A valid order date is required.")
    if order.order_time is None:
        hour = np.nan
        warn.append("Order time unknown: hour features are filled by the model's training values (less accurate).")
    else:
        hour = int(order.order_time.hour)
        if hour not in ref["train_hours"]:
            warn.append(f"No training orders were placed at {hour:02d}:00 (data has none between 01:00 and 07:59); the prediction is an extrapolation.")
    dow = order.order_date.weekday()

    # Rounding mirrors Stage 3 exactly (distance to 4 dp, hour sin/cos to 6 dp).
    row = {
        "distance_km": round(dist, 4) if dist is not None else np.nan,
        "restaurant_lat": rlat if rlat is not None else np.nan,
        "restaurant_lon": rlon if rlon is not None else np.nan,
        "order_hour": hour,
        "order_hour_sin": round(float(np.sin(2 * np.pi * hour / 24)), 6) if hour == hour else np.nan,
        "order_hour_cos": round(float(np.cos(2 * np.pi * hour / 24)), 6) if hour == hour else np.nan,
        "order_day_of_month": order.order_date.day,
        "order_is_weekend": int(dow >= 5),
        "city_type": city_type, "order_type": order_type, "festival": festival,
        "order_time_band": _time_band(hour) if hour == hour else np.nan,
        "order_day_of_week": dow,
    }
    features = load_models()["features"]
    frame = pd.DataFrame([row])[features].assign(**{TARGET: np.nan})
    X, _ = get_model_inputs(frame, features)              # identical dtype handling to training
    # A one-row categorical column holding only a blank is inferred as float64. Keep categorical columns
    # as object (strings + np.nan), as in training, so 'unknown' inputs reach the imputer correctly.
    cat_cols = load_models()["eta"]["pipeline"].named_steps["prep"].transformers_[1][2]
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X, warn, dist


def risk_level(p: float) -> str:
    """Risk tiers: High = riskiest 10% of training orders; Elevated = above the decision threshold."""
    ref = reference()
    if p >= ref["high_risk_cutoff"]:
        return "High"
    if p >= ref["decision_threshold"]:
        return "Elevated"
    return "Low"


def _drivers(X: pd.DataFrame) -> list[dict]:
    """Per-order model sensitivity: how much the ETA and risk change if one factor group is reset to a
    typical training order. Describes model behaviour, not causation."""
    m, t = load_models(), reference()["typical"]
    eta, late = m["eta"]["pipeline"], m["late"]["pipeline"]
    base_eta, base_p = float(eta.predict(X)[0]), float(late.predict_proba(X)[0, 1])
    h = t["order_hour"]
    groups = {
        f"Time of day (vs {h:02d}:00)": {"order_hour": float(h), "order_hour_sin": np.sin(2 * np.pi * h / 24),
                                          "order_hour_cos": np.cos(2 * np.pi * h / 24), "order_time_band": _time_band(h)},
        f"Distance (vs {t['distance_km']:.1f} km)": {"distance_km": t["distance_km"]},
        f"Festival (vs {t['festival']})": {"festival": t["festival"]},
        f"City type (vs {t['city_type']})": {"city_type": t["city_type"]},
        f"Order type (vs {t['order_type']})": {"order_type": t["order_type"]},
        "Calendar (vs typical weekday)": {"order_day_of_week": str(t["order_day_of_week"]),
                                          "order_day_of_month": float(t["order_day_of_month"]), "order_is_weekend": 0.0},
    }
    out = []
    for name, vals in groups.items():
        Xr = X.copy()
        for k, v in vals.items():
            Xr[k] = v
        out.append({"factor": name, "eta_effect_min": base_eta - float(eta.predict(Xr)[0]),
                    "risk_effect_pp": 100 * (base_p - float(late.predict_proba(Xr)[0, 1]))})
    return sorted(out, key=lambda d: -abs(d["eta_effect_min"]))


def predict_order(order: OrderInput) -> Prediction:
    """Validate, build features and return ETA, late-risk score, tier and per-order drivers."""
    X, warn, dist = build_feature_row(order)
    m = load_models()
    eta = float(m["eta"]["pipeline"].predict(X)[0])
    p = float(m["late"]["pipeline"].predict_proba(X)[0, 1])
    return Prediction(eta_min=eta, late_probability=p, risk_level=risk_level(p),
                      flagged=p >= reference()["decision_threshold"], distance_km=dist, warnings=warn,
                      drivers=_drivers(X), features=X.iloc[0].to_dict())
