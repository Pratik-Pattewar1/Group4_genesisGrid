"""Geographic delivery-zone clustering for DeliveryIQ (Stage 4).

K-Means on the delivery coordinates of geo-valid orders (``dq_geo_valid``), projected to an
approximately metric km grid so that Euclidean distance is meaningful. The number of zones is
chosen from elbow (inertia) and silhouette analysis. The model is fitted on TRAINING-period rows only;
zone profiles are descriptive.

Rider segmentation is NOT performed: the audit showed ``delivery_person_id`` does not identify a
person (each ID has 4–22 different ages), so person-level segments would be meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


def project_km(lat: pd.Series, lon: pd.Series, ref_lat: float) -> np.ndarray:
    """Equirectangular projection to km around ``ref_lat`` (adequate at city/country scale)."""
    x = lon.to_numpy(float) * KM_PER_DEG_LON_EQ * np.cos(np.radians(ref_lat))
    y = lat.to_numpy(float) * KM_PER_DEG_LAT
    return np.column_stack([x, y])


def scan_k(X: np.ndarray, k_values, silhouette_sample: int = 10_000) -> pd.DataFrame:
    """Inertia (elbow) and sampled silhouette score for each k."""
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(silhouette_sample, len(X)), random_state=SEED)
        rows.append({"k": int(k), "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)


def elbow_k(scan: pd.DataFrame) -> int:
    """Elbow by maximum distance to the line joining the first and last inertia points (normalised)."""
    k = scan["k"].to_numpy(float); w = scan["inertia"].to_numpy(float)
    kn = (k - k.min()) / (k.max() - k.min()); wn = (w - w.min()) / (w.max() - w.min())
    dist = np.abs(wn - (1 - kn)) / np.sqrt(2)          # line from (0,1) to (1,0)
    return int(scan["k"].iloc[int(np.argmax(dist))])


def fit_zones(X: np.ndarray, k: int) -> KMeans:
    """Fit the final K-Means zone model."""
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)


def profile_zones(df: pd.DataFrame, labels: np.ndarray, late_minutes: int = 30) -> pd.DataFrame:
    """Per-zone volume, ETA and late-rate profile (descriptive)."""
    d = df.assign(zone=labels)
    g = d.groupby("zone")
    prof = pd.DataFrame({
        "orders": g.size(),
        "share_pct": (100 * g.size() / len(d)).round(2),
        "median_eta_min": g["time_taken_min"].median(),
        "mean_eta_min": g["time_taken_min"].mean().round(2),
        f"late_gt{late_minutes}_pct": (100 * g["time_taken_min"].apply(lambda s: (s > late_minutes).mean())).round(2),
        "mean_distance_km": g["distance_km"].mean().round(2),
        "centroid_lat": g["delivery_lat"].mean().round(4),
        "centroid_lon": g["delivery_lon"].mean().round(4),
        "dominant_city_code": g["city_code"].agg(lambda s: s.value_counts().index[0]),
        "n_city_codes": g["city_code"].nunique(),
    })
    return prof.reset_index()
