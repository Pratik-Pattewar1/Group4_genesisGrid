"""ETA regression models for DeliveryIQ (Stage 4).

Candidates (each wrapped in a pipeline whose preprocessing is fitted on the training rows only):
    * ``baseline_median`` — predicts the training-median ETA (mandatory naive baseline)
    * ``ridge``           — regularised linear regression
    * ``random_forest``   — bagged trees
    * ``hist_gbm``        — histogram gradient boosting

Hyper-parameters are chosen by chronological cross-validation on the training split only.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.evaluation import regression_report

SEED = 42

GRIDS: dict[str, list[dict]] = {
    "baseline_median": [{}],
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (5, 20)],
    "hist_gbm": [{"learning_rate": lr, "max_leaf_nodes": ml, "l2_regularization": l2}
                 for lr, ml, l2 in product((0.05, 0.1), (15, 31), (0.0, 1.0))],
}


def make_estimator(name: str, params: dict | None = None):
    """Return an unfitted regressor for ``name`` with ``params``."""
    params = params or {}
    if name == "baseline_median":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=SEED, **params)
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(max_iter=600, early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=25, random_state=SEED, **params)
    raise ValueError(name)


def make_pipeline(preprocessor, name: str, params: dict | None = None) -> Pipeline:
    """Preprocessor + regressor. The whole pipeline is fitted on training rows only."""
    return Pipeline([("prep", clone(preprocessor)), ("model", make_estimator(name, params))])


def cv_score(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds, train_mask: np.ndarray | None = None) -> dict:
    """Mean chronological-CV metrics. ``train_mask`` optionally drops rows from each fold's TRAINING
    portion only (validation folds stay complete), for with/without-subset comparisons."""
    res = []
    for tr, va in folds:
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        res.append(regression_report(y.iloc[va], m.predict(X.iloc[va])))
    return {k: float(np.mean([r[k] for r in res])) for k in ("MAE", "RMSE", "R2")} | {"folds": res}


def tune(preprocessor, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    """Grid-search every candidate by chronological CV MAE (training data only)."""
    out = {}
    for name, grid in GRIDS.items():
        trials = [{"params": p, **cv_score(make_pipeline(preprocessor, name, p), X, y, folds)} for p in grid]
        best = min(trials, key=lambda t: t["MAE"])
        out[name] = {"best_params": best["params"], "cv": {k: best[k] for k in ("MAE", "RMSE", "R2")},
                     "trials": [{"params": t["params"], "MAE": t["MAE"]} for t in trials]}
    return out
