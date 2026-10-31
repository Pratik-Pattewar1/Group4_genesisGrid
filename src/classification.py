"""Late-delivery risk classification for DeliveryIQ (Stage 4).

Late label: project-defined benchmark ``time_taken_min > 30`` (NOT an official SLA; approved
2026-10-31), with sensitivity analysis at > 35 and > 40 minutes.

Candidates (preprocessing fitted on training rows only):
    * ``baseline_prior`` — constant training prevalence (ROC-AUC 0.5 by construction)
    * ``logistic``       — L2 logistic regression
    * ``hist_gbm``       — histogram gradient boosting

Selection metric: chronological-CV PR-AUC on the training split. The decision threshold is chosen
on out-of-fold training predictions (see ``src.evaluation.select_threshold``), never on test data.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

SEED = 42
LATE_THRESHOLDS = (30, 35, 40)

GRIDS: dict[str, list[dict]] = {
    "baseline_prior": [{}],
    "logistic": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "hist_gbm": [{"learning_rate": lr, "max_leaf_nodes": ml, "l2_regularization": l2}
                 for lr, ml, l2 in product((0.05, 0.1), (15, 31), (0.0, 1.0))],
}


def make_late_label(y_minutes: pd.Series, threshold_minutes: int) -> pd.Series:
    """1 if delivery took strictly more than ``threshold_minutes`` (project benchmark), else 0."""
    if threshold_minutes not in LATE_THRESHOLDS:
        raise ValueError(f"Threshold {threshold_minutes} is not an approved benchmark {LATE_THRESHOLDS}")
    return (y_minutes > threshold_minutes).astype(int)


def make_estimator(name: str, params: dict | None = None):
    """Return an unfitted classifier for ``name`` with ``params``."""
    params = params or {}
    if name == "baseline_prior":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(max_iter=3000, **params)
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(max_iter=600, early_stopping=True, validation_fraction=0.1,
                                              n_iter_no_change=25, random_state=SEED, **params)
    raise ValueError(name)


def make_pipeline(preprocessor, name: str, params: dict | None = None) -> Pipeline:
    """Preprocessor + classifier. The whole pipeline is fitted on training rows only."""
    return Pipeline([("prep", clone(preprocessor)), ("model", make_estimator(name, params))])


def cv_oof(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds, train_mask: np.ndarray | None = None) -> dict:
    """Chronological CV: mean ROC-AUC / PR-AUC plus out-of-fold probabilities (validation blocks only)."""
    oof_idx, oof_p, rocs, prs = [], [], [], []
    for tr, va in folds:
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        rocs.append(roc_auc_score(y.iloc[va], p)); prs.append(average_precision_score(y.iloc[va], p))
        oof_idx.append(va); oof_p.append(p)
    return {"ROC_AUC": float(np.mean(rocs)), "PR_AUC": float(np.mean(prs)),
            "oof_index": np.concatenate(oof_idx), "oof_proba": np.concatenate(oof_p)}


def tune(preprocessor, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    """Grid-search every candidate by chronological-CV PR-AUC (training data only)."""
    out = {}
    for name, grid in GRIDS.items():
        trials = []
        for p in grid:
            r = cv_oof(make_pipeline(preprocessor, name, p), X, y, folds)
            trials.append({"params": p, **r})
        best = max(trials, key=lambda t: t["PR_AUC"])
        out[name] = {"best_params": best["params"], "cv": {"ROC_AUC": best["ROC_AUC"], "PR_AUC": best["PR_AUC"]},
                     "oof_index": best["oof_index"], "oof_proba": best["oof_proba"],
                     "trials": [{"params": t["params"], "PR_AUC": t["PR_AUC"], "ROC_AUC": t["ROC_AUC"]} for t in trials]}
    return out
