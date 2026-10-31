"""Shared evaluation utilities for DeliveryIQ (Stage 4).

* Chronological (forward-chaining) cross-validation folds built from the TRAINING split only.
* Regression metrics: MAE, RMSE, R².
* Classification metrics: precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix.
* Decision-threshold selection on out-of-fold training predictions (never on the test set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score)


def chronological_folds(order_date: pd.Series, n_folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds on whole dates within the training data.

    The training dates are cut into ``n_folds + 1`` contiguous blocks of roughly equal row counts.
    Fold *i* trains on blocks ``0..i`` and validates on block ``i + 1``, so validation is always later
    than training. Returns positional index arrays relative to ``order_date``.
    """
    d = pd.to_datetime(order_date).reset_index(drop=True)
    counts = d.value_counts().sort_index()
    cum = counts.cumsum() / counts.sum()
    block_of_date = np.minimum((cum.shift(fill_value=0) * (n_folds + 1)).astype(int), n_folds)
    block = d.map(block_of_date).to_numpy()
    folds = []
    for i in range(n_folds):
        tr, va = np.where(block <= i)[0], np.where(block == i + 1)[0]
        if len(tr) == 0 or len(va) == 0:
            raise ValueError("Empty chronological fold; reduce n_folds")
        folds.append((tr, va))
    return folds


def regression_report(y_true, y_pred) -> dict:
    """MAE, RMSE and R² (minutes)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)), "n": int(len(y_true))}


def classification_report(y_true, y_score, threshold: float) -> dict:
    """Threshold-dependent and threshold-free metrics for a binary late-delivery classifier."""
    y_true, y_score = np.asarray(y_true, int), np.asarray(y_score, float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y_true)) < 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "prevalence": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def select_threshold(y_true, y_score, min_recall: float = 0.80) -> dict:
    """Pick decision thresholds from OUT-OF-FOLD training predictions.

    * ``f1``: the threshold maximising F1 (the default operating point).
    * ``recall_target``: the highest threshold whose recall is still ≥ ``min_recall``
      (an alternative when missing a late order is costlier than a false alarm).
    """
    p, r, t = precision_recall_curve(np.asarray(y_true, int), np.asarray(y_score, float))
    p, r = p[:-1], r[:-1]                       # align with thresholds
    f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0)
    best = int(np.argmax(f1))
    ok = np.where(r >= min_recall)[0]
    rec_idx = int(ok[-1]) if len(ok) else 0
    return {"f1": {"threshold": float(t[best]), "precision": float(p[best]), "recall": float(r[best]), "F1": float(f1[best])},
            "recall_target": {"min_recall": min_recall, "threshold": float(t[rec_idx]),
                              "precision": float(p[rec_idx]), "recall": float(r[rec_idx]), "F1": float(f1[rec_idx])},
            "curve": {"threshold": t.tolist(), "precision": p.tolist(), "recall": r.tolist(), "F1": f1.tolist()}}
