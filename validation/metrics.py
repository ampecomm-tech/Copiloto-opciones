"""Métricas de validación: regresión (RMSE, MSE, MAE, MAPE, SMAPE, Variance Score) y clasificación."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, explained_variance_score, f1_score, precision_score, recall_score


def metricas_regresion(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MSE, MAE, MAPE, SMAPE y Variance Score (explained variance)."""
    y, p = np.asarray(y_real, float), np.asarray(y_pred, float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {k: float("nan") for k in ("rmse", "mse", "mae", "mape", "smape", "variance_score")}
    err = y - p
    mse = float(np.mean(err**2))
    denom_mape = np.where(np.abs(y) < 1e-12, np.nan, np.abs(y))
    mape = float(np.nanmean(np.abs(err) / denom_mape) * 100)
    denom_smape = (np.abs(y) + np.abs(p))
    smape = float(np.mean(np.where(denom_smape < 1e-12, 0.0, 2 * np.abs(err) / np.where(denom_smape < 1e-12, 1, denom_smape))) * 100)
    return {"rmse": float(np.sqrt(mse)), "mse": mse, "mae": float(np.mean(np.abs(err))), "mape": mape, "smape": smape,
            "variance_score": float(explained_variance_score(y, p))}


def metricas_clasificacion(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 y matriz de confusión [[TN, FP], [FN, TP]]."""
    y, p = np.asarray(y_real, int), np.asarray(y_pred, int)
    if len(y) == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "matriz_confusion": [[0, 0], [0, 0]]}
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "matriz_confusion": cm.tolist(),
    }
