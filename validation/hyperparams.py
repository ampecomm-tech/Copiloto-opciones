"""Simulación de hiperparámetros del RandomForest y tabla comparativa.

Para cada combinación del grid se entrena con los primeros 70 % de las velas
(orden temporal) y se evalúa en el 30 % restante con métricas de clasificación
(sobre la dirección) y de regresión (probabilidad vs retorno futuro normalizado).
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from core.config import cfg
from validation.metrics import metricas_clasificacion, metricas_regresion


def dividir_temporal(X: pd.DataFrame, y: pd.Series, frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición temporal (sin barajar) entrenamiento/prueba."""
    n = int(len(X) * frac)
    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]


def evaluar_modelo(modelo: Any, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series, y_te: pd.Series, ret_te: pd.Series) -> dict[str, Any]:
    """Entrena y evalúa un clasificador; devuelve métricas y las probabilidades de prueba."""
    modelo.fit(X_tr, y_tr)
    prob = modelo.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    clas = metricas_clasificacion(y_te.values, pred)
    # Regresión: comparamos (prob − 0.5) con el retorno futuro, ambos como "señal" en [-0.5, 0.5]
    objetivo = np.clip(ret_te.values / (2 * np.nanstd(ret_te.values) + 1e-12), -0.5, 0.5)
    reg = metricas_regresion(objetivo, prob - 0.5)
    return {**clas, **reg, "prob": prob, "pred": pred, "error": objetivo - (prob - 0.5)}


def simular_grid(X: pd.DataFrame, y: pd.Series, ret: pd.Series, grid: dict[str, list[Any]] | None = None, semilla: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recorre el grid de hiperparámetros del RF y devuelve la tabla comparativa y el mejor.

    Returns:
        (tabla ordenada por F1 desc, mejor_fila como dict con `hyperparams` y `error`).
    """
    grid = grid or cfg("bot", "random_forest", "grid") or {"n_estimators": [100], "max_depth": [6], "min_samples_leaf": [5]}
    X_tr, X_te, y_tr, y_te = dividir_temporal(X, y)
    ret_te = ret.loc[X_te.index]
    filas: list[dict[str, Any]] = []
    claves = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in claves]):
        hp = dict(zip(claves, combo))
        rf = RandomForestClassifier(random_state=semilla, n_jobs=-1, **hp)
        res = evaluar_modelo(rf, X_tr, X_te, y_tr, y_te, ret_te)
        filas.append({**hp, **{k: v for k, v in res.items() if k not in ("prob", "pred", "error", "matriz_confusion")}, "matriz_confusion": res["matriz_confusion"], "_error": res["error"]})
    tabla = pd.DataFrame(filas).sort_values(["f1", "accuracy"], ascending=False).reset_index(drop=True)
    mejor = tabla.iloc[0].to_dict()
    mejor["hyperparams"] = {k: (int(mejor[k]) if mejor[k] == mejor[k] and k != "max_depth" or (k == "max_depth" and mejor[k] == mejor[k]) else None) for k in claves}
    mejor["hyperparams"] = {k: (None if pd.isna(v) else int(v)) for k, v in mejor["hyperparams"].items()}
    tabla_salida = tabla.drop(columns=["_error"])
    return tabla_salida, mejor


def comparar_con_knn(X: pd.DataFrame, y: pd.Series, ret: pd.Series, k: int = 15) -> dict[str, Any]:
    """Entrena un KNN como modelo base y devuelve sus métricas y errores."""
    X_tr, X_te, y_tr, y_te = dividir_temporal(X, y)
    knn = KNeighborsClassifier(n_neighbors=k)
    res = evaluar_modelo(knn, X_tr, X_te, y_tr, y_te, ret.loc[X_te.index])
    return {k_: v for k_, v in res.items() if k_ != "pred"}
