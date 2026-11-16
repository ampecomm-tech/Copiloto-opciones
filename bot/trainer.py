"""Entrenamiento automático del RandomForest por activo + validación completa.

Pasos: descargar histórico → features → grid de hiperparámetros → KNN base →
errores de las estrategias técnicas → Diebold-Mariano → guardar `model_runs`
y el modelo final en `bot_models`.
"""
from __future__ import annotations

import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bot.features import construir_features, etiqueta_objetivo, retorno_futuro
from bot.strategies import EMA50Chandelier, MACDCross, RSIReversion
from core import db
from core.config import cfg
from core.events import emit
from data_sources.market import velas
from validation.diebold_mariano import diebold_mariano
from validation.hyperparams import comparar_con_knn, dividir_temporal, simular_grid
from validation.metrics import metricas_clasificacion

Progreso = Callable[[str, float], None]


def _errores_estrategia(estrategia: Any, df: pd.DataFrame, idx_prueba: pd.Index, ret: pd.Series, paso: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Errores de una estrategia técnica en el tramo de prueba (señal −0.5/0/0.5 vs retorno normalizado).

    Para no recalcular indicadores en cada vela, se evalúa cada `paso` velas.
    """
    objetivo = np.clip(ret.loc[idx_prueba].values / (2 * np.nanstd(ret.loc[idx_prueba].values) + 1e-12), -0.5, 0.5)
    pos = {ts: i for i, ts in enumerate(df.index)}
    senales, objs, preds = [], [], []
    for j, ts in enumerate(idx_prueba[::paso]):
        i = pos[ts]
        s = estrategia.generate_signal(df.iloc[max(0, i - 300): i + 1])
        val = 0.5 if s.action == "BUY" else -0.5 if s.action == "SELL" else 0.0
        senales.append(val)
        objs.append(objetivo[j * paso])
        preds.append(1 if val > 0 else 0)
    return np.asarray(objs) - np.asarray(senales), np.asarray(preds)


def entrenar_activo(asset_id: int, symbol: str, venue: str, sentimiento: float | None = None, progreso: Progreso | None = None) -> dict[str, Any]:
    """Entrena, valida y guarda el modelo de un activo. Devuelve el resumen de `model_runs`.

    Args:
        asset_id: Id en `bot_assets`.
        symbol: Símbolo del activo.
        venue: `binance_spot_testnet`, `alpaca_paper` o `paper`.
        sentimiento: Último puntaje de sentimiento (feature constante).
        progreso: Callback `(mensaje, fraccion)` para la UI.
    """
    def avisar(msg: str, frac: float) -> None:
        if progreso:
            progreso(msg, frac)

    avisar("Descargando histórico…", 0.05)
    db.ejecutar("UPDATE bot_assets SET training_status = 'entrenando' WHERE id = ?", (asset_id,))
    df = velas(symbol, venue)
    if len(df) < 300:
        db.ejecutar("UPDATE bot_assets SET training_status = 'error' WHERE id = ?", (asset_id,))
        emit("sistema", "error", f"Histórico insuficiente para {symbol}", {"velas": len(df)})
        raise RuntimeError(f"Histórico insuficiente ({len(df)} velas) para {symbol}")

    h = int(cfg("bot", "random_forest", "horizonte_velas", default=4))
    avisar("Calculando features…", 0.15)
    X = construir_features(df, sentimiento)
    y = etiqueta_objetivo(df, h)
    ret = retorno_futuro(df, h)
    validos = X.notna().all(axis=1) & y.notna()
    X, y, ret = X[validos], y[validos].astype(int), ret[validos]
    features = list(X.columns)

    avisar("Simulando hiperparámetros del RandomForest…", 0.3)
    tabla, mejor = simular_grid(X, y, ret)

    avisar("Entrenando KNN de referencia…", 0.55)
    knn = comparar_con_knn(X, y, ret)

    avisar("Evaluando estrategias técnicas y Diebold-Mariano…", 0.65)
    X_tr, X_te, _, _ = dividir_temporal(X, y)
    dm: dict[str, Any] = {"rf_vs_knn": diebold_mariano(mejor["_error"], knn["error"], h=h).to_dict()}
    metricas_estrategias: dict[str, Any] = {"knn": {k: v for k, v in knn.items() if k not in ("prob", "error")}}
    for est in (EMA50Chandelier(), RSIReversion(), MACDCross()):
        paso = max(1, len(X_te) // 400)
        err, pred = _errores_estrategia(est, df, X_te.index, ret, paso)
        err_rf = mejor["_error"][::paso][: len(err)]
        dm[f"rf_vs_{est.name}"] = diebold_mariano(err_rf, err, h=h).to_dict()
        y_te_sub = y.loc[X_te.index].values[::paso][: len(pred)]
        metricas_estrategias[est.name] = metricas_clasificacion(y_te_sub, pred)

    avisar("Entrenando modelo final…", 0.85)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **mejor["hyperparams"])
    rf.fit(X, y)
    importancias = sorted(zip(features, rf.feature_importances_.round(4)), key=lambda t: -t[1])[:10]

    metricas = {
        "mejor_rf": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in mejor.items() if k not in ("_error", "hyperparams")},
        "tabla_hiperparametros": tabla.to_dict(orient="records"),
        "estrategias": metricas_estrategias,
        "importancias": [{"feature": f, "importancia": float(i)} for f, i in importancias],
        "n_velas": int(len(df)), "n_entrenamiento": int(len(X_tr)), "n_prueba": int(len(X_te)), "horizonte_velas": h,
    }
    ts = db.ahora_iso()
    run_id = db.ejecutar("INSERT INTO model_runs(asset_id, ts, hyperparams_json, metrics_json, dm_results_json) VALUES (?,?,?,?,?)",
                         (asset_id, ts, db.a_json(mejor["hyperparams"]), db.a_json(metricas), db.a_json(dm)))
    db.ejecutar("INSERT INTO bot_models(asset_id, ts, model_blob, features_json) VALUES (?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET ts=excluded.ts, model_blob=excluded.model_blob, features_json=excluded.features_json",
                (asset_id, ts, pickle.dumps(rf), db.a_json(features)))
    db.ejecutar("UPDATE bot_assets SET training_status = 'listo' WHERE id = ?", (asset_id,))
    avisar("Listo", 1.0)
    emit("sistema", "ok", f"Modelo RF entrenado para {symbol}", {
        "hiperparametros": mejor["hyperparams"], "accuracy": round(float(mejor["accuracy"]), 3), "f1": round(float(mejor["f1"]), 3),
        "rmse": round(float(mejor["rmse"]), 4), "dm_rf_vs_knn": dm["rf_vs_knn"]["mejor"], "velas": len(df),
    })
    return {"run_id": run_id, "hyperparams": mejor["hyperparams"], "metrics": metricas, "dm": dm}
