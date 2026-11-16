"""Estrategia con RandomForest entrenado por activo (modelo guardado en la DB)."""
from __future__ import annotations

import pickle
from typing import Any

import pandas as pd

from bot.features import construir_features
from bot.strategies.base import MIN_VELAS, Signal, senal_insuficiente
from core import db
from core.config import cfg


class RandomForestStrategy:
    """BUY si P(sube) > umbral_compra; SELL si P(sube) < umbral_venta; HOLD en medio."""

    name = "random_forest"

    def __init__(self, asset_id: int, sentimiento: float | None = None) -> None:
        """Carga el modelo del activo desde `bot_models` (si existe)."""
        self.asset_id = asset_id
        self.sentimiento = sentimiento
        self.modelo: Any = None
        self.features: list[str] = []
        fila = db.consultar_uno("SELECT model_blob, features_json FROM bot_models WHERE asset_id = ?", (asset_id,))
        if fila:
            self.modelo = pickle.loads(fila["model_blob"])
            self.features = db.de_json(fila["features_json"], [])

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela con la probabilidad del RF."""
        if len(df) < MIN_VELAS:
            return senal_insuficiente(len(df))
        if self.modelo is None:
            return Signal("HOLD", 0.0, {"razon": "modelo no entrenado todavía"})
        X = construir_features(df, self.sentimiento)[self.features].iloc[[-1]].fillna(0)
        prob = float(self.modelo.predict_proba(X)[0][1])
        up, dn = float(cfg("bot", "random_forest", "umbral_compra", default=0.58)), float(cfg("bot", "random_forest", "umbral_venta", default=0.42))
        det = {"prob_subida": round(prob, 4), "umbral_compra": up, "umbral_venta": dn, "close": float(df["close"].iloc[-1]),
               "rsi14": round(float(X["rsi14"].iloc[0]), 2), "macd_hist_rel": round(float(X["macd_hist_rel"].iloc[0]), 5), "sentimiento": self.sentimiento}
        if prob >= up:
            return Signal("BUY", prob, det | {"razon": f"RF estima {prob:.0%} de subida"})
        if prob <= dn:
            return Signal("SELL", 1 - prob, det | {"razon": f"RF estima {1 - prob:.0%} de bajada"})
        return Signal("HOLD", 1 - abs(prob - 0.5) * 2, det | {"razon": "RF sin convicción"})
