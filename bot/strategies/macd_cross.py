"""Cruce de MACD con su línea de señal."""
from __future__ import annotations

import pandas as pd

from bot.features import macd
from bot.strategies.base import MIN_VELAS, Signal, senal_insuficiente


class MACDCross:
    """BUY en cruce alcista del MACD sobre la señal; SELL en cruce bajista."""

    name = "macd_cross"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela."""
        if len(df) < MIN_VELAS:
            return senal_insuficiente(len(df))
        m = macd(df["close"])
        h, h_prev = float(m["macd_hist"].iloc[-1]), float(m["macd_hist"].iloc[-2])
        c = float(df["close"].iloc[-1])
        det = {"macd": float(m["macd"].iloc[-1]), "signal": float(m["macd_signal"].iloc[-1]), "hist": h, "hist_prev": h_prev, "close": c}
        fuerza = min(0.5 + abs(h) / c * 200, 0.95)
        if h > 0 and h_prev <= 0:
            return Signal("BUY", fuerza, det | {"razon": "cruce alcista MACD"})
        if h < 0 and h_prev >= 0:
            return Signal("SELL", fuerza, det | {"razon": "cruce bajista MACD"})
        return Signal("HOLD", 0.4, det | {"razon": "sin cruce"})
