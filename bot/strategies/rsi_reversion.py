"""Reversión a la media con RSI: compra sobreventa, vende sobrecompra."""
from __future__ import annotations

import pandas as pd

from bot.features import rsi
from bot.strategies.base import MIN_VELAS, Signal, senal_insuficiente


class RSIReversion:
    """BUY si RSI14 < 30 y empieza a girar; SELL si RSI14 > 70."""

    name = "rsi_reversion"

    def __init__(self, bajo: float = 30, alto: float = 70) -> None:
        """Umbrales de sobreventa y sobrecompra."""
        self.bajo, self.alto = bajo, alto

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela."""
        if len(df) < MIN_VELAS:
            return senal_insuficiente(len(df))
        r = rsi(df["close"], 14)
        actual, previo = float(r.iloc[-1]), float(r.iloc[-2])
        det = {"rsi14": actual, "rsi14_prev": previo, "close": float(df["close"].iloc[-1])}
        if actual < self.bajo and actual > previo:
            return Signal("BUY", min(0.5 + (self.bajo - actual) / 40, 0.95), det | {"razon": "RSI en sobreventa girando al alza"})
        if actual > self.alto:
            return Signal("SELL", min(0.5 + (actual - self.alto) / 40, 0.95), det | {"razon": "RSI en sobrecompra"})
        return Signal("HOLD", 0.4, det | {"razon": "RSI en zona neutral"})
