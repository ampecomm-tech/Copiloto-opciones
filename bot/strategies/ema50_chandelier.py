"""Estrategia tendencial: precio sobre EMA50 con Chandelier Exit como filtro de salida."""
from __future__ import annotations

import pandas as pd

from bot.features import chandelier_exit, ema
from bot.strategies.base import MIN_VELAS, Signal, senal_insuficiente


class EMA50Chandelier:
    """Compra cuando el cierre cruza por encima de la EMA50 y está sobre el Chandelier largo; vende al perderlos."""

    name = "ema50_chandelier"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela."""
        if len(df) < MIN_VELAS:
            return senal_insuficiente(len(df))
        c = df["close"]
        e50 = ema(c, 50)
        ch = chandelier_exit(df)
        cierre, e, ch_l, ch_s = float(c.iloc[-1]), float(e50.iloc[-1]), float(ch["chandelier_long"].iloc[-1]), float(ch["chandelier_short"].iloc[-1])
        cierre_prev, e_prev = float(c.iloc[-2]), float(e50.iloc[-2])
        det = {"close": cierre, "ema50": e, "chandelier_long": ch_l, "chandelier_short": ch_s}
        dist = (cierre - e) / e
        if cierre > e and cierre > ch_l and (cierre_prev <= e_prev or cierre_prev <= float(ch["chandelier_long"].iloc[-2])):
            return Signal("BUY", min(0.5 + abs(dist) * 10, 0.95), det | {"razon": "cruce alcista de EMA50 con soporte Chandelier"})
        if cierre < e or cierre < ch_l:
            return Signal("SELL", min(0.5 + abs(dist) * 10, 0.95), det | {"razon": "cierre bajo EMA50 o bajo Chandelier Exit"})
        return Signal("HOLD", 0.5, det | {"razon": "tendencia intacta, sin nueva entrada"})
