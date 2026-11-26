"""Estrategias de swing trading en velas diarias.

Operan sobre velas de `config.bot.swing.timeframe` (por defecto 1d) y mantienen la posición días o semanas,
en lugar de minutos u horas. El runner sigue revisando el stop en cada vela de 15 min, pero con el ATR
diario: el stop queda más lejos, el tamaño de la posición es menor (mismo riesgo en dólares) y el ruido
intradía ya no saca de la operación.

Las señales con `details["solo_cierre"] = True` solo cierran una posición abierta; nunca abren una nueva en
sentido contrario. Así una estrategia puede decir "sal del largo" sin que el runner lo tome como "abre un corto".
"""
from __future__ import annotations

import pandas as pd

from bot.features import ema, rsi
from bot.strategies.base import Signal, senal_insuficiente
from core.config import cfg


def _p(clave: str, defecto: float) -> float:
    """Parámetro de `config.bot.swing`."""
    return float(cfg("bot", "swing", clave, default=defecto))


def timeframe_swing() -> str:
    """Temporalidad de las estrategias swing (`config.bot.swing.timeframe`)."""
    return str(cfg("bot", "swing", "timeframe", default="1d"))


class DonchianSwing:
    """Ruptura del canal de Donchian (estilo tortugas).

    - BUY: el cierre supera el máximo de las `donchian_entrada` velas anteriores (ruptura alcista).
    - SELL: el cierre perfora el mínimo de esas velas (ruptura bajista; en paper/Alpaca abre corto).
    - Salida más rápida (solo cierre): mínimo/máximo de las `donchian_salida` velas anteriores.
    """

    name = "donchian_swing"

    def __init__(self) -> None:
        """Lee los periodos del canal de `config.yaml`."""
        self.timeframe = timeframe_swing()
        self.entrada, self.salida = int(_p("donchian_entrada", 20)), int(_p("donchian_salida", 10))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela diaria cerrada."""
        if len(df) < self.entrada + 2:
            return senal_insuficiente(len(df))
        previas = df.iloc[:-1]
        cierre = float(df["close"].iloc[-1])
        alto_e, bajo_e = float(previas["high"].tail(self.entrada).max()), float(previas["low"].tail(self.entrada).min())
        alto_s, bajo_s = float(previas["high"].tail(self.salida).max()), float(previas["low"].tail(self.salida).min())
        det = {"close": cierre, f"max_{self.entrada}": alto_e, f"min_{self.entrada}": bajo_e, f"max_{self.salida}": alto_s, f"min_{self.salida}": bajo_s,
               "timeframe": self.timeframe}
        ancho = max(alto_e - bajo_e, 1e-12)
        if cierre > alto_e:
            return Signal("BUY", min(0.6 + (cierre - alto_e) / ancho, 0.95), det | {"razon": f"ruptura del máximo de {self.entrada} días"})
        if cierre < bajo_e:
            return Signal("SELL", min(0.6 + (bajo_e - cierre) / ancho, 0.95), det | {"razon": f"ruptura del mínimo de {self.entrada} días"})
        if cierre < bajo_s:
            return Signal("SELL", 0.55, det | {"razon": f"perdió el mínimo de {self.salida} días: salir del largo", "solo_cierre": True})
        if cierre > alto_s:
            return Signal("BUY", 0.55, det | {"razon": f"superó el máximo de {self.salida} días: salir del corto", "solo_cierre": True})
        return Signal("HOLD", 0.4, det | {"razon": "precio dentro del canal"})


class EMAPullbackSwing:
    """Compra retrocesos dentro de una tendencia alcista diaria.

    - Tendencia: EMA rápida (20) sobre la lenta (50) y cierre sobre la lenta.
    - BUY: la vela tocó la EMA rápida (o quedó a `pullback_tolerancia` de ella) y cerró por encima: rebote.
    - SELL: la tendencia se rompe (EMA rápida bajo la lenta y cierre bajo la lenta); simétrico para cortos.
    """

    name = "ema_pullback_swing"

    def __init__(self) -> None:
        """Lee los periodos de las EMA y la tolerancia del retroceso."""
        self.timeframe = timeframe_swing()
        self.rapida, self.lenta = int(_p("ema_rapida", 20)), int(_p("ema_lenta", 50))
        self.tol = _p("pullback_tolerancia", 0.01)

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela diaria cerrada."""
        if len(df) < self.lenta + 10:
            return senal_insuficiente(len(df))
        c = df["close"]
        er, el = ema(c, self.rapida), ema(c, self.lenta)
        cierre, alto, bajo = float(c.iloc[-1]), float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        r, l = float(er.iloc[-1]), float(el.iloc[-1])
        det = {"close": cierre, f"ema{self.rapida}": r, f"ema{self.lenta}": l, "timeframe": self.timeframe}
        fuerza = min(abs(r - l) / l * 20, 0.3)
        if r > l and cierre > l:
            if bajo <= r * (1 + self.tol) and cierre > r:
                return Signal("BUY", 0.6 + fuerza, det | {"razon": f"retroceso a la EMA{self.rapida} en tendencia alcista y rebote"})
            return Signal("HOLD", 0.5, det | {"razon": "tendencia alcista, esperando retroceso"})
        if r < l and cierre < l:
            if alto >= r * (1 - self.tol) and cierre < r:
                return Signal("SELL", 0.6 + fuerza, det | {"razon": f"rebote fallido en la EMA{self.rapida} en tendencia bajista"})
            return Signal("SELL", 0.5, det | {"razon": "tendencia bajista: salir del largo", "solo_cierre": True})
        return Signal("HOLD", 0.4, det | {"razon": "sin tendencia clara"})


class RSI2Swing:
    """Reversión de corto plazo a favor de la tendencia larga (RSI de 2 periodos, estilo Connors).

    - BUY: cierre sobre la SMA200 y RSI2 < `rsi2_bajo` (caída fuerte dentro de una tendencia alcista).
    - SELL: cierre bajo la SMA200 y RSI2 > `rsi2_alto` (rebote fuerte dentro de una tendencia bajista).
    - Salida (solo cierre): el cierre cruza la SMA5 a favor de la operación.
    """

    name = "rsi2_swing"

    def __init__(self) -> None:
        """Lee umbrales y medias de `config.yaml`."""
        self.timeframe = timeframe_swing()
        self.bajo, self.alto = _p("rsi2_bajo", 10), _p("rsi2_alto", 90)
        self.sma_t, self.sma_s = int(_p("sma_tendencia", 200)), int(_p("sma_salida", 5))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Genera la señal sobre la última vela diaria cerrada."""
        if len(df) < self.sma_t + 2:
            return senal_insuficiente(len(df))
        c = df["close"]
        r2 = float(rsi(c, 2).iloc[-1])
        sma_t, sma_s = float(c.rolling(self.sma_t).mean().iloc[-1]), float(c.rolling(self.sma_s).mean().iloc[-1])
        cierre = float(c.iloc[-1])
        det = {"close": cierre, "rsi2": r2, f"sma{self.sma_t}": sma_t, f"sma{self.sma_s}": sma_s, "timeframe": self.timeframe}
        if cierre > sma_t and r2 < self.bajo:
            return Signal("BUY", min(0.6 + (self.bajo - r2) / 25, 0.95), det | {"razon": f"RSI2 en sobreventa sobre la SMA{self.sma_t}"})
        if cierre < sma_t and r2 > self.alto:
            return Signal("SELL", min(0.6 + (r2 - self.alto) / 25, 0.95), det | {"razon": f"RSI2 en sobrecompra bajo la SMA{self.sma_t}"})
        if cierre > sma_s:
            return Signal("SELL", 0.5, det | {"razon": f"cierre sobre la SMA{self.sma_s}: tomar el rebote", "solo_cierre": True})
        if cierre < sma_s:
            return Signal("BUY", 0.5, det | {"razon": f"cierre bajo la SMA{self.sma_s}: cubrir el corto", "solo_cierre": True})
        return Signal("HOLD", 0.4, det | {"razon": "sin extremo de RSI2"})
