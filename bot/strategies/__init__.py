"""Registro de estrategias del bot."""
from __future__ import annotations

from bot.strategies.base import Signal, Strategy
from bot.strategies.ema50_chandelier import EMA50Chandelier
from bot.strategies.macd_cross import MACDCross
from bot.strategies.random_forest import RandomForestStrategy
from bot.strategies.rsi_reversion import RSIReversion
from bot.strategies.swing import DonchianSwing, EMAPullbackSwing, RSI2Swing
from bot.strategies.voting import Voting

NOMBRES: tuple[str, ...] = ("random_forest", "ema50_chandelier", "rsi_reversion", "macd_cross", "voting",
                             "donchian_swing", "ema_pullback_swing", "rsi2_swing")
SWING: tuple[str, ...] = ("donchian_swing", "ema_pullback_swing", "rsi2_swing")
"""Estrategias que operan en velas diarias (`config.bot.swing.timeframe`)."""


def timeframe_de(nombre: str) -> str:
    """Temporalidad de las velas con que opera una estrategia: la de swing o la base del bot."""
    from core.config import cfg

    if nombre in SWING:
        return str(cfg("bot", "swing", "timeframe", default="1d"))
    return str(cfg("bot", "timeframe", default="15m"))


def crear(nombre: str, asset_id: int, sentimiento: float | None = None) -> Strategy:
    """Instancia una estrategia por nombre para un activo."""
    if nombre == "random_forest":
        return RandomForestStrategy(asset_id, sentimiento)
    if nombre == "ema50_chandelier":
        return EMA50Chandelier()
    if nombre == "rsi_reversion":
        return RSIReversion()
    if nombre == "macd_cross":
        return MACDCross()
    if nombre == "donchian_swing":
        return DonchianSwing()
    if nombre == "ema_pullback_swing":
        return EMAPullbackSwing()
    if nombre == "rsi2_swing":
        return RSI2Swing()
    if nombre == "voting":
        return Voting([RandomForestStrategy(asset_id, sentimiento), EMA50Chandelier(), RSIReversion(), MACDCross()])
    raise ValueError(f"Estrategia desconocida: {nombre}")


__all__ = ["Signal", "Strategy", "NOMBRES", "SWING", "crear", "timeframe_de"]
