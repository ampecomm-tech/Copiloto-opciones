"""Registro de estrategias del bot."""
from __future__ import annotations

from bot.strategies.base import Signal, Strategy
from bot.strategies.ema50_chandelier import EMA50Chandelier
from bot.strategies.macd_cross import MACDCross
from bot.strategies.random_forest import RandomForestStrategy
from bot.strategies.rsi_reversion import RSIReversion
from bot.strategies.voting import Voting

NOMBRES: tuple[str, ...] = ("random_forest", "ema50_chandelier", "rsi_reversion", "macd_cross", "voting")


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
    if nombre == "voting":
        return Voting([RandomForestStrategy(asset_id, sentimiento), EMA50Chandelier(), RSIReversion(), MACDCross()])
    raise ValueError(f"Estrategia desconocida: {nombre}")


__all__ = ["Signal", "Strategy", "NOMBRES", "crear"]
