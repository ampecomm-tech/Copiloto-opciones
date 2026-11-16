"""Interfaz común de las estrategias del bot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

import pandas as pd


@dataclass
class Signal:
    """Señal de una estrategia sobre la última vela."""

    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    details: dict = field(default_factory=dict)   # indicadores y razonamiento, se publica en #bot-señales

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return asdict(self)


class Strategy(Protocol):
    """Toda estrategia expone `name` y `generate_signal(df)`."""

    name: str

    def generate_signal(self, df: pd.DataFrame) -> Signal: ...


MIN_VELAS = 60
"""Velas mínimas para que una estrategia pueda opinar."""


def senal_insuficiente(n: int) -> Signal:
    """Señal HOLD estándar cuando faltan datos."""
    return Signal("HOLD", 0.0, {"razon": f"datos insuficientes ({n} velas)"})
