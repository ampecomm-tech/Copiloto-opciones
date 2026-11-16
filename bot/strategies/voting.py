"""Estrategia de votación: combina las señales de las demás estrategias."""
from __future__ import annotations

import pandas as pd

from bot.strategies.base import Signal, Strategy


class Voting:
    """BUY/SELL si la mayoría ponderada por confianza coincide; HOLD si no hay consenso."""

    name = "voting"

    def __init__(self, estrategias: list[Strategy], umbral: float = 0.5) -> None:
        """`estrategias`: lista de estrategias base; `umbral`: fracción mínima de votos ponderados."""
        self.estrategias = estrategias
        self.umbral = umbral

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Suma votos ponderados por confianza."""
        votos: dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        detalle: dict[str, str] = {}
        for e in self.estrategias:
            s = e.generate_signal(df)
            votos[s.action] += s.confidence
            detalle[e.name] = f"{s.action} ({s.confidence:.2f})"
        total = sum(votos.values()) or 1.0
        mejor = max(votos, key=votos.get)
        frac = votos[mejor] / total
        det = {"votos": {k: round(v, 3) for k, v in votos.items()}, "por_estrategia": detalle, "close": float(df["close"].iloc[-1]) if len(df) else None}
        if mejor in ("BUY", "SELL") and frac >= self.umbral:
            return Signal(mejor, frac, det | {"razon": f"mayoría {mejor} con {frac:.0%} del peso"})
        return Signal("HOLD", frac, det | {"razon": "sin consenso"})
