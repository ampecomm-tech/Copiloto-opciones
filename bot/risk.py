"""Gestión de riesgo común a todas las estrategias: SL/TP por ATR, tamaño por volatilidad y kill-switch diario."""
from __future__ import annotations

from dataclasses import dataclass

from core.config import cfg


@dataclass
class ParametrosRiesgo:
    """Parámetros de riesgo leídos de `config.yaml`."""

    atr_periodo: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    riesgo_por_operacion_pct: float = 0.02
    perdida_diaria_max_pct: float = 0.05

    @classmethod
    def desde_config(cls) -> "ParametrosRiesgo":
        """Construye desde `config.bot.riesgo`."""
        c = cfg("bot", "riesgo") or {}
        return cls(**{k: c.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__})


def niveles_sl_tp(precio: float, atr_valor: float, lado: str, p: ParametrosRiesgo | None = None) -> tuple[float, float]:
    """Stop loss y take profit a partir del ATR.

    Args:
        precio: Precio de entrada.
        atr_valor: ATR actual.
        lado: "long" o "short".
    """
    p = p or ParametrosRiesgo.desde_config()
    if lado == "long":
        return precio - p.sl_atr_mult * atr_valor, precio + p.tp_atr_mult * atr_valor
    return precio + p.sl_atr_mult * atr_valor, precio - p.tp_atr_mult * atr_valor


def tamano_posicion(capital: float, precio: float, atr_valor: float, p: ParametrosRiesgo | None = None, paso: float = 1e-6) -> float:
    """Cantidad tal que la pérdida al tocar el SL ≈ `riesgo_por_operacion_pct` del capital.

    Nunca excede el capital disponible (sin apalancamiento).
    """
    p = p or ParametrosRiesgo.desde_config()
    if precio <= 0 or atr_valor <= 0 or capital <= 0:
        return 0.0
    riesgo_usd = capital * p.riesgo_por_operacion_pct
    distancia_sl = p.sl_atr_mult * atr_valor
    qty = riesgo_usd / distancia_sl
    qty = min(qty, capital / precio)
    return max(round(qty / paso) * paso, 0.0)


def kill_switch_activo(capital_inicial_dia: float, pnl_dia: float, p: ParametrosRiesgo | None = None) -> bool:
    """True si la pérdida acumulada del día supera el máximo permitido."""
    p = p or ParametrosRiesgo.desde_config()
    return capital_inicial_dia > 0 and pnl_dia <= -p.perdida_diaria_max_pct * capital_inicial_dia


def evaluar_salida(posicion: dict, precio_alto: float, precio_bajo: float, precio_cierre: float) -> str | None:
    """Comprueba si una vela tocó el SL o el TP de la posición.

    Returns:
        "SL", "TP" o None.
    """
    if posicion["side"] == "long":
        if precio_bajo <= posicion["sl"]:
            return "SL"
        if precio_alto >= posicion["tp"]:
            return "TP"
    else:
        if precio_alto >= posicion["sl"]:
            return "SL"
        if precio_bajo <= posicion["tp"]:
            return "TP"
    return None
