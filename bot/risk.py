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
    trailing_habilitado: bool = True
    trailing_activacion_atr: float = 1.0
    trailing_distancia_atr: float = 2.0

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
    """Comprueba si una vela tocó el stop o el TP de la posición.

    Con el trailing armado el TP fijo deja de aplicar, para no cortar la tendencia.

    Returns:
        "SL", "TRAILING", "TP" o None.
    """
    trailing = bool(posicion.get("trailing_activo"))
    razon_stop = "TRAILING" if trailing else "SL"
    if posicion["side"] == "long":
        if precio_bajo <= posicion["sl"]:
            return razon_stop
        if not trailing and precio_alto >= posicion["tp"]:
            return "TP"
    else:
        if precio_alto >= posicion["sl"]:
            return razon_stop
        if not trailing and precio_bajo <= posicion["tp"]:
            return "TP"
    return None


def actualizar_trailing(posicion: dict, precio_alto: float, precio_bajo: float, atr_valor: float, p: ParametrosRiesgo | None = None) -> dict:
    """Mueve el stop a favor de la posición con una vela ya cerrada; nunca lo aleja.

    El trailing se arma cuando el precio avanza `trailing_activacion_atr` ATR desde la entrada.
    Desde entonces el stop sigue al mejor precio a `trailing_distancia_atr` ATR de distancia.

    Args:
        posicion: Posición abierta (admite posiciones creadas antes de existir el trailing).
        precio_alto: Máximo de la vela cerrada.
        precio_bajo: Mínimo de la vela cerrada.
        atr_valor: ATR actual.

    Returns:
        Copia de la posición con `sl`, `extremo` y `trailing_activo` actualizados.
    """
    p = p or ParametrosRiesgo.desde_config()
    nueva = dict(posicion)
    if not p.trailing_habilitado or atr_valor <= 0:
        return nueva
    entrada = float(posicion["entry_price"])
    extremo = float(posicion.get("extremo", entrada))
    largo = posicion["side"] == "long"
    extremo = max(extremo, precio_alto) if largo else min(extremo, precio_bajo)
    avance = extremo - entrada if largo else entrada - extremo
    armado = bool(posicion.get("trailing_activo")) or avance >= p.trailing_activacion_atr * atr_valor
    nueva.update({"extremo": extremo, "trailing_activo": armado})
    if armado:
        distancia = p.trailing_distancia_atr * atr_valor
        nueva["sl"] = max(posicion["sl"], extremo - distancia) if largo else min(posicion["sl"], extremo + distancia)
    return nueva
