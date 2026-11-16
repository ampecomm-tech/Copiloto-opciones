"""Semáforo de viabilidad: reglas deterministas que alimentan al juez de la mesa.

Entrada: estrategia (con Monte Carlo y griegas), contexto de mercado (IV/HV,
sentimiento) y opinión del usuario. Salida: `verde | amarillo | rojo`, razones
y alternativa sugerida si no es verde.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from core.config import cfg

TIPOS_COMPRA_VOL = {"long_call", "long_put", "long_straddle", "long_strangle", "long_butterfly"}
TIPOS_VENTA_VOL = {"bull_put_spread", "bear_call_spread", "iron_condor", "iron_butterfly"}
ALTERNATIVAS: dict[str, str] = {
    "long_call": "bull_call_spread", "long_put": "bear_put_spread",
    "long_straddle": "long_strangle", "long_strangle": "long_butterfly",
    "bull_call_spread": "bull_put_spread", "bear_put_spread": "bear_call_spread",
    "bull_put_spread": "bull_call_spread", "bear_call_spread": "bear_put_spread",
    "iron_condor": "long_butterfly", "iron_butterfly": "iron_condor", "long_butterfly": "iron_condor",
}


@dataclass
class Semaforo:
    """Resultado del semáforo."""

    color: str                       # verde | amarillo | rojo
    puntaje: float                   # 0..1
    razones: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    alternativa: str | None = None
    reglas: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario."""
        return asdict(self)


def _spread_relativo(estrategia: dict[str, Any]) -> float:
    """Spread bid-ask relativo promedio de las patas."""
    return float(np.mean([(l["ask"] - l["bid"]) / max((l["ask"] + l["bid"]) / 2, 1e-9) for l in estrategia["legs"]]))


def _oi_minimo(estrategia: dict[str, Any]) -> float:
    """Menor open interest entre las patas."""
    return float(min(l.get("oi", 0) or 0 for l in estrategia["legs"]))


def evaluar(estrategia: dict[str, Any], iv_hv: float | None, sentimiento: float | None, opinion: dict[str, Any]) -> Semaforo:
    """Aplica las reglas del semáforo a una estrategia.

    Args:
        estrategia: Estrategia con `montecarlo`, `griegas`, `legs`, `max_perdida`, `max_ganancia`.
        iv_hv: Cociente IV/HV del subyacente (None si no se pudo calcular).
        sentimiento: Puntaje de sentimiento de X en [-1, 1] (None si no hay).
        opinion: Diccionario con `direccion`, `horizonte_dias`, `presupuesto`.

    Returns:
        `Semaforo` con color, puntaje, razones, advertencias, reglas y alternativa.
    """
    c = cfg("semaforo") or {}
    mc = estrategia.get("montecarlo", {})
    g = estrategia.get("griegas", {})
    tipo = estrategia["tipo"]
    razones: list[str] = []
    advert: list[str] = []
    reglas: dict[str, bool] = {}

    # 1. Valor esperado > 0
    ev = float(mc.get("valor_esperado", 0.0))
    reglas["valor_esperado_positivo"] = ev > 0
    (razones if ev > 0 else advert).append(f"Valor esperado Monte Carlo: {ev:+,.2f} USD")

    # 2. Probabilidad de ganancia y relación ganancia/riesgo
    pg = float(mc.get("prob_ganancia", 0.0))
    reglas["prob_ganancia_ok"] = pg >= float(c.get("prob_ganancia_min_amarillo", 0.30))
    reglas["prob_ganancia_alta"] = pg >= float(c.get("prob_ganancia_min_verde", 0.45))
    max_g = estrategia["max_ganancia"] if not estrategia.get("ganancia_ilimitada") else estrategia["max_perdida"] * 3
    ratio = max_g / max(estrategia["max_perdida"], 1e-9)
    reglas["ratio_ganancia_riesgo_ok"] = ratio >= float(c.get("ratio_ganancia_riesgo_min", 0.5))
    (razones if reglas["prob_ganancia_alta"] else advert).append(f"Probabilidad de ganancia: {pg:.0%}")
    (razones if reglas["ratio_ganancia_riesgo_ok"] else advert).append(f"Ganancia máx./riesgo: {ratio:.2f}x" + (" (ganancia ilimitada)" if estrategia.get("ganancia_ilimitada") else ""))

    # 3. IV vs HV: penalizar comprar caro / vender barato
    reglas["iv_hv_ok"] = True
    if iv_hv is not None and iv_hv == iv_hv:
        if tipo in TIPOS_COMPRA_VOL and iv_hv > float(c.get("iv_hv_max_para_comprar", 1.4)):
            reglas["iv_hv_ok"] = False
            advert.append(f"IV/HV = {iv_hv:.2f}: las opciones están caras para comprar volatilidad")
        elif tipo in TIPOS_VENTA_VOL and iv_hv < float(c.get("iv_hv_min_para_vender", 0.8)):
            reglas["iv_hv_ok"] = False
            advert.append(f"IV/HV = {iv_hv:.2f}: las opciones están baratas para vender volatilidad")
        else:
            razones.append(f"IV/HV = {iv_hv:.2f}, coherente con la estructura")

    # 4. Liquidez
    sp = _spread_relativo(estrategia)
    oi = _oi_minimo(estrategia)
    reglas["liquidez_ok"] = sp <= float(c.get("spread_relativo_max", 0.15)) and oi >= float(cfg("opciones", "min_open_interest", default=10))
    (razones if reglas["liquidez_ok"] else advert).append(f"Spread bid-ask relativo {sp:.1%}, OI mínimo {oi:.0f}")

    # 5. Theta vs horizonte
    theta = float(g.get("theta", 0.0))
    presupuesto = float(opinion.get("presupuesto", estrategia["max_perdida"]))
    theta_pct = abs(theta) / max(presupuesto, 1e-9) if theta < 0 else 0.0
    reglas["theta_ok"] = theta_pct <= float(c.get("theta_diario_max_pct_presupuesto", 0.03))
    if theta < 0:
        (razones if reglas["theta_ok"] else advert).append(f"Theta diaria {theta:,.2f} USD ({theta_pct:.1%} del presupuesto por día)")
    else:
        razones.append(f"Theta positiva ({theta:,.2f} USD/día): el paso del tiempo juega a favor")

    # 6. Alineación con el sentimiento
    reglas["sentimiento_alineado"] = True
    direccion = opinion.get("direccion", estrategia.get("direccion"))
    if sentimiento is not None:
        if direccion == "alcista" and sentimiento < -0.3 or direccion == "bajista" and sentimiento > 0.3:
            reglas["sentimiento_alineado"] = False
            advert.append(f"El sentimiento de X ({sentimiento:+.2f}) va en contra de la opinión {direccion}")
        else:
            razones.append(f"Sentimiento de X ({sentimiento:+.2f}) no contradice la opinión")

    # 7. Pérdida máxima ≤ presupuesto
    reglas["dentro_presupuesto"] = estrategia["max_perdida"] <= presupuesto * 1.0001 and not estrategia.get("perdida_ilimitada")
    (razones if reglas["dentro_presupuesto"] else advert).append(f"Pérdida máxima {estrategia['max_perdida']:,.2f} USD vs presupuesto {presupuesto:,.2f} USD")

    # Puntaje y color
    pesos = {"valor_esperado_positivo": 0.2, "prob_ganancia_ok": 0.1, "prob_ganancia_alta": 0.1, "ratio_ganancia_riesgo_ok": 0.1,
             "iv_hv_ok": 0.15, "liquidez_ok": 0.1, "theta_ok": 0.1, "sentimiento_alineado": 0.05, "dentro_presupuesto": 0.1}
    puntaje = sum(p for k, p in pesos.items() if reglas.get(k, False))
    criticas_fallidas = not reglas["dentro_presupuesto"] or not reglas["prob_ganancia_ok"] or (not reglas["valor_esperado_positivo"] and not reglas["ratio_ganancia_riesgo_ok"])
    # Verde exige, además del puntaje, que no falle ninguna regla "estructural".
    estructurales_ok = all(reglas[k] for k in ("valor_esperado_positivo", "iv_hv_ok", "liquidez_ok", "dentro_presupuesto"))
    if criticas_fallidas or puntaje < 0.5:
        color = "rojo"
    elif puntaje >= 0.8 and estructurales_ok:
        color = "verde"
    else:
        color = "amarillo"
    alternativa = None if color == "verde" else ALTERNATIVAS.get(tipo)
    return Semaforo(color=color, puntaje=round(puntaje, 3), razones=razones, advertencias=advert, alternativa=alternativa, reglas=reglas)
