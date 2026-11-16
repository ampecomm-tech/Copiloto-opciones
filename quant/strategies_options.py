"""Definición de estrategias de opciones multi-pata y su payoff.

Una estrategia es un diccionario serializable:

```
{
  "nombre": "Bull call spread 95/105 (30d)",
  "tipo": "bull_call_spread",
  "direccion": "alcista",
  "legs": [ {type, strike, expiry, dte, side(+1/-1), qty, price, iv, bid, ask}, ... ],
  "multiplicador": 100,
  "costo_neto": 320.0,        # positivo = débito pagado, negativo = crédito recibido
  "max_ganancia": 680.0, "max_perdida": 320.0, "ganancia_ilimitada": False,
  "breakevens": [98.2], "griegas": {delta, gamma, theta, vega, rho}
}
```

Las patas comparten vencimiento (no se generan calendarios). Los precios de
ejecución son los reales: **ask al comprar, bid al vender**.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from quant.black_scholes import anios, griegas, valor_intrinseco

Leg = dict[str, Any]
Estrategia = dict[str, Any]

DESCRIPCIONES: dict[str, str] = {
    "long_call": "Compra de call: apuesta alcista con pérdida limitada a la prima.",
    "long_put": "Compra de put: apuesta bajista con pérdida limitada a la prima.",
    "bull_call_spread": "Compra call K1 y vende call K2>K1: alcista con costo y ganancia acotados.",
    "bear_put_spread": "Compra put K2 y vende put K1<K2: bajista con costo y ganancia acotados.",
    "bull_put_spread": "Vende put K2 y compra put K1<K2: cobra crédito si el precio se mantiene arriba de K2.",
    "bear_call_spread": "Vende call K1 y compra call K2>K1: cobra crédito si el precio se mantiene abajo de K1.",
    "long_straddle": "Compra call y put al mismo strike: gana con movimientos grandes en cualquier dirección.",
    "long_strangle": "Compra call OTM y put OTM: como el straddle, más barato pero necesita más movimiento.",
    "iron_condor": "Vende strangle y compra alas: cobra crédito si el precio queda en un rango.",
    "long_butterfly": "Compra 1 call K1, vende 2 call K2, compra 1 call K3: apuesta a que el precio termine cerca de K2.",
    "iron_butterfly": "Vende straddle y compra alas: crédito máximo si el precio termina en el strike central.",
}


def crear_leg(contrato: dict[str, Any], side: int, qty: float) -> Leg:
    """Construye una pata a partir de una fila de la cadena de opciones.

    El precio de ejecución es `ask` si se compra (side=+1) y `bid` si se vende (side=-1).
    """
    precio_ejec = float(contrato["ask"]) if side > 0 else float(contrato["bid"])
    return {
        "type": contrato["type"],
        "strike": float(contrato["strike"]),
        "expiry": str(contrato["expiry"]),
        "dte": float(contrato["dte"]),
        "side": int(side),
        "qty": float(qty),
        "price": precio_ejec,
        "iv": float(contrato["iv"]) if contrato.get("iv") == contrato.get("iv") else None,
        "bid": float(contrato["bid"]),
        "ask": float(contrato["ask"]),
        "oi": float(contrato.get("oi") or 0),
        "volume": float(contrato.get("volume") or 0),
        "symbol": str(contrato.get("symbol", "")),
        "tick_size": float(contrato.get("tick_size") or 0) if contrato.get("tick_size") == contrato.get("tick_size") else 0.0,
    }


def payoff_vencimiento(legs: list[Leg], S_T: np.ndarray | float, multiplicador: float = 1.0) -> np.ndarray:
    """P&L total al vencimiento para uno o varios precios finales del subyacente.

    P&L = Σ side·qty·mult·(intrínseco(S_T) − precio_pagado).
    """
    S_T = np.atleast_1d(np.asarray(S_T, dtype=float))
    total = np.zeros_like(S_T)
    for leg in legs:
        intr = np.maximum(S_T - leg["strike"], 0.0) if leg["type"] == "call" else np.maximum(leg["strike"] - S_T, 0.0)
        total += leg["side"] * leg["qty"] * multiplicador * (intr - leg["price"])
    return total


def valor_actual(legs: list[Leg], S: float, r: float, dias_transcurridos: float = 0.0, iv_shift: float = 0.0, multiplicador: float = 1.0) -> float:
    """Valor de mercado teórico (Black-Scholes) de la posición hoy o tras `dias_transcurridos`.

    Args:
        S: Precio del subyacente a evaluar.
        r: Tasa libre de riesgo.
        dias_transcurridos: Días que han pasado desde la apertura (para el what-if).
        iv_shift: Cambio aditivo en la IV de cada pata (0.05 = +5 puntos).
        multiplicador: Tamaño del contrato.

    Returns:
        Valor liquidativo (positivo = la posición vale dinero).
    """
    valor = 0.0
    for leg in legs:
        T = anios(leg["dte"] - dias_transcurridos)
        sigma = max((leg["iv"] or 0.5) + iv_shift, 0.01)
        p = griegas(S, leg["strike"], T, r, sigma, leg["type"]).precio
        valor += leg["side"] * leg["qty"] * multiplicador * p
    return float(valor)


def pnl_actual(estrategia: Estrategia, S: float, r: float, dias_transcurridos: float = 0.0, iv_shift: float = 0.0) -> float:
    """P&L hoy = valor actual − costo neto pagado."""
    return valor_actual(estrategia["legs"], S, r, dias_transcurridos, iv_shift, estrategia["multiplicador"]) - estrategia["costo_neto"]


def griegas_posicion(legs: list[Leg], S: float, r: float, multiplicador: float = 1.0, dias_transcurridos: float = 0.0) -> dict[str, float]:
    """Griegas agregadas de la posición (suma ponderada por side·qty·multiplicador)."""
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        T = anios(leg["dte"] - dias_transcurridos)
        g = griegas(S, leg["strike"], T, r, leg["iv"] or 0.5, leg["type"])
        w = leg["side"] * leg["qty"] * multiplicador
        for k in agg:
            agg[k] += w * getattr(g, k)
    return {k: float(v) for k, v in agg.items()}


def _breakevens(legs: list[Leg], S: float, multiplicador: float) -> list[float]:
    """Encuentra los cruces por cero del payoff al vencimiento en una malla fina."""
    grid = np.linspace(max(S * 0.2, 0.01), S * 2.5, 4001)
    p = payoff_vencimiento(legs, grid, multiplicador)
    cruces: list[float] = []
    signo = np.sign(p)
    for i in range(1, len(grid)):
        if signo[i] != signo[i - 1] and signo[i] != 0:
            x0, x1, y0, y1 = grid[i - 1], grid[i], p[i - 1], p[i]
            cruces.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0))
    return [round(c, 4) for c in cruces]


def resumen_estrategia(nombre: str, tipo: str, direccion: str, legs: list[Leg], S: float, r: float, multiplicador: float) -> Estrategia:
    """Completa una estrategia con costo neto, máximos, breakevens y griegas.

    `max_perdida` se expresa como número positivo. Si la ganancia es ilimitada
    (p. ej. long call) `ganancia_ilimitada=True` y `max_ganancia` es el valor
    en el borde de la malla (2.5·S).
    """
    costo_neto = sum(leg["side"] * leg["qty"] * multiplicador * leg["price"] for leg in legs)
    grid = np.linspace(0.0, S * 2.5, 2001)
    p = payoff_vencimiento(legs, grid, multiplicador)
    pendiente_final = p[-1] - p[-2]
    ganancia_ilimitada = pendiente_final > 1e-9
    # Pérdida ilimitada en la práctica: pendiente negativa al alza (call corto descubierto).
    perdida_ilimitada = pendiente_final < -1e-9
    return {
        "nombre": nombre,
        "tipo": tipo,
        "descripcion": DESCRIPCIONES.get(tipo, ""),
        "direccion": direccion,
        "legs": legs,
        "multiplicador": multiplicador,
        "expiry": legs[0]["expiry"],
        "dte": legs[0]["dte"],
        "costo_neto": float(costo_neto),
        "max_ganancia": float(p.max()),
        "max_perdida": float(-p.min()),
        "ganancia_ilimitada": bool(ganancia_ilimitada),
        "perdida_ilimitada": bool(perdida_ilimitada),
        "breakevens": _breakevens(legs, S, multiplicador),
        "griegas": griegas_posicion(legs, S, r, multiplicador),
        "spot": float(S),
    }


# ---------- Constructores de cada estrategia ----------
# Cada constructor recibe los contratos ya elegidos de la cadena y devuelve la estrategia.

def long_call(c: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra de call."""
    legs = [crear_leg(c, +1, qty)]
    return resumen_estrategia(f"Long call {c['strike']:g} ({c['dte']:.0f}d)", "long_call", "alcista", legs, S, r, mult)


def long_put(p: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra de put."""
    legs = [crear_leg(p, +1, qty)]
    return resumen_estrategia(f"Long put {p['strike']:g} ({p['dte']:.0f}d)", "long_put", "bajista", legs, S, r, mult)


def bull_call_spread(c1: dict, c2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra call K1, vende call K2 > K1."""
    legs = [crear_leg(c1, +1, qty), crear_leg(c2, -1, qty)]
    return resumen_estrategia(f"Bull call spread {c1['strike']:g}/{c2['strike']:g} ({c1['dte']:.0f}d)", "bull_call_spread", "alcista", legs, S, r, mult)


def bear_put_spread(p1: dict, p2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra put K2, vende put K1 < K2."""
    legs = [crear_leg(p2, +1, qty), crear_leg(p1, -1, qty)]
    return resumen_estrategia(f"Bear put spread {p2['strike']:g}/{p1['strike']:g} ({p2['dte']:.0f}d)", "bear_put_spread", "bajista", legs, S, r, mult)


def bull_put_spread(p1: dict, p2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Vende put K2, compra put K1 < K2 (crédito)."""
    legs = [crear_leg(p2, -1, qty), crear_leg(p1, +1, qty)]
    return resumen_estrategia(f"Bull put spread {p2['strike']:g}/{p1['strike']:g} ({p2['dte']:.0f}d)", "bull_put_spread", "alcista", legs, S, r, mult)


def bear_call_spread(c1: dict, c2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Vende call K1, compra call K2 > K1 (crédito)."""
    legs = [crear_leg(c1, -1, qty), crear_leg(c2, +1, qty)]
    return resumen_estrategia(f"Bear call spread {c1['strike']:g}/{c2['strike']:g} ({c1['dte']:.0f}d)", "bear_call_spread", "bajista", legs, S, r, mult)


def long_straddle(c: dict, p: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra call y put al mismo strike."""
    legs = [crear_leg(c, +1, qty), crear_leg(p, +1, qty)]
    return resumen_estrategia(f"Long straddle {c['strike']:g} ({c['dte']:.0f}d)", "long_straddle", "volatil", legs, S, r, mult)


def long_strangle(p: dict, c: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra put OTM y call OTM."""
    legs = [crear_leg(p, +1, qty), crear_leg(c, +1, qty)]
    return resumen_estrategia(f"Long strangle {p['strike']:g}/{c['strike']:g} ({c['dte']:.0f}d)", "long_strangle", "volatil", legs, S, r, mult)


def iron_condor(p1: dict, p2: dict, c1: dict, c2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra put K1, vende put K2, vende call K3, compra call K4 (K1<K2<S<K3<K4)."""
    legs = [crear_leg(p1, +1, qty), crear_leg(p2, -1, qty), crear_leg(c1, -1, qty), crear_leg(c2, +1, qty)]
    return resumen_estrategia(f"Iron condor {p1['strike']:g}/{p2['strike']:g}/{c1['strike']:g}/{c2['strike']:g} ({c1['dte']:.0f}d)", "iron_condor", "neutral", legs, S, r, mult)


def long_butterfly(c1: dict, c2: dict, c3: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra 1 call K1, vende 2 call K2, compra 1 call K3 (equidistantes)."""
    legs = [crear_leg(c1, +1, qty), crear_leg(c2, -1, 2 * qty), crear_leg(c3, +1, qty)]
    return resumen_estrategia(f"Butterfly {c1['strike']:g}/{c2['strike']:g}/{c3['strike']:g} ({c1['dte']:.0f}d)", "long_butterfly", "neutral", legs, S, r, mult)


def iron_butterfly(p1: dict, p0: dict, c0: dict, c2: dict, qty: float, S: float, r: float, mult: float) -> Estrategia:
    """Compra put K1, vende put y call K0, compra call K2 (K1<K0<K2)."""
    legs = [crear_leg(p1, +1, qty), crear_leg(p0, -1, qty), crear_leg(c0, -1, qty), crear_leg(c2, +1, qty)]
    return resumen_estrategia(f"Iron butterfly {p1['strike']:g}/{c0['strike']:g}/{c2['strike']:g} ({c0['dte']:.0f}d)", "iron_butterfly", "neutral", legs, S, r, mult)


def curva_payoff(estrategia: Estrategia, r: float, dias_transcurridos: float = 0.0, iv_shift: float = 0.0, puntos: int = 200) -> dict[str, list[float]]:
    """Curvas para graficar: payoff al vencimiento y P&L teórico hoy (o en un what-if).

    Returns:
        `{"precios": [...], "vencimiento": [...], "actual": [...]}`.
    """
    S = estrategia["spot"]
    precios = np.linspace(S * 0.6, S * 1.4, puntos)
    venc = payoff_vencimiento(estrategia["legs"], precios, estrategia["multiplicador"])
    actual = [pnl_actual(estrategia, float(p), r, dias_transcurridos, iv_shift) for p in precios]
    return {"precios": precios.tolist(), "vencimiento": venc.tolist(), "actual": actual}
