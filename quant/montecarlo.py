"""Simulación Monte Carlo (GBM) de una estrategia de opciones.

Simula ≥10,000 trayectorias del subyacente con la volatilidad implícita y
evalúa el P&L de la estrategia: al vencimiento (payoff exacto) y, si el
horizonte del usuario es menor que el vencimiento, en el horizonte
(re-valuando con Black-Scholes).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.config import cfg
from quant.black_scholes import anios, precio as bs_precio
from quant.strategies_options import Estrategia, payoff_vencimiento


def _iv_estrategia(estrategia: Estrategia, fallback: float) -> float:
    """IV representativa: promedio de las IV de las patas (o `fallback`)."""
    ivs = [leg["iv"] for leg in estrategia["legs"] if leg.get("iv")]
    return float(np.mean(ivs)) if ivs else fallback


def simular_precios(S: float, sigma: float, T: float, n: int, mu: float | np.ndarray = 0.0, semilla: int | None = None) -> np.ndarray:
    """Precios finales S_T bajo GBM: S·exp((μ − σ²/2)T + σ√T·Z).

    `mu` puede ser un escalar o un vector por trayectoria (para mezclas de escenarios).
    """
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal(n)
    return S * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def deriva_desde_opinion(opinion: dict[str, Any] | None, T: float, r: float, n: int, semilla: int | None = None) -> float | np.ndarray:
    """Convierte la opinión del usuario en una deriva anual μ para el GBM.

    - alcista/bajista: μ tal que E[S_T] = S·(1 ± magnitud) en el horizonte.
    - neutral: μ = r (precio esperado sin cambio real).
    - volátil: mezcla 50/50 de los escenarios alcista y bajista (vector de μ).
    Sin opinión devuelve `r` (medida neutral al riesgo).
    """
    if not opinion or T <= 0:
        return r
    mag = float(opinion.get("magnitud_pct", 0.0)) / 100.0
    direccion = str(opinion.get("direccion", "neutral"))
    mu_up = np.log1p(mag) / T if mag > 0 else r
    mu_dn = np.log1p(-min(mag, 0.95)) / T if mag > 0 else r
    if direccion == "alcista":
        return float(mu_up)
    if direccion == "bajista":
        return float(mu_dn)
    if direccion == "volatil":
        rng = np.random.default_rng(None if semilla is None else semilla + 1)
        return np.where(rng.random(n) < 0.5, mu_up, mu_dn)
    return r


def valor_en_horizonte(estrategia: Estrategia, precios: np.ndarray, dias_restantes: float, r: float) -> np.ndarray:
    """Valor de la posición (vectorizado) cuando aún quedan `dias_restantes` al vencimiento."""
    T = anios(dias_restantes)
    valor = np.zeros_like(precios)
    mult = estrategia["multiplicador"]
    for leg in estrategia["legs"]:
        sigma = leg["iv"] or 0.5
        if T <= 0:
            intr = np.maximum(precios - leg["strike"], 0) if leg["type"] == "call" else np.maximum(leg["strike"] - precios, 0)
            valor += leg["side"] * leg["qty"] * mult * intr
        else:
            # Vectorizamos Black-Scholes a mano para no llamar precio() por trayectoria.
            from scipy.stats import norm

            K = leg["strike"]
            d1 = (np.log(precios / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            disc = np.exp(-r * T)
            if leg["type"] == "call":
                p = precios * norm.cdf(d1) - K * disc * norm.cdf(d2)
            else:
                p = K * disc * norm.cdf(-d2) - precios * norm.cdf(-d1)
            valor += leg["side"] * leg["qty"] * mult * p
    return valor


def simular(estrategia: Estrategia, S: float, r: float, horizonte_dias: float | None = None, iv: float | None = None, n: int | None = None, opinion: dict[str, Any] | None = None) -> dict[str, Any]:
    """Corre la simulación y devuelve probabilidad de ganancia, valor esperado y percentiles.

    Se corren dos escenarios con la misma volatilidad implícita:
    - **con opinión**: la deriva refleja la magnitud y dirección esperadas por el usuario
      (es la que usan el ranking y el semáforo);
    - **neutral**: deriva = tasa libre de riesgo (referencia de mercado; con ella el valor
      esperado de cualquier estrategia es ≈ −costos de transacción).

    Args:
        estrategia: Estrategia completa (ver `strategies_options`).
        S: Spot actual.
        r: Tasa libre de riesgo.
        horizonte_dias: Horizonte del usuario; si es menor al DTE se evalúa ahí.
        iv: Volatilidad para simular (por defecto, promedio de las patas).
        n: Trayectorias (por defecto `config.opciones.montecarlo_trayectorias`).
        opinion: Diccionario con `direccion` y `magnitud_pct` (opcional).

    Returns:
        Diccionario con `prob_ganancia`, `valor_esperado`, percentiles y el escenario neutral.
    """
    n = int(n or cfg("opciones", "montecarlo_trayectorias", default=10000))
    semilla = cfg("opciones", "montecarlo_semilla", default=None)
    sigma = iv or _iv_estrategia(estrategia, 0.5)
    dte = float(estrategia["dte"])
    dias_eval = min(float(horizonte_dias), dte) if horizonte_dias else dte
    T = anios(dias_eval)

    def pnl_para(mu: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        precios = simular_precios(S, sigma, T, n, mu, semilla)
        if dias_eval >= dte - 1e-9:
            return precios, payoff_vencimiento(estrategia["legs"], precios, estrategia["multiplicador"])
        return precios, valor_en_horizonte(estrategia, precios, dte - dias_eval, r) - estrategia["costo_neto"]

    mu = deriva_desde_opinion(opinion, T, r, n, semilla)
    sigma_vista = sigma
    if opinion and str(opinion.get("direccion")) == "neutral" and float(opinion.get("magnitud_pct", 0)) > 0 and T > 0:
        # Opinión neutral: la magnitud se lee como rango esperado (±mag) al horizonte,
        # es decir, el usuario espera menos movimiento que el implícito en la IV.
        sigma_vista = min(sigma, float(opinion["magnitud_pct"]) / 100.0 / np.sqrt(T))
    sigma_mercado, sigma = sigma, sigma_vista
    precios, pnl = pnl_para(mu)
    sigma = sigma_mercado
    _, pnl_neutral = pnl_para(r)

    pct = np.percentile(pnl, [5, 25, 50, 75, 95])
    ev = float(pnl.mean())
    max_perdida = max(estrategia["max_perdida"], 1e-9)
    from quant.riesgo import var_cvar_historico

    r95, r99 = var_cvar_historico(pnl, 0.95), var_cvar_historico(pnl, 0.99)
    return {
        "n": n,
        "sigma": float(sigma),
        "dias_evaluados": float(dias_eval),
        "deriva_anual": float(np.mean(mu)),
        "prob_ganancia": float((pnl > 0).mean()),
        "valor_esperado": ev,
        "ev_sobre_riesgo": ev / max_perdida,
        "neutral": {"prob_ganancia": float((pnl_neutral > 0).mean()), "valor_esperado": float(pnl_neutral.mean())},
        "pnl_p5": float(pct[0]),
        "pnl_p25": float(pct[1]),
        "pnl_p50": float(pct[2]),
        "pnl_p75": float(pct[3]),
        "pnl_p95": float(pct[4]),
        "prob_perdida_total": float((pnl <= -0.95 * estrategia["max_perdida"]).mean()),
        "precio_final_medio": float(precios.mean()),
        "var_95": r95["var"], "cvar_95": r95["cvar"], "var_99": r99["var"], "cvar_99": r99["cvar"],
        "histograma": _histograma(pnl),
        "histograma_precios": _histograma(precios, 50),
        "cuantiles_pnl": np.percentile(pnl, np.arange(0, 101)).round(4).tolist(),
    }


def _histograma(pnl: np.ndarray, bins: int = 40) -> dict[str, list[float]]:
    """Histograma compacto para graficar.

    El rango va del percentil 0.5 al 99.5 y los valores de fuera se suman a la barra del extremo, para que una cola
    larga (por ejemplo, la ganancia ilimitada de un long call) no aplaste la gráfica. Ninguna trayectoria se pierde.
    """
    bajo, alto = np.percentile(pnl, [0.5, 99.5])
    if alto <= bajo:
        bajo, alto = float(np.min(pnl)), float(np.max(pnl)) + 1e-9
    h, edges = np.histogram(np.clip(pnl, bajo, alto), bins=bins, range=(bajo, alto))
    centros = (edges[:-1] + edges[1:]) / 2
    return {"x": centros.tolist(), "y": h.tolist()}
