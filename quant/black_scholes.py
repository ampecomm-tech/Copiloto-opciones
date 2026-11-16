"""Modelo de Black-Scholes: precio, griegas y volatilidad implícita.

Convenciones:
- `S`: precio del subyacente, `K`: strike, `T`: tiempo a vencimiento en años,
  `r`: tasa libre de riesgo continua, `sigma`: volatilidad anualizada.
- `tipo`: "call" o "put".
- Las griegas se devuelven por unidad de subyacente (no por contrato):
  theta en unidades monetarias por día, vega y rho por punto porcentual (1%).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class Griegas:
    """Griegas de una opción europea bajo Black-Scholes."""

    precio: float
    delta: float
    gamma: float
    theta: float   # por día
    vega: float    # por 1% de vol
    rho: float     # por 1% de tasa

    def to_dict(self) -> dict[str, float]:
        """Convierte a diccionario."""
        return asdict(self)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Calcula d1 y d2 de Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T y sigma deben ser positivos")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def valor_intrinseco(S: float, K: float, tipo: str) -> float:
    """Valor intrínseco al vencimiento (payoff) de una opción."""
    return max(S - K, 0.0) if tipo == "call" else max(K - S, 0.0)


def precio(S: float, K: float, T: float, r: float, sigma: float, tipo: str = "call") -> float:
    """Precio teórico de una opción europea.

    Si `T` ≤ 0 devuelve el valor intrínseco. Si `sigma` ≤ 0 devuelve el valor
    descontado del payoff forward (límite determinista).
    """
    if T <= 0:
        return valor_intrinseco(S, K, tipo)
    if sigma <= 0:
        fwd = S * math.exp(r * T)
        return math.exp(-r * T) * valor_intrinseco(fwd, K, tipo)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if tipo == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def griegas(S: float, K: float, T: float, r: float, sigma: float, tipo: str = "call") -> Griegas:
    """Precio y griegas analíticas (delta, gamma, theta/día, vega/1%, rho/1%).

    En el vencimiento (`T` ≤ 0) o con vol nula devuelve griegas degeneradas:
    delta 0/±1 según el dinero y el resto en cero.
    """
    if T <= 0 or sigma <= 0:
        itm = (S > K) if tipo == "call" else (S < K)
        delta = (1.0 if tipo == "call" else -1.0) if itm else 0.0
        return Griegas(precio=precio(S, K, T, r, sigma, tipo), delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    sqT = math.sqrt(T)
    disc = math.exp(-r * T)
    gamma = pdf / (S * sigma * sqT)
    vega = S * pdf * sqT / 100.0
    if tipo == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf * sigma) / (2 * sqT) - r * K * disc * norm.cdf(d2)) / 365.0
        rho = K * T * disc * norm.cdf(d2) / 100.0
        p = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-(S * pdf * sigma) / (2 * sqT) + r * K * disc * norm.cdf(-d2)) / 365.0
        rho = -K * T * disc * norm.cdf(-d2) / 100.0
        p = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
    return Griegas(precio=p, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def volatilidad_implicita(
    precio_mercado: float, S: float, K: float, T: float, r: float, tipo: str = "call",
    sigma_min: float = 1e-4, sigma_max: float = 5.0,
) -> float | None:
    """Volatilidad implícita por el método de Brent (con Newton como refinamiento).

    Returns:
        Sigma anualizada o `None` si el precio está fuera de los límites de
        no-arbitraje o no hay raíz en el intervalo.
    """
    if T <= 0 or precio_mercado <= 0 or S <= 0 or K <= 0:
        return None
    disc = math.exp(-r * T)
    # Límites de no arbitraje
    if tipo == "call":
        lo, hi = max(S - K * disc, 0.0), S
    else:
        lo, hi = max(K * disc - S, 0.0), K * disc
    if precio_mercado <= lo + 1e-12 or precio_mercado >= hi - 1e-12:
        return None

    def f(sig: float) -> float:
        return precio(S, K, T, r, sig, tipo) - precio_mercado

    try:
        if f(sigma_min) * f(sigma_max) > 0:
            return None
        sigma = brentq(f, sigma_min, sigma_max, xtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return None
    # Un paso de Newton para pulir
    try:
        v = griegas(S, K, T, r, sigma, tipo).vega * 100.0
        if v > 1e-10:
            sigma_n = sigma - f(sigma) / v
            if sigma_min < sigma_n < sigma_max:
                sigma = sigma_n
    except ValueError:
        pass
    return float(sigma)


def anios(dias: float) -> float:
    """Convierte días calendario a años (base 365)."""
    return max(dias, 0.0) / 365.0
