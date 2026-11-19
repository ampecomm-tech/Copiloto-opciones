"""Medidas de riesgo de cola: Value at Risk (VaR) y Conditional Value at Risk (CVaR o Expected Shortfall).

Convención: ambas se reportan como **pérdidas positivas**. Un VaR al 95 % de 120 significa que en el 5 % de los
peores escenarios se pierde al menos 120; el CVaR es la pérdida promedio dentro de ese 5 % y siempre es ≥ VaR.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def var_cvar_historico(resultados: np.ndarray | list[float], nivel: float = 0.95) -> dict[str, float]:
    """VaR y CVaR históricos (o de simulación) a partir de una muestra de P&L o de retornos.

    Args:
        resultados: P&L o retornos; las ganancias son positivas y las pérdidas negativas.
        nivel: Nivel de confianza, por ejemplo 0.95 o 0.99.

    Returns:
        `{"var", "cvar", "n"}`; NaN si hay menos de dos observaciones.
    """
    x = np.asarray(resultados, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return {"var": float("nan"), "cvar": float("nan"), "n": int(len(x))}
    corte = float(np.quantile(x, 1 - nivel))
    cola = x[x <= corte]
    return {"var": -corte, "cvar": -float(cola.mean()) if len(cola) else -corte, "n": int(len(x))}


def var_cvar_normal(media: float, desviacion: float, nivel: float = 0.95) -> dict[str, float]:
    """VaR y CVaR paramétricos bajo normalidad.

    VaR = −(μ + σ·z), con z el cuantil 1 − nivel de la normal estándar.
    CVaR = −(μ − σ·φ(z) / (1 − nivel)).
    """
    z = stats.norm.ppf(1 - nivel)
    return {"var": -(media + desviacion * z), "cvar": -(media - desviacion * stats.norm.pdf(z) / (1 - nivel))}


def resumen_riesgo(resultados: np.ndarray | list[float], niveles: tuple[float, ...] = (0.95, 0.99)) -> dict[str, Any]:
    """VaR y CVaR históricos y normales a varios niveles, más media, desviación y asimetría.

    Returns:
        `{"n", "media", "desviacion", "asimetria", "curtosis", "var_95", "cvar_95", "var_95_normal", "cvar_95_normal", ...}`.
    """
    x = np.asarray(resultados, dtype=float)
    x = x[~np.isnan(x)]
    out: dict[str, Any] = {"n": int(len(x))}
    if len(x) < 2:
        return out
    media, desv = float(x.mean()), float(x.std(ddof=1))
    out.update({"media": media, "desviacion": desv,
                "asimetria": float(stats.skew(x)) if len(x) > 2 else float("nan"),
                "curtosis": float(stats.kurtosis(x)) if len(x) > 3 else float("nan")})
    for nivel in niveles:
        k = int(round(nivel * 100))
        h = var_cvar_historico(x, nivel)
        p = var_cvar_normal(media, desv, nivel)
        out.update({f"var_{k}": h["var"], f"cvar_{k}": h["cvar"], f"var_{k}_normal": p["var"], f"cvar_{k}_normal": p["cvar"]})
    return out
