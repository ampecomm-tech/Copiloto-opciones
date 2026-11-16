"""Prueba de Diebold-Mariano (1995) con la corrección de Harvey-Leybourne-Newbold (1997).

Compara la precisión predictiva de dos modelos a partir de sus errores de pronóstico.
H0: ambas predicciones tienen la misma precisión (E[d_t] = 0).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class ResultadoDM:
    """Resultado de la prueba DM."""

    dm: float            # estadístico original
    dm_hln: float        # estadístico corregido (HLN)
    p_valor: float       # bilateral, con t de Student (n−1 g.l.)
    n: int
    horizonte: int
    mejor: str           # "modelo_1" | "modelo_2" | "empate"
    diferencia_media: float

    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return asdict(self)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, perdida: str = "mse", alpha: float = 0.05) -> ResultadoDM:
    """Prueba DM con corrección HLN para dos series de errores de pronóstico.

    Args:
        e1: Errores del modelo 1 (real − predicho).
        e2: Errores del modelo 2.
        h: Horizonte de pronóstico (número de rezagos de la autocovarianza a incluir = h−1).
        perdida: "mse" (diferencial e1² − e2²) o "mae" (|e1| − |e2|).
        alpha: Nivel de significancia para declarar un ganador.

    Returns:
        `ResultadoDM`. Si DM < 0 el modelo 1 tiene menor pérdida (es mejor).
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)
    if n < 5:
        return ResultadoDM(float("nan"), float("nan"), float("nan"), n, h, "empate", float("nan"))
    d = (e1**2 - e2**2) if perdida == "mse" else (np.abs(e1) - np.abs(e2))
    d_media = float(d.mean())
    # Varianza de largo plazo con autocovarianzas hasta h−1 (ventana rectangular)
    gamma0 = float(np.mean((d - d_media) ** 2))
    suma_gammas = 0.0
    for k in range(1, h):
        if k >= n:
            break
        gk = float(np.mean((d[k:] - d_media) * (d[:-k] - d_media)))
        suma_gammas += gk
    var_lp = (gamma0 + 2 * suma_gammas) / n
    if var_lp <= 0:
        var_lp = gamma0 / n if gamma0 > 0 else 1e-12
    dm = d_media / np.sqrt(var_lp)
    # Corrección de Harvey, Leybourne y Newbold (1997)
    factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * factor
    p = float(2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    if p < alpha:
        mejor = "modelo_1" if dm_hln < 0 else "modelo_2"
    else:
        mejor = "empate"
    return ResultadoDM(float(dm), float(dm_hln), p, n, h, mejor, d_media)
