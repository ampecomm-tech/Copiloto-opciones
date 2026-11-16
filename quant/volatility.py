"""Volatilidad histórica (HV) y comparación con la volatilidad implícita (IV)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def volatilidad_historica(cierres: pd.Series, ventana: int = 30, periodos_por_anio: int = 365) -> float:
    """HV anualizada: desviación estándar de log-retornos de la ventana × √periodos.

    Args:
        cierres: Serie de precios de cierre (índice temporal).
        ventana: Número de retornos a considerar (los últimos).
        periodos_por_anio: 365 para cripto (opera todos los días), 252 para acciones.

    Returns:
        HV anualizada como fracción (0.65 = 65%). NaN si no hay datos suficientes.
    """
    s = pd.Series(cierres).dropna().astype(float)
    if len(s) < 3:
        return float("nan")
    rets = np.log(s / s.shift(1)).dropna().tail(ventana)
    if len(rets) < 2:
        return float("nan")
    return float(rets.std(ddof=1) * math.sqrt(periodos_por_anio))


def cono_volatilidad(cierres: pd.Series, ventanas: tuple[int, ...] = (10, 20, 30, 60, 90), periodos_por_anio: int = 365) -> pd.DataFrame:
    """HV para varias ventanas: útil para mostrar dónde cae la IV respecto a la historia."""
    filas = [{"ventana": v, "hv": volatilidad_historica(cierres, v, periodos_por_anio)} for v in ventanas]
    return pd.DataFrame(filas)


def ratio_iv_hv(iv: float, hv: float) -> float:
    """Cociente IV/HV; > 1 opciones caras respecto a la historia, < 1 baratas."""
    if hv is None or hv != hv or hv <= 0 or iv is None or iv != iv:
        return float("nan")
    return float(iv / hv)


def iv_atm(cadena: pd.DataFrame, S: float, dte_objetivo: float | None = None) -> float:
    """IV at-the-money: promedio de la IV de call y put con strike más cercano a `S`.

    Args:
        cadena: DataFrame con columnas `strike`, `type`, `iv`, `dte`.
        S: Precio del subyacente.
        dte_objetivo: Si se da, usa el vencimiento más cercano a estos días.
    """
    c = cadena.dropna(subset=["iv"])
    if c.empty:
        return float("nan")
    if dte_objetivo is not None:
        dte_mas_cercano = c.iloc[(c["dte"] - dte_objetivo).abs().argsort().iloc[0]]["dte"]
        c = c[c["dte"] == dte_mas_cercano]
    c = c.iloc[(c["strike"] - S).abs().argsort()[:4]]
    return float(c["iv"].mean())
