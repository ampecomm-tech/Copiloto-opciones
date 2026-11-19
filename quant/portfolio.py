"""Portafolio: griegas agregadas, frontera eficiente de Markowitz y correlación de Pearson."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant.strategies_options import griegas_posicion


def griegas_portafolio(posiciones: list[dict[str, Any]], spots: dict[str, float], r: float) -> dict[str, Any]:
    """Suma las griegas de todas las posiciones abiertas, expresadas en dólares.

    Args:
        posiciones: Filas de `option_positions` con `legs` (lista) y `multiplicador`.
        spots: Precio actual por activo.
        r: Tasa libre de riesgo.

    Returns:
        `{delta_dolares, gamma_dolares, theta_diaria, vega_total, por_activo: {...}}`.
    """
    total = {"delta_dolares": 0.0, "gamma_dolares": 0.0, "theta_diaria": 0.0, "vega_total": 0.0}
    por_activo: dict[str, dict[str, float]] = {}
    for pos in posiciones:
        S = spots.get(pos["asset"])
        if not S:
            continue
        g = griegas_posicion(pos["legs"], S, r, pos.get("multiplicador", 1.0), pos.get("dias_transcurridos", 0.0))
        fila = {"delta_dolares": g["delta"] * S, "gamma_dolares": g["gamma"] * S * S / 100, "theta_diaria": g["theta"], "vega_total": g["vega"]}
        acc = por_activo.setdefault(pos["asset"], {k: 0.0 for k in total})
        for k in total:
            total[k] += fila[k]
            acc[k] += fila[k]
    return {**total, "por_activo": por_activo}


def correlacion_pearson(retornos: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación de Pearson entre columnas (activos)."""
    return retornos.dropna().corr(method="pearson")


def pearson(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    """Coeficiente de correlación de Pearson entre dos series (NaN si no hay datos)."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def frontera_eficiente(retornos: pd.DataFrame, n_carteras: int = 3000, rf: float = 0.0, periodos_por_anio: int = 365, semilla: int = 7) -> dict[str, Any]:
    """Frontera eficiente de Markowitz por simulación de carteras aleatorias (long-only).

    Returns:
        `{"carteras": DataFrame(ret, vol, sharpe, pesos...), "max_sharpe": {...}, "min_vol": {...}}`.
    """
    rets = retornos.dropna()
    activos = list(rets.columns)
    if len(activos) < 2 or len(rets) < 10:
        return {"carteras": pd.DataFrame(), "max_sharpe": {}, "min_vol": {}, "activos": activos}
    mu = rets.mean().values * periodos_por_anio
    cov = rets.cov().values * periodos_por_anio
    rng = np.random.default_rng(semilla)
    pesos = rng.dirichlet(np.ones(len(activos)), size=n_carteras)
    ret_c = pesos @ mu
    vol_c = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))
    sharpe = (ret_c - rf) / np.where(vol_c > 0, vol_c, np.nan)
    df = pd.DataFrame({"ret": ret_c, "vol": vol_c, "sharpe": sharpe})
    for i, a in enumerate(activos):
        df[f"w_{a}"] = pesos[:, i]
    i_ms, i_mv = int(np.nanargmax(sharpe)), int(np.nanargmin(vol_c))
    return {"carteras": df, "max_sharpe": df.iloc[i_ms].to_dict(), "min_vol": df.iloc[i_mv].to_dict(), "activos": activos}


def metricas_desempeno(equity: pd.Series, periodos_por_anio: int = 365 * 24 * 4) -> dict[str, float]:
    """ROI, Sharpe, Sortino, drawdown máximo, VaR y CVaR al 95 % de una curva de capital.

    VaR y CVaR son históricos sobre los retornos de la curva (por operación si la curva es por operación),
    expresados como pérdida positiva; NaN con menos de dos retornos.

    Args:
        equity: Serie de capital en el tiempo.
        periodos_por_anio: Número de periodos por año (velas de 15 min → 35,040).
    """
    from quant.riesgo import var_cvar_historico

    e = pd.Series(equity).dropna().astype(float)
    if len(e) < 2 or e.iloc[0] <= 0:
        return {"roi": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "var_95": float("nan"), "cvar_95": float("nan")}
    rets = e.pct_change().dropna()
    cola = var_cvar_historico(rets.values, 0.95)
    roi = float(e.iloc[-1] / e.iloc[0] - 1)
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(periodos_por_anio)) if std > 0 else 0.0
    neg = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(periodos_por_anio)) if len(neg) > 1 and neg.std() > 0 else 0.0
    dd = (e / e.cummax() - 1).min()
    return {"roi": roi, "sharpe": sharpe, "sortino": sortino, "max_drawdown": float(dd), "var_95": cola["var"], "cvar_95": cola["cvar"]}
