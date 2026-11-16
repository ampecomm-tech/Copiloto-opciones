"""Validación: nuestro Black-Scholes contra el mark price, la IV y las griegas que publica Binance.

El mark price de Binance Options también sale de Black-Scholes (con su IV de marca y
su tasa), así que las diferencias deben ser pequeñas: sirve para comprobar la
implementación de `quant/black_scholes.py` con datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.black_scholes import anios, griegas, volatilidad_implicita


def comparar(cadena: pd.DataFrame, S: float, r_default: float = 0.045, max_filas: int = 400) -> pd.DataFrame:
    """Tabla contrato a contrato: precio BS vs mark, IV recuperada vs markIV y griegas nuestras vs Binance.

    Args:
        cadena: DataFrame de `binance_options.cadena_opciones` (con columnas `mark_*` y `bin_*`).
        S: Precio índice del subyacente.
        r_default: Tasa si Binance no publica `riskFreeInterest`.
        max_filas: Límite de contratos (los más cercanos al dinero primero).

    Returns:
        DataFrame con columnas `symbol, type, strike, dte, mark_iv, bs_precio, mark_price, err_precio, err_precio_pct,
        iv_recuperada, err_iv, delta_bs, delta_bin, gamma_bs, gamma_bin, theta_bs, theta_bin, vega_bs, vega_bin`.
    """
    if cadena.empty or "mark_price" not in cadena:
        return pd.DataFrame()
    c = cadena.dropna(subset=["mark_iv", "mark_price"]).copy()
    c = c[c["mark_price"] > 0]
    c["moneyness"] = (c["strike"] / S - 1).abs()
    c = c.sort_values("moneyness").head(max_filas)
    filas = []
    for _, f in c.iterrows():
        T = anios(float(f["dte"]))
        r = float(f["bin_r"]) if f.get("bin_r") == f.get("bin_r") and f.get("bin_r") is not None else r_default
        g = griegas(S, float(f["strike"]), T, r, float(f["mark_iv"]), str(f["type"]))
        iv_rec = volatilidad_implicita(float(f["mark_price"]), S, float(f["strike"]), T, r, str(f["type"]))
        filas.append({
            "symbol": f["symbol"], "type": f["type"], "strike": f["strike"], "dte": round(float(f["dte"]), 2), "mark_iv": float(f["mark_iv"]),
            "bs_precio": g.precio, "mark_price": float(f["mark_price"]), "err_precio": g.precio - float(f["mark_price"]),
            "err_precio_pct": (g.precio / float(f["mark_price"]) - 1) * 100 if f["mark_price"] else np.nan,
            "iv_recuperada": iv_rec, "err_iv": (iv_rec - float(f["mark_iv"])) if iv_rec is not None else np.nan,
            "delta_bs": g.delta, "delta_bin": f.get("bin_delta"), "gamma_bs": g.gamma, "gamma_bin": f.get("bin_gamma"),
            "theta_bs": g.theta, "theta_bin": f.get("bin_theta"), "vega_bs": g.vega, "vega_bin": f.get("bin_vega"),
        })
    return pd.DataFrame(filas)


def resumen(tabla: pd.DataFrame) -> dict[str, float]:
    """Errores agregados: MAE y error porcentual mediano del precio, MAE de IV y de cada griega, correlación de deltas."""
    if tabla.empty:
        return {}
    out = {
        "n": int(len(tabla)),
        "mae_precio": float(tabla["err_precio"].abs().mean()),
        "err_precio_pct_mediano": float(tabla["err_precio_pct"].abs().median()),
        "mae_iv": float(tabla["err_iv"].abs().mean()),
    }
    for g in ("delta", "gamma", "theta", "vega"):
        a, b = pd.to_numeric(tabla[f"{g}_bs"], errors="coerce"), pd.to_numeric(tabla[f"{g}_bin"], errors="coerce")
        m = a.notna() & b.notna()
        out[f"mae_{g}"] = float((a[m] - b[m]).abs().mean()) if m.any() else float("nan")
        out[f"corr_{g}"] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")
    return out
