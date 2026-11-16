"""Cadena de opciones de acciones y ETFs con yfinance (`Ticker.options`, `Ticker.option_chain`).

Respaldo de Alpaca. Devuelve el mismo formato normalizado que `binance_options.py` y `alpaca.py`. El multiplicador del
contrato es 100 acciones; los precios se dejan por acción.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from core.config import cfg
from core.events import log_app

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL_S = 180


def _cacheado(clave: str, fn, ttl: float = _TTL_S):
    """Ejecuta `fn()` con caché simple en memoria por clave."""
    ahora = time.time()
    if clave in _CACHE and ahora - _CACHE[clave][0] < ttl:
        return _CACHE[clave][1]
    val = fn()
    _CACHE[clave] = (ahora, val)
    return val


def spot(ticker: str) -> float:
    """Último precio del subyacente (fast_info o último cierre)."""
    import yfinance as yf

    def _f() -> float:
        t = yf.Ticker(ticker)
        try:
            p = t.fast_info.get("last_price") or t.fast_info.get("lastPrice")
            if p:
                return float(p)
        except Exception:
            pass
        h = t.history(period="5d")
        return float(h["Close"].dropna().iloc[-1])

    return _cacheado(f"spot:{ticker}", _f, 60)


def cadena_opciones(ticker: str, hasta_dias: int = 60, max_vencimientos: int = 10) -> tuple[pd.DataFrame, float]:
    """Descarga la cadena de los vencimientos hasta `hasta_dias` y la normaliza.

    Args:
        ticker: Símbolo de Yahoo (NVDA, AAPL, SPY…).
        hasta_dias: Se incluyen los vencimientos con DTE ≤ este valor (y al menos uno posterior).
        max_vencimientos: Tope de vencimientos a descargar (cada uno es una petición).

    Returns:
        (DataFrame normalizado, spot). DataFrame vacío si falla la descarga.
    """
    import yfinance as yf

    def _f() -> tuple[pd.DataFrame, float]:
        t = yf.Ticker(ticker)
        S = spot(ticker)
        ahora = datetime.now(timezone.utc)
        vtos: list[str] = []
        for v in list(t.options or []):
            dte_v = (datetime.strptime(v, "%Y-%m-%d").replace(tzinfo=timezone.utc) - ahora).days
            vtos.append(v)
            if dte_v > hasta_dias or len(vtos) >= max_vencimientos:
                break
        filas: list[dict[str, Any]] = []
        for v in vtos:
            try:
                oc = t.option_chain(v)
            except Exception as e:
                log_app.warning("No se pudo bajar %s %s: %s", ticker, v, e)
                continue
            exp = datetime.strptime(v, "%Y-%m-%d").replace(hour=20, tzinfo=timezone.utc)
            dte = (exp - ahora).total_seconds() / 86400
            for tipo, df in (("call", oc.calls), ("put", oc.puts)):
                for _, c in df.iterrows():
                    bid, ask = float(c.get("bid") or 0), float(c.get("ask") or 0)
                    iv = c.get("impliedVolatility")
                    filas.append({
                        "expiry": v, "dte": round(dte, 3), "strike": float(c["strike"]), "type": tipo,
                        "bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid and ask else float(c.get("lastPrice") or 0),
                        "iv": float(iv) if iv and iv == iv and iv > 0.001 else None,
                        "oi": float(c.get("openInterest") or 0), "volume": float(c.get("volume") or 0),
                        "symbol": str(c.get("contractSymbol", "")),
                    })
        out = pd.DataFrame(filas)
        if not out.empty:
            out = out[out["dte"] > 0].sort_values(["dte", "strike"]).reset_index(drop=True)
        return out, S

    try:
        return _cacheado(f"chain:{ticker}:{hasta_dias}", _f)
    except Exception as e:
        log_app.error("Cadena de yfinance no disponible para %s: %s", ticker, e)
        return pd.DataFrame(), float("nan")


def fecha_earnings(ticker: str) -> str | None:
    """Próxima fecha de resultados según yfinance (None si no hay)."""
    import yfinance as yf

    def _f() -> str | None:
        try:
            cal = yf.Ticker(ticker).calendar
            if isinstance(cal, dict):
                fechas = cal.get("Earnings Date") or []
                if fechas:
                    return str(fechas[0])
            elif cal is not None and not cal.empty and "Earnings Date" in cal.index:
                return str(cal.loc["Earnings Date"].iloc[0])
        except Exception:
            return None
        return None

    return _cacheado(f"earn:{ticker}", _f, 3600)
