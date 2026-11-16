"""Cadena de opciones europeas de Binance (API pública `https://eapi.binance.com`).

Subyacentes: BTC, ETH, BNB, SOL, XRP, DOGE (liquidación en USDT). Devuelve el
DataFrame normalizado que usa todo el pipeline (`expiry, dte, strike, type, bid,
ask, mid, iv, oi, volume, symbol`) y, además, el mark price, la IV y las griegas
que publica Binance, para la validación Black-Scholes vs Binance.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from core.events import log_app

BASE = "https://eapi.binance.com/eapi/v1"
SUBYACENTES: tuple[str, ...] = ("BTC", "ETH", "BNB", "SOL", "XRP", "DOGE")
_CACHE: dict[str, tuple[float, Any]] = {}
_TTL_S = 90


def _get(ruta: str, ttl: float = _TTL_S, **params: Any) -> Any:
    """GET con caché en memoria; lanza si la API falla."""
    clave = ruta + str(sorted(params.items()))
    ahora = time.time()
    if clave in _CACHE and ahora - _CACHE[clave][0] < ttl:
        return _CACHE[clave][1]
    r = requests.get(f"{BASE}/{ruta}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _CACHE[clave] = (ahora, data)
    return data


def precio_indice(moneda: str) -> float:
    """Precio índice (USDT) del subyacente."""
    return float(_get("index", ttl=20, underlying=f"{moneda.upper()}USDT")["indexPrice"])


def _open_interest(moneda: str, expiraciones: list[str]) -> dict[str, float]:
    """Open interest por símbolo para las expiraciones dadas (formato `YYMMDD`)."""
    oi: dict[str, float] = {}
    for exp in expiraciones:
        try:
            for fila in _get("openInterest", ttl=300, underlyingAsset=moneda.upper(), expiration=exp):
                oi[fila["symbol"]] = float(fila.get("sumOpenInterest") or 0)
        except Exception as e:
            log_app.warning("Open interest de Binance no disponible para %s %s: %s", moneda, exp, e)
    return oi


def cadena_opciones(moneda: str = "BTC", max_expiraciones: int = 8) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Descarga la cadena completa de la moneda con bid/ask, mark, IV y griegas de Binance.

    Returns:
        (DataFrame normalizado, spot en USDT, meta con `unit` y `paso_qty`).
        DataFrame vacío si la API falla.
    """
    moneda = moneda.upper()
    try:
        info = _get("exchangeInfo", ttl=600)
        marks = {m["symbol"]: m for m in _get("mark", ttl=30)}
        ticks = {t["symbol"]: t for t in _get("ticker", ttl=30)}
        S = precio_indice(moneda)
    except Exception as e:
        log_app.error("Binance Options no disponible para %s: %s", moneda, e)
        return pd.DataFrame(), float("nan"), {}
    ahora = datetime.now(timezone.utc)
    simbolos = [s for s in info["optionSymbols"] if s["underlying"] == f"{moneda}USDT" and s.get("status") == "TRADING"]
    if not simbolos:
        return pd.DataFrame(), S, {}
    expiraciones = sorted({s["symbol"].split("-")[1] for s in simbolos})[:max_expiraciones]
    oi = _open_interest(moneda, expiraciones)
    unit = float(simbolos[0].get("unit") or 1)
    paso = float(simbolos[0].get("minQty") or 0.01)
    filas = []
    for s in simbolos:
        try:
            _, fecha, _, _ = s["symbol"].split("-")
            if fecha not in expiraciones:
                continue
            exp = datetime.fromtimestamp(int(s["expiryDate"]) / 1000, tz=timezone.utc)
            dte = (exp - ahora).total_seconds() / 86400
            m, t = marks.get(s["symbol"], {}), ticks.get(s["symbol"], {})
            bid, ask = float(t.get("bidPrice") or 0), float(t.get("askPrice") or 0)
            mark = float(m.get("markPrice") or 0)
            filas.append({
                "expiry": exp.date().isoformat(), "dte": round(dte, 3), "strike": float(s["strikePrice"]),
                "type": "call" if s["side"] == "CALL" else "put", "bid": bid, "ask": ask,
                "mid": (bid + ask) / 2 if bid and ask else mark,
                "iv": float(m["markIV"]) if m.get("markIV") else None,
                "oi": oi.get(s["symbol"], 0.0), "volume": float(t.get("volume") or 0), "symbol": s["symbol"],
                # Datos publicados por Binance para la validación
                "mark_price": mark, "mark_iv": float(m["markIV"]) if m.get("markIV") else None,
                "bin_delta": float(m["delta"]) if m.get("delta") else None, "bin_gamma": float(m["gamma"]) if m.get("gamma") else None,
                "bin_theta": float(m["theta"]) if m.get("theta") else None, "bin_vega": float(m["vega"]) if m.get("vega") else None,
                "bin_r": float(m["riskFreeInterest"]) if m.get("riskFreeInterest") else None,
                "tick_size": float(next((f["tickSize"] for f in s.get("filters", []) if f["filterType"] == "PRICE_FILTER"), 0.001)),
                "unit": unit,
            })
        except Exception:
            continue
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df[df["dte"] > 0].sort_values(["dte", "strike"]).reset_index(drop=True)
    return df, S, {"unit": unit, "paso_qty": paso}


def velas(moneda: str = "BTC", dias: int = 120) -> pd.DataFrame:
    """Velas diarias del spot `XXXUSDT` en el Binance real (público) para la HV."""
    try:
        data = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": f"{moneda.upper()}USDT", "interval": "1d", "limit": min(dias, 1000)}, timeout=15).json()
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "_1", "_2", "_3", "_4", "_5", "_6"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")[["open", "high", "low", "close", "volume"]].astype(float)
    except Exception as e:
        log_app.error("Velas de Binance no disponibles: %s", e)
        return pd.DataFrame()
