"""Alpaca: datos de opciones de acciones (fuente principal) y órdenes en Alpaca Paper.

- Datos: `https://data.alpaca.markets/v1beta1/options/snapshots/{subyacente}` (feed indicativo,
  disponible en cuentas paper) + `/v2/options/contracts` para strikes, vencimientos y OI.
- Órdenes: `POST /v2/orders` en `ALPACA_BASE_URL` (paper), con soporte multipata (`order_class=mleg`).
- Reloj de mercado: `GET /v2/clock`.

Sin `ALPACA_API_KEY` todas las funciones devuelven vacío/None y el llamador cae a yfinance o paper.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from core.config import env
from core.events import log_app

DATA_BASE = "https://data.alpaca.markets"
_CACHE: dict[str, tuple[float, Any]] = {}


def disponible() -> bool:
    """True si hay credenciales de Alpaca."""
    return bool(env("ALPACA_API_KEY") and env("ALPACA_SECRET_KEY"))


def _headers() -> dict[str, str]:
    """Cabeceras de autenticación."""
    return {"APCA-API-KEY-ID": env("ALPACA_API_KEY"), "APCA-API-SECRET-KEY": env("ALPACA_SECRET_KEY"), "Content-Type": "application/json"}


def _base() -> str:
    """URL base de trading (paper por defecto)."""
    return env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def _get(url: str, ttl: float = 60, **params: Any) -> Any:
    """GET autenticado con caché."""
    clave = url + str(sorted(params.items()))
    ahora = time.time()
    if clave in _CACHE and ahora - _CACHE[clave][0] < ttl:
        return _CACHE[clave][1]
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    _CACHE[clave] = (ahora, data)
    return data


def reloj() -> dict[str, Any] | None:
    """Estado del mercado de EE. UU.: `{is_open, next_open, next_close}` o None sin credenciales."""
    if not disponible():
        return None
    try:
        return _get(f"{_base()}/v2/clock", ttl=30)
    except Exception as e:
        log_app.warning("Reloj de Alpaca no disponible: %s", e)
        return None


def mercado_abierto() -> bool | None:
    """True/False según el reloj de Alpaca; None si no se pudo consultar."""
    c = reloj()
    return bool(c["is_open"]) if c else None


def spot(ticker: str) -> float | None:
    """Último precio del subyacente (trade más reciente)."""
    if not disponible():
        return None
    try:
        d = _get(f"{DATA_BASE}/v2/stocks/{ticker}/trades/latest", ttl=20, feed="iex")
        return float(d["trade"]["p"])
    except Exception as e:
        log_app.warning("Spot de Alpaca no disponible para %s: %s", ticker, e)
        return None


def _contratos(ticker: str, hasta_dias: int) -> list[dict[str, Any]]:
    """Lista de contratos activos del subyacente hasta `hasta_dias` (paginando)."""
    hoy = datetime.now(timezone.utc).date()
    params: dict[str, Any] = {"underlying_symbols": ticker, "status": "active", "expiration_date_gte": hoy.isoformat(),
                              "expiration_date_lte": (hoy + timedelta(days=hasta_dias)).isoformat(), "limit": 10000}
    out: list[dict[str, Any]] = []
    token = None
    for _ in range(10):
        if token:
            params["page_token"] = token
        d = _get(f"{_base()}/v2/options/contracts", ttl=300, **params)
        out += d.get("option_contracts", [])
        token = d.get("next_page_token")
        if not token:
            break
    return out


def cadena_opciones(ticker: str, hasta_dias: int = 60) -> tuple[pd.DataFrame, float]:
    """Cadena normalizada desde Alpaca (contratos + snapshots con cotización, IV y griegas).

    Returns:
        (DataFrame, spot). DataFrame vacío si no hay credenciales o la API falla.
    """
    if not disponible():
        return pd.DataFrame(), float("nan")
    try:
        contratos = _contratos(ticker, hasta_dias)
        snaps: dict[str, Any] = {}
        token = None
        for _ in range(20):
            params: dict[str, Any] = {"feed": "indicative", "limit": 1000}
            if token:
                params["page_token"] = token
            d = _get(f"{DATA_BASE}/v1beta1/options/snapshots/{ticker}", ttl=60, **params)
            snaps.update(d.get("snapshots", {}))
            token = d.get("next_page_token")
            if not token:
                break
        S = spot(ticker) or float("nan")
    except Exception as e:
        log_app.error("Cadena de Alpaca no disponible para %s: %s", ticker, e)
        return pd.DataFrame(), float("nan")
    ahora = datetime.now(timezone.utc)
    filas = []
    for c in contratos:
        s = snaps.get(c["symbol"])
        if not s:
            continue
        q = s.get("latestQuote") or {}
        g = s.get("greeks") or {}
        exp = datetime.strptime(c["expiration_date"], "%Y-%m-%d").replace(hour=20, tzinfo=timezone.utc)
        bid, ask = float(q.get("bp") or 0), float(q.get("ap") or 0)
        iv = s.get("impliedVolatility")
        filas.append({
            "expiry": c["expiration_date"], "dte": round((exp - ahora).total_seconds() / 86400, 3), "strike": float(c["strike_price"]),
            "type": c["type"], "bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid and ask else float((s.get("latestTrade") or {}).get("p") or 0),
            "iv": float(iv) if iv else None, "oi": float(c.get("open_interest") or 0), "volume": float((s.get("dailyBar") or {}).get("v") or 0),
            "symbol": c["symbol"], "alp_delta": g.get("delta"), "alp_gamma": g.get("gamma"), "alp_theta": g.get("theta"), "alp_vega": g.get("vega"),
        })
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df[df["dte"] > 0].sort_values(["dte", "strike"]).reset_index(drop=True)
    return df, S


# ---------- Órdenes ----------

def enviar_orden_multipata(legs: list[dict[str, Any]], qty: int, limite: float | None = None) -> dict[str, Any]:
    """Orden de opciones en Alpaca Paper: una pata simple o `mleg` para spreads/condors.

    Args:
        legs: Patas con `symbol` (OCC), `side` (+1 compra / -1 venta) y `ratio` (entero).
        qty: Número de "unidades" de la estructura.
        limite: Precio límite neto (débito positivo / crédito negativo); None = mercado.

    Returns:
        Respuesta de Alpaca (incluye `id`, `status`, `legs`). Lanza si la API rechaza.
    """
    if len(legs) == 1:
        cuerpo: dict[str, Any] = {"symbol": legs[0]["symbol"], "qty": str(qty * legs[0].get("ratio", 1)), "side": "buy" if legs[0]["side"] > 0 else "sell",
                                  "type": "limit" if limite is not None else "market", "time_in_force": "day"}
        if limite is not None:
            cuerpo["limit_price"] = str(round(abs(limite), 2))
    else:
        cuerpo = {"order_class": "mleg", "qty": str(qty), "type": "limit" if limite is not None else "market", "time_in_force": "day",
                  "legs": [{"symbol": l["symbol"], "ratio_qty": str(l.get("ratio", 1)), "side": "buy" if l["side"] > 0 else "sell",
                            "position_intent": "buy_to_open" if l["side"] > 0 else "sell_to_open"} for l in legs]}
        if limite is not None:
            cuerpo["limit_price"] = str(round(limite, 2))
    r = requests.post(f"{_base()}/v2/orders", headers=_headers(), json=cuerpo, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca rechazó la orden ({r.status_code}): {r.text[:300]}")
    return r.json()


def enviar_orden_accion(symbol: str, lado: str, qty: float) -> dict[str, Any]:
    """Orden de mercado de acciones/ETFs en Alpaca Paper (para el bot). Permite fracciones en compras."""
    cuerpo = {"symbol": symbol, "qty": str(round(qty, 4)) if lado == "buy" else str(int(qty)), "side": lado, "type": "market", "time_in_force": "day"}
    r = requests.post(f"{_base()}/v2/orders", headers=_headers(), json=cuerpo, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca rechazó la orden ({r.status_code}): {r.text[:300]}")
    return r.json()


def orden(order_id: str) -> dict[str, Any] | None:
    """Consulta una orden por id."""
    try:
        return _get(f"{_base()}/v2/orders/{order_id}", ttl=5)
    except Exception:
        return None


def posiciones() -> list[dict[str, Any]]:
    """Posiciones abiertas en la cuenta paper (para reconciliar)."""
    if not disponible():
        return []
    try:
        return _get(f"{_base()}/v2/positions", ttl=10)
    except Exception as e:
        log_app.warning("Posiciones de Alpaca no disponibles: %s", e)
        return []
