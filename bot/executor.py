"""Ejecución de órdenes del bot, enrutada por el venue del activo.

- `binance_spot_testnet`: órdenes de mercado reales en `testnet.binance.vision` vía ccxt (sandbox).
- `alpaca_paper`: órdenes de mercado en Alpaca Paper (acciones y ETFs de EE. UU.).
- `paper`: simulación interna al último precio de yfinance / Binance mainnet.

Todos los modos devuelven el mismo diccionario: `{precio, qty, venue_order_id, modo}`.
Cualquier error del venue se registra, se emite a `sistema` y la orden cae a paper
(nunca se detiene el bot por un fallo externo). La DB es la fuente de verdad.
"""
from __future__ import annotations

import time
from typing import Any

from core.config import cfg, env
from core.events import emit, log_bot
from data_sources import alpaca

_exchange_cache: dict[str, Any] = {}


def _forzar_paper() -> bool:
    """True si el bot está forzado a paper en `config.yaml`."""
    return str(cfg("bot", "forzar_paper", default=False)).lower() == "true"


def _binance_testnet() -> Any | None:
    """Cliente ccxt autenticado en el Binance spot testnet (None si faltan credenciales)."""
    if "binance" in _exchange_cache:
        return _exchange_cache["binance"]
    import ccxt

    key, secret = env("BINANCE_SPOT_TESTNET_API_KEY"), env("BINANCE_SPOT_TESTNET_SECRET")
    if not key or not secret:
        _exchange_cache["binance"] = None
        return None
    try:
        ex = ccxt.binance({"apiKey": key, "secret": secret, "enableRateLimit": True, "options": {"defaultType": "spot"}})
        ex.set_sandbox_mode(True)
        ex.load_markets()
        _exchange_cache["binance"] = ex
    except Exception as e:
        log_bot.error("No se pudo conectar al Binance spot testnet: %s", e)
        _exchange_cache["binance"] = None
    return _exchange_cache["binance"]


def _paper(qty: float, precio_ref: float) -> dict[str, Any]:
    """Orden simulada."""
    return {"precio": float(precio_ref), "qty": float(qty), "venue_order_id": None, "modo": "paper"}


def _resultado_alpaca(orden: dict[str, Any], qty: float, precio_ref: float) -> dict[str, Any]:
    """Espera el llenado de una orden de Alpaca y devuelve la cantidad y el precio reales.

    Alpaca responde al envío con `filled_qty` en "0": esa cantidad no debe guardarse como la de la posición.
    """
    actual = orden
    for _ in range(int(cfg("bot", "alpaca_confirmacion_intentos", default=6))):
        if actual.get("status") == "filled":
            break
        time.sleep(float(cfg("bot", "alpaca_confirmacion_espera_s", default=0.5)))
        actual = alpaca.orden(str(orden.get("id")), ttl=0) or actual
    if actual.get("status") in ("rejected", "canceled", "expired"):
        raise RuntimeError(f"Alpaca dejó la orden en estado {actual.get('status')}")
    llenada = float(actual.get("filled_qty") or 0)
    return {"precio": float(actual.get("filled_avg_price") or precio_ref), "qty": llenada if llenada > 0 else float(actual.get("qty") or qty),
            "venue_order_id": str(orden.get("id")), "modo": "alpaca_paper"}


def ejecutar_orden(symbol: str, venue: str, lado: str, qty: float, precio_ref: float, abre_corto: bool = False) -> dict[str, Any]:
    """Envía una orden de mercado al venue del activo o la simula en paper.

    Args:
        symbol: Par (`SOL/USDT`) o ticker (`AAPL`).
        venue: `binance_spot_testnet` | `alpaca_paper` | `paper`.
        lado: "buy" o "sell".
        qty: Cantidad.
        precio_ref: Último precio conocido (para paper o si el venue no devuelve precio).
        abre_corto: True si la orden abre un corto (Alpaca solo los acepta en acciones enteras).
    """
    if _forzar_paper() or venue == "paper":
        return _paper(qty, precio_ref)
    try:
        if venue == "binance_spot_testnet":
            ex = _binance_testnet()
            if ex is None:
                raise RuntimeError("faltan BINANCE_SPOT_TESTNET_API_KEY/SECRET")
            q = float(ex.amount_to_precision(symbol, qty))
            if q <= 0:
                raise RuntimeError("cantidad por debajo del mínimo del par")
            orden = ex.create_order(symbol, "market", lado, q)
            return {"precio": float(orden.get("average") or orden.get("price") or precio_ref), "qty": float(orden.get("filled") or q),
                    "venue_order_id": str(orden.get("id")), "modo": "binance_spot_testnet"}
        if venue == "alpaca_paper":
            if not alpaca.disponible():
                raise RuntimeError("faltan ALPACA_API_KEY/ALPACA_SECRET_KEY")
            if alpaca.mercado_abierto() is False:
                raise RuntimeError("mercado de EE. UU. cerrado")
            return _resultado_alpaca(alpaca.enviar_orden_accion(symbol, lado, qty, abre_corto), qty, precio_ref)
        raise RuntimeError(f"venue desconocido {venue}")
    except Exception as e:
        log_bot.error("Orden en %s falló (%s %s %s): %s", venue, lado, qty, symbol, e)
        emit("sistema", "warning", f"Orden del bot cayó a paper ({symbol})", {"venue": venue, "lado": lado, "qty": qty, "motivo": str(e)[:300]})
        return _paper(qty, precio_ref)


def saldo_testnet() -> dict[str, float] | None:
    """Saldo libre en el Binance spot testnet (para mostrar en la app)."""
    ex = _binance_testnet()
    if ex is None:
        return None
    try:
        bal = ex.fetch_balance()
        return {k: float(v) for k, v in (bal.get("free") or {}).items() if v}
    except Exception as e:
        log_bot.warning("No se pudo leer el saldo del testnet: %s", e)
        return None


def reconciliar() -> dict[str, Any]:
    """Compara las posiciones de la DB con las del testnet/Alpaca al arrancar.

    El testnet de Binance se reinicia periódicamente: la DB manda. Si una posición
    abierta en la DB ya no existe en el venue, se registra en `sistema` y se sigue
    operando en paper para esa posición (la cuenta virtual no cambia).
    """
    from core import db

    resumen: dict[str, Any] = {"binance": None, "alpaca": None, "deribit": None, "huerfanas": []}
    from data_sources import deribit

    if deribit.disponible():
        resumen["deribit"] = {m: len(deribit.posiciones(m)) for m in deribit.MONEDAS}
    ex = _binance_testnet()
    if ex is not None:
        try:
            bal = ex.fetch_balance().get("total") or {}
            resumen["binance"] = {k: float(v) for k, v in bal.items() if v and float(v) > 0}
        except Exception as e:
            resumen["binance"] = f"error: {e}"
    if alpaca.disponible():
        resumen["alpaca"] = {p["symbol"]: float(p.get("qty") or 0) for p in alpaca.posiciones()}
    for est in db.consultar("SELECT s.*, a.symbol, a.venue FROM bot_strategy_state s JOIN bot_assets a ON a.id = s.asset_id WHERE s.position_json IS NOT NULL"):
        pos = db.de_json(est["position_json"], {})
        if est["venue"] == "binance_spot_testnet" and isinstance(resumen["binance"], dict):
            base = est["symbol"].split("/")[0]
            if pos.get("side") == "long" and resumen["binance"].get(base, 0.0) < pos.get("qty", 0) * 0.5:
                resumen["huerfanas"].append(f"{est['symbol']} [{est['strategy']}]")
        if est["venue"] == "alpaca_paper" and isinstance(resumen["alpaca"], dict) and est["symbol"] not in resumen["alpaca"]:
            resumen["huerfanas"].append(f"{est['symbol']} [{est['strategy']}]")
    if resumen["huerfanas"]:
        emit("sistema", "warning", "Reconciliación: posiciones de la DB que ya no existen en el venue (posible reseteo del testnet)",
             {"posiciones": resumen["huerfanas"], "accion": "la DB sigue siendo la fuente de verdad; se cierran en paper al precio actual"})
    return resumen
