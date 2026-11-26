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


# ---------- Alpaca con órdenes netas ----------
# Alpaca guarda una sola posición neta por acción para toda la cuenta, mientras el bot lleva una posición por
# estrategia. Si cada estrategia mandara su propia orden, chocarían: no se puede abrir un corto con la cuenta en
# largo, y dos órdenes opuestas seguidas se bloquean como "wash trade". Por eso, en Alpaca, cada apertura o cierre
# mueve la cuenta a la suma de las posiciones reales de todas las estrategias y solo envía la diferencia.

def es_posicion_en_broker(pos: dict[str, Any] | None, venue: str) -> bool:
    """True si la posición de una estrategia está realmente en el broker (no se abrió en paper interno).

    Para posiciones anteriores al campo `modo` se usa el id de orden guardado en `bot_trades`.
    """
    from core import db

    if not pos or float(pos.get("qty") or 0) <= 0:
        return False
    modo = pos.get("modo")
    if modo is None:
        fila = db.consultar_uno("SELECT venue_order_id FROM bot_trades WHERE id = ?", (pos.get("trade_id"),))
        return bool(fila and fila.get("venue_order_id"))
    return modo == venue


def coberturas_en_alpaca(symbol: str) -> float:
    """Acciones netas que las coberturas abiertas del copiloto tienen en Alpaca para ese símbolo."""
    from core import db

    fila = db.consultar_uno("SELECT COALESCE(SUM(qty), 0) AS q FROM option_hedges WHERE venue = 'alpaca_paper' AND instrumento = ? AND status = 'abierta'", (symbol,))
    return float((fila or {}).get("q") or 0.0)


def neta_de_estrategias(asset_id: int, venue: str, excluir: str | None = None) -> float:
    """Suma con signo de las posiciones en el broker de todas las estrategias del activo, salvo `excluir`.

    En Alpaca también suma las coberturas abiertas del copiloto sobre el mismo símbolo, para que el bot no las deshaga.
    """
    from core import db

    total = 0.0
    for est in db.consultar("SELECT strategy, position_json FROM bot_strategy_state WHERE asset_id = ? AND position_json IS NOT NULL", (asset_id,)):
        if est["strategy"] == excluir:
            continue
        pos = db.de_json(est["position_json"], {})
        if es_posicion_en_broker(pos, venue):
            total += float(pos["qty"]) * (1 if pos["side"] == "long" else -1)
    if venue == "alpaca_paper":
        fila = db.consultar_uno("SELECT symbol FROM bot_assets WHERE id = ?", (asset_id,))
        if fila:
            total += coberturas_en_alpaca(fila["symbol"])
    return total


def neta_alpaca_simbolo(symbol: str, extra: float = 0.0) -> float:
    """Posición neta que la cuenta de Alpaca debe tener en `symbol`: estrategias del bot, coberturas y `extra`."""
    from core import db

    fila = db.consultar_uno("SELECT id FROM bot_assets WHERE symbol = ? AND venue = 'alpaca_paper'", (symbol,))
    bot = neta_de_estrategias(int(fila["id"]), "alpaca_paper") if fila else coberturas_en_alpaca(symbol)
    return bot + extra


def plan_alpaca(actual: float, objetivo: float) -> list[tuple[str, float, bool]]:
    """Órdenes para llevar la cuenta de `actual` a `objetivo` acciones: `(lado, cantidad, abre_corto)`.

    Los cortos solo se admiten en acciones enteras, así que un objetivo negativo se redondea hacia cero.
    Cruzar de largo a corto (o al revés) se parte en dos órdenes: primero cerrar y después abrir.
    """
    import math

    if objetivo < 0:
        objetivo = -math.floor(-objetivo + 1e-9)
    if abs(objetivo - actual) < 1e-4:
        return []
    if actual > 0 > objetivo:
        return [("sell", actual, False), ("sell", -objetivo, True)]
    if actual < 0 < objetivo:
        return [("buy", -actual, False), ("buy", objetivo, False)]
    delta = objetivo - actual
    if delta > 0:
        return [("buy", delta, False)]
    return [("sell", -delta, objetivo < 0)]


def _esperar_sin_ordenes_abiertas(symbol: str) -> None:
    """Espera a que no haya órdenes abiertas del símbolo, para no disparar el bloqueo de wash trade de Alpaca."""
    for _ in range(int(cfg("bot", "alpaca_confirmacion_intentos", default=6)) * 2):
        if not alpaca.ordenes_abiertas(symbol):
            return
        time.sleep(float(cfg("bot", "alpaca_confirmacion_espera_s", default=0.5)))
    raise RuntimeError("hay órdenes del mismo símbolo todavía abiertas en Alpaca")


def mover_cuenta_alpaca(symbol: str, objetivo: float, precio_ref: float) -> dict[str, Any]:
    """Lleva la cuenta de Alpaca de su posición actual a `objetivo` acciones; lanza RuntimeError si no puede.

    Returns:
        `{precio, ids, actual}`: precio promedio de lo ejecutado, ids de las órdenes y la posición antes de mover.
    """
    if not alpaca.disponible():
        raise RuntimeError("faltan ALPACA_API_KEY/ALPACA_SECRET_KEY")
    if alpaca.mercado_abierto() is False:
        raise RuntimeError("mercado de EE. UU. cerrado")
    _esperar_sin_ordenes_abiertas(symbol)
    actual = alpaca.posicion_neta(symbol)
    ids, valor, cantidad = [], 0.0, 0.0
    for lado_o, q, corto in plan_alpaca(actual, objetivo):
        r = _resultado_alpaca(alpaca.enviar_orden_accion(symbol, lado_o, q, corto), q, precio_ref)
        ids.append(r["venue_order_id"])
        valor += r["precio"] * r["qty"]
        cantidad += r["qty"]
    return {"precio": valor / cantidad if cantidad > 0 else float(precio_ref), "ids": ids, "actual": actual}


def ejecutar_neta_alpaca(asset: dict[str, Any], estrategia: str, objetivo: float, qty: float, lado: str, precio_ref: float) -> dict[str, Any]:
    """Lleva la cuenta de Alpaca a `objetivo` acciones netas y registra para la estrategia su propia cantidad.

    Args:
        asset: Fila de `bot_assets`.
        estrategia: Estrategia que abre o cierra (solo para el aviso si falla).
        objetivo: Posición neta que la cuenta debe tener después de la operación.
        qty: Cantidad de la estrategia (lo que se registra en su posición virtual).
        lado: "buy" o "sell", desde el punto de vista de la estrategia.
        precio_ref: Último precio conocido.

    Returns:
        `{precio, qty, venue_order_id, modo}`. Si el neto no cambia (otra estrategia compensa), no se envía nada y
        el modo sigue siendo `alpaca_paper`. Si Alpaca falla, cae a paper con aviso en `sistema`.
    """
    symbol = asset["symbol"]
    if _forzar_paper():
        return _paper(qty, precio_ref)
    try:
        r = mover_cuenta_alpaca(symbol, objetivo, precio_ref)
        if not r["ids"]:
            log_bot.info("Alpaca %s [%s]: la operación se compensa con otras estrategias, sin orden al broker", symbol, estrategia)
        return {"precio": r["precio"], "qty": float(qty), "venue_order_id": ",".join(r["ids"]) if r["ids"] else "neteada", "modo": "alpaca_paper",
                "neto_antes": r["actual"], "neto_despues": objetivo}
    except Exception as e:
        log_bot.error("Orden neta en Alpaca falló (%s %s %s): %s", lado, qty, symbol, e)
        emit("sistema", "warning", f"Orden del bot cayó a paper ({symbol})",
             {"venue": "alpaca_paper", "estrategia": estrategia, "lado": lado, "qty": qty, "objetivo_neto": round(objetivo, 4), "motivo": str(e)[:300]})
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
        if pos.get("modo") == "paper":
            continue  # abierta en paper interno: no existe en el venue y no debe buscarse ahí
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
