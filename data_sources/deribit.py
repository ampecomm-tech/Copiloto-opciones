"""Deribit testnet (`https://test.deribit.com`): ejecución de opciones cripto del copiloto.

Los datos de la cadena siguen saliendo de Binance Options; al confirmar, cada pata se
mapea al instrumento de Deribit con el mismo subyacente, vencimiento, strike y tipo
(`BTC-25DEC26-70000-C`). Deribit cotiza las primas en la moneda base (BTC/ETH), así
que el precio en USD se convierte con el precio índice y se redondea al tick.
Solo BTC y ETH tienen opciones en el testnet.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import requests

from core.config import env
from core.events import log_app

BASE = "https://test.deribit.com/api/v2"
MONEDAS = ("BTC", "ETH")
_token: dict[str, Any] = {"valor": None, "expira": 0.0}
_instrumentos: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def disponible() -> bool:
    """True si hay credenciales del testnet de Deribit."""
    return bool(env("DERIBIT_TESTNET_CLIENT_ID") and env("DERIBIT_TESTNET_CLIENT_SECRET"))


def _auth() -> str:
    """Token OAuth (client_credentials) con caché hasta su expiración."""
    if _token["valor"] and time.time() < _token["expira"] - 30:
        return str(_token["valor"])
    r = requests.get(f"{BASE}/public/auth", params={"grant_type": "client_credentials", "client_id": env("DERIBIT_TESTNET_CLIENT_ID"),
                                                    "client_secret": env("DERIBIT_TESTNET_CLIENT_SECRET")}, timeout=15).json()
    if "result" not in r:
        raise RuntimeError(f"Deribit auth: {r.get('error')}")
    _token["valor"], _token["expira"] = r["result"]["access_token"], time.time() + float(r["result"].get("expires_in", 900))
    return str(_token["valor"])


def _privado(metodo: str, **params: Any) -> Any:
    """Llamada privada GET; lanza con el mensaje de Deribit si hay error."""
    r = requests.get(f"{BASE}/private/{metodo}", params=params, headers={"Authorization": f"Bearer {_auth()}"}, timeout=20).json()
    if "result" not in r:
        err = r.get("error") or {}
        raise RuntimeError(f"Deribit {metodo}: {err.get('message')} {err.get('data') or ''}".strip())
    return r["result"]


def _publico(metodo: str, **params: Any) -> Any:
    """Llamada pública GET."""
    r = requests.get(f"{BASE}/public/{metodo}", params=params, timeout=20).json()
    if "result" not in r:
        raise RuntimeError(f"Deribit {metodo}: {(r.get('error') or {}).get('message')}")
    return r["result"]


def instrumentos(moneda: str) -> dict[str, dict[str, Any]]:
    """Opciones activas de la moneda en el testnet, indexadas por `expiry|strike|type` (caché 10 min)."""
    moneda = moneda.upper()
    ahora = time.time()
    if moneda in _instrumentos and ahora - _instrumentos[moneda][0] < 600:
        return _instrumentos[moneda][1]
    idx: dict[str, dict[str, Any]] = {}
    for i in _publico("get_instruments", currency=moneda, kind="option", expired="false"):
        exp = datetime.utcfromtimestamp(i["expiration_timestamp"] / 1000).date().isoformat()
        tipo = "call" if i["option_type"] == "call" else "put"
        idx[f"{exp}|{float(i['strike']):g}|{tipo}"] = i
    _instrumentos[moneda] = (ahora, idx)
    return idx


def precio_indice(moneda: str) -> float:
    """Precio índice en USD."""
    return float(_publico("get_index_price", index_name=f"{moneda.lower()}_usd")["index_price"])


def mapear_pata(moneda: str, leg: dict[str, Any]) -> dict[str, Any]:
    """Instrumento de Deribit equivalente a una pata (mismo vencimiento, strike y tipo); lanza si no existe."""
    clave = f"{leg['expiry']}|{float(leg['strike']):g}|{leg['type']}"
    ins = instrumentos(moneda).get(clave)
    if not ins:
        raise RuntimeError(f"no existe en Deribit testnet el instrumento {moneda} {leg['expiry']} {leg['strike']:g} {leg['type']}")
    return ins


def _redondear(valor: float, paso: float) -> float:
    """Redondea al múltiplo de `paso` sin residuos binarios (usa Decimal)."""
    from decimal import ROUND_HALF_UP, Decimal

    d_paso = Decimal(str(paso))
    return float((Decimal(str(valor)) / d_paso).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * d_paso)


def tick_para(ins: dict[str, Any], precio: float) -> float:
    """Tick aplicable a un precio: Deribit usa ticks escalonados (`tick_size_steps`) para primas altas."""
    tick = float(ins["tick_size"])
    for paso in sorted(ins.get("tick_size_steps") or [], key=lambda x: float(x["above_price"])):
        if precio >= float(paso["above_price"]):
            tick = float(paso["tick_size"])
    return tick


def ordenar_estrategia(moneda: str, legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Envía cada pata como orden límite (compra al ask / venta al bid, en moneda base). Cancela lo enviado si una falla.

    Returns:
        `{order_ids, instrumentos, detalle}`.
    """
    moneda = moneda.upper()
    S = precio_indice(moneda)
    enviadas: list[dict[str, Any]] = []
    try:
        for leg in legs:
            ins = mapear_pata(moneda, leg)
            cantidad = max(_redondear(float(leg["qty"]), float(ins["min_trade_amount"])), float(ins["min_trade_amount"]))
            bruto = float(leg["price"]) / S
            tick = tick_para(ins, bruto)
            precio_base = max(_redondear(bruto, tick), float(ins["tick_size"]))
            metodo = "buy" if leg["side"] > 0 else "sell"
            res = _privado(metodo, instrument_name=ins["instrument_name"], amount=cantidad, type="limit", price=precio_base, time_in_force="good_til_cancelled")
            enviadas.append(res["order"])
    except Exception as e:
        for o in enviadas:
            try:
                _privado("cancel", order_id=o["order_id"])
            except Exception:
                pass
        raise RuntimeError(f"pata rechazada por Deribit testnet ({e})") from e
    return {"order_ids": [str(o["order_id"]) for o in enviadas], "instrumentos": [o["instrument_name"] for o in enviadas],
            "detalle": f"{len(enviadas)} órdenes límite en Deribit testnet ({', '.join(o['instrument_name'] for o in enviadas)})"}


def resumen_cuenta(moneda: str = "BTC") -> dict[str, Any]:
    """Equity y fondos disponibles en el testnet (requiere scope account:read)."""
    r = _privado("get_account_summary", currency=moneda.upper())
    return {"equity": r.get("equity"), "disponible": r.get("available_funds"), "moneda": moneda.upper()}


def posiciones(moneda: str = "BTC") -> list[dict[str, Any]]:
    """Posiciones de opciones abiertas en el testnet (para reconciliar)."""
    try:
        return _privado("get_positions", currency=moneda.upper(), kind="option")
    except Exception as e:
        log_app.warning("Posiciones de Deribit no disponibles: %s", e)
        return []


def perpetuo(moneda: str) -> dict[str, Any]:
    """Especificación del perpetuo inverso de la moneda (`BTC-PERPETUAL`): tamaño de contrato y mínimo, en USD."""
    return _publico("get_instrument", instrument_name=f"{moneda.upper()}-PERPETUAL")


def monto_perpetuo(qty_base: float, precio: float, contrato: float) -> float:
    """USD a enviar para cubrir `qty_base` unidades del subyacente: múltiplo del tamaño de contrato (0 si no alcanza uno)."""
    return float(round(abs(qty_base) * precio / contrato) * contrato)


def ordenar_perpetuo(moneda: str, lado: str, qty_base: float | None = None, monto_usd: float | None = None, reduce_only: bool = False) -> dict[str, Any]:
    """Orden a mercado en el perpetuo de Deribit testnet, para cubrir la delta de una posición de opciones.

    El perpetuo es inverso: se opera en USD (`amount`, múltiplo de `contract_size`), pero su ganancia en dólares
    equivale a tener `monto / precio` unidades del subyacente, así que la cobertura se expresa en esas unidades.

    Args:
        lado: `buy` o `sell`.
        qty_base: Unidades del subyacente a cubrir (se convierten a USD con el precio índice).
        monto_usd: Alternativa a `qty_base`: USD exactos (se usa al cerrar, para deshacer lo mismo que se abrió).
        reduce_only: True al cerrar, para que la orden nunca abra en sentido contrario.

    Returns:
        `{order_id, instrumento, precio, monto_usd, qty_base}` con lo realmente ejecutado; `qty_base` lleva signo.
    """
    moneda = moneda.upper()
    ins = perpetuo(moneda)
    contrato = float(ins.get("contract_size") or 10)
    if monto_usd is None:
        monto_usd = monto_perpetuo(float(qty_base or 0), precio_indice(moneda), contrato)
    monto_usd = float(round(monto_usd / contrato) * contrato)
    if monto_usd < float(ins.get("min_trade_amount") or contrato):
        raise RuntimeError(f"la cobertura es menor que un contrato de {ins['instrument_name']} ({contrato:g} USD)")
    res = _privado(lado, instrument_name=ins["instrument_name"], amount=monto_usd, type="market", reduce_only=str(reduce_only).lower())
    orden = res["order"]
    lleno = float(orden.get("filled_amount") or 0)
    precio = float(orden.get("average_price") or 0)
    if lleno <= 0 or precio <= 0:
        raise RuntimeError(f"la orden {orden.get('order_id')} no se llenó (estado {orden.get('order_state')})")
    signo = 1 if lado == "buy" else -1
    return {"order_id": str(orden["order_id"]), "instrumento": ins["instrument_name"], "precio": precio, "monto_usd": lleno, "qty_base": signo * lleno / precio}
