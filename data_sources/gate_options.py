"""Gate.io testnet (`https://api-testnet.gateapi.io`): opciones cripto como respaldo de Deribit.

Cubre BTC, ETH, SOL, DOGE, XRP, BNB y otros subyacentes que el testnet de Deribit no tiene. Cada pata de la
cadena de Binance se mapea al contrato de Gate.io con el mismo subyacente, vencimiento, strike y tipo
(`BTC_USDT-20261204-58000-C`). Gate.io cotiza la prima en USDT por unidad del subyacente, igual que Binance,
y cada contrato vale `multiplier` unidades (0.01 BTC, 1 SOL…), así que la cantidad se convierte a contratos enteros.

Limitación verificada el 19 nov 2026: el libro de opciones del testnet está vacío, así que las órdenes quedan
abiertas sin llenarse. Por eso el copiloto las cancela si nadie las llena en `ejecucion.gate_minutos_espera`
y la posición pasa a paper interno con un aviso (ver `core/positions.py`).

Por seguridad este módulo solo habla con el host de testnet: no hay forma de configurarlo contra producción.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import requests

from core.config import env
from core.events import log_app

HOST, PREFIJO = "https://api-testnet.gateapi.io", "/api/v4"
_CONTRATOS: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def disponible() -> bool:
    """True si hay llaves del testnet de Gate.io."""
    return bool(env("GATE_API_KEY") and env("GATE_API_SECRET"))


def _llamar(metodo: str, ruta: str, params: dict[str, Any] | None = None, cuerpo: dict[str, Any] | None = None, firmado: bool = True) -> Any:
    """Petición REST v4 al testnet, firmada con HMAC-SHA512 (método, ruta, query, hash del cuerpo y timestamp)."""
    query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    texto = json.dumps(cuerpo) if cuerpo is not None else ""
    cabeceras = {"Accept": "application/json", "Content-Type": "application/json"}
    if firmado:
        ts = str(int(time.time()))
        firma_base = f"{metodo}\n{PREFIJO}{ruta}\n{query}\n{hashlib.sha512(texto.encode()).hexdigest()}\n{ts}"
        cabeceras.update({"KEY": env("GATE_API_KEY"), "Timestamp": ts,
                          "SIGN": hmac.new(env("GATE_API_SECRET").encode(), firma_base.encode(), hashlib.sha512).hexdigest()})
    url = f"{HOST}{PREFIJO}{ruta}" + (f"?{query}" if query else "")
    r = requests.request(metodo, url, headers=cabeceras, data=texto or None, timeout=20)
    if r.status_code >= 400:
        try:
            d = r.json()
            msg = f"{d.get('label', '')} {d.get('message', '')}".strip()
        except ValueError:
            msg = r.text[:200]
        raise RuntimeError(f"Gate.io testnet {r.status_code}: {msg}")
    return r.json()


def contratos(activo: str) -> list[dict[str, Any]]:
    """Contratos de opciones del subyacente (`BTC` → `BTC_USDT`), con caché de 10 minutos."""
    und = f"{activo.upper()}_USDT"
    if und in _CONTRATOS and time.time() - _CONTRATOS[und][0] < 600:
        return _CONTRATOS[und][1]
    datos = _llamar("GET", "/options/contracts", {"underlying": und}, firmado=False)
    _CONTRATOS[und] = (time.time(), datos)
    return datos


def _redondear(valor: float, paso: float) -> float:
    """Redondea al múltiplo de `paso` más cercano con aritmética decimal (sin errores de coma flotante)."""
    if paso <= 0:
        return valor
    p = Decimal(str(paso))
    return float((Decimal(str(valor)) / p).quantize(Decimal(1), rounding=ROUND_HALF_UP) * p)


def mapear_pata(activo: str, leg: dict[str, Any]) -> dict[str, Any]:
    """Traduce una pata de Binance a una orden de Gate.io: contrato, tamaño con signo y precio redondeado.

    Raises:
        ValueError: si el contrato no existe en el testnet o la cantidad es menor a un contrato.
    """
    vencimiento = str(leg["expiry"])[:10]
    es_call = str(leg["type"]).lower().startswith("c")
    for c in contratos(activo):
        fecha = datetime.fromtimestamp(float(c["expiration_time"]), tz=timezone.utc).date().isoformat()
        if fecha == vencimiento and bool(c["is_call"]) == es_call and abs(float(c["strike_price"]) - float(leg["strike"])) < 1e-9:
            mult = float(c["multiplier"])
            n = int(round(float(leg["qty"]) / mult))
            if n < int(c.get("order_size_min") or 1):
                raise ValueError(f"{float(leg['qty']):g} {activo} es menos de un contrato de {mult:g} en Gate.io")
            precio = max(_redondear(float(leg["price"]), float(c.get("order_price_round") or 0)), float(c.get("order_price_round") or 0))
            return {"contract": c["name"], "size": n if leg["side"] > 0 else -n, "price": precio, "multiplicador": mult}
    raise ValueError(f"no existe en Gate.io testnet el contrato {activo} {vencimiento} {float(leg['strike']):g} {'call' if es_call else 'put'}")


def _texto_precio(precio: float) -> str:
    """Precio como texto sin notación científica ni ceros sobrantes."""
    return format(Decimal(str(precio)).normalize(), "f")


def ordenar_estrategia(activo: str, legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Envía cada pata como orden límite GTC. Si alguna falla, cancela las ya enviadas y lanza RuntimeError."""
    ordenes = [mapear_pata(activo, leg) for leg in legs]  # primero validar todas, para no dejar patas sueltas
    enviadas: list[dict[str, Any]] = []
    try:
        for o in ordenes:
            enviadas.append(_llamar("POST", "/options/orders", cuerpo={"contract": o["contract"], "size": o["size"], "price": _texto_precio(o["price"]),
                                                                       "tif": "gtc", "text": "t-copiloto"}))
    except Exception as e:
        for o in enviadas:
            try:
                cancelar(str(o["id"]))
            except Exception:
                pass
        raise RuntimeError(f"pata rechazada por Gate.io testnet ({e})") from e
    return {"order_ids": [str(o["id"]) for o in enviadas], "contratos": [o["contract"] for o in ordenes],
            "detalle": f"{len(enviadas)} órdenes límite en Gate.io testnet ({', '.join(o['contract'] for o in ordenes)})"}


def orden(order_id: str) -> dict[str, Any]:
    """Estado de una orden: `status` (open | finished), `left` (contratos sin llenar) y `finish_as`."""
    return _llamar("GET", f"/options/orders/{order_id}")


def cancelar(order_id: str) -> dict[str, Any]:
    """Cancela una orden abierta."""
    return _llamar("DELETE", f"/options/orders/{order_id}")


def estado_ordenes(order_ids: list[str]) -> dict[str, Any]:
    """Resume varias órdenes: cuántas se llenaron por completo y cuáles siguen abiertas."""
    llenas, abiertas, otras = [], [], []
    for oid in order_ids:
        try:
            o = orden(oid)
        except Exception as e:
            log_app.warning("No se pudo consultar la orden %s de Gate.io: %s", oid, e)
            otras.append(oid)
            continue
        if o.get("status") == "open":
            abiertas.append(oid)
        elif int(o.get("left") or 0) == 0 and o.get("finish_as") == "filled":
            llenas.append(oid)
        else:
            otras.append(oid)
    return {"llenas": llenas, "abiertas": abiertas, "otras": otras, "todas_llenas": len(llenas) == len(order_ids) and bool(order_ids)}


def cuenta() -> dict[str, Any]:
    """Saldo de la cuenta de opciones del testnet."""
    return _llamar("GET", "/options/accounts")
