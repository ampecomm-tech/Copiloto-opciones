"""Ejecución de estrategias de opciones al confirmar en el copiloto.

- Cripto: órdenes reales en el **testnet de Deribit** (BTC y ETH; `config.ejecucion.venue_cripto =
  deribit_testnet`, por defecto) o en el testnet de opciones de Binance (`binance_options_testnet`,
  REST base configurable, firmado con la llave del testnet de futuros).
- Acciones/ETFs: órdenes en **Alpaca Paper**, multipata (`mleg`) para spreads y condors.
- Respaldo cripto: si Deribit no puede (subyacente sin opciones, strike inexistente, API caída), la estrategia
  se envía al **testnet de opciones de Gate.io** (`config.ejecucion.gate_respaldo`). Como ese libro no tiene
  contraparte, la orden queda "en libro" y `core/positions.py` la cancela si nadie la llena a tiempo.
- Si el venue rechaza una pata o la API falla, se usa **paper interno** y se emite un
  evento a `sistema` explicando por qué. La DB es la fuente de verdad.

Resultado común: `{"venue": str, "order_ids": list[str], "detalle": str, "estado": str}`.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import requests

from core.config import cfg, env
from core.events import emit, log_app
from data_sources import alpaca, deribit, gate_options

VENUES = ("deribit_testnet", "gate_options_testnet", "binance_options_testnet", "alpaca_paper", "paper")


def gate_respaldo_activo() -> bool:
    """True si el respaldo de Gate.io está habilitado en config y hay llaves del testnet."""
    return str(cfg("ejecucion", "gate_respaldo", default=True)).lower() == "true" and gate_options.disponible()


def ordenar_gate(estrategia: dict[str, Any]) -> dict[str, Any]:
    """Envía la estrategia al testnet de opciones de Gate.io; queda en libro hasta que alguien la llene."""
    from datetime import datetime, timezone

    activo = str(estrategia.get("activo", "BTC")).upper()
    r = gate_options.ordenar_estrategia(activo, estrategia["legs"])
    return {"venue": "gate_options_testnet", "order_ids": r["order_ids"], "estado": "en_libro", "contratos": r["contratos"],
            "enviada_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "detalle": r["detalle"] + f" · en libro: se cancela si nadie la llena en {int(cfg('ejecucion', 'gate_minutos_espera', default=10))} min"}


def venue_cripto() -> str:
    """Venue configurado para opciones cripto (`deribit_testnet` por defecto)."""
    return str(cfg("ejecucion", "venue_cripto", default="deribit_testnet"))


def ordenar_deribit(estrategia: dict[str, Any]) -> dict[str, Any]:
    """Envía la estrategia al testnet de Deribit mapeando cada pata al instrumento equivalente."""
    moneda = str(estrategia.get("activo", "BTC")).upper()
    if moneda not in deribit.MONEDAS:
        raise RuntimeError(f"Deribit testnet solo tiene opciones de {', '.join(deribit.MONEDAS)} (activo {moneda})")
    r = deribit.ordenar_estrategia(moneda, estrategia["legs"])
    return {"venue": "deribit_testnet", "order_ids": r["order_ids"], "detalle": r["detalle"], "estado": "enviada", "instrumentos": r["instrumentos"]}


# ---------- Binance options testnet (firmado) ----------

def _binance_base() -> str:
    """URL base del testnet de opciones (configurable)."""
    return str(cfg("ejecucion", "binance_options_testnet_url", default="https://demo-fapi.binance.com")).rstrip("/")


def binance_disponible() -> bool:
    """True si hay llave del testnet de futuros de Binance."""
    return bool(env("BINANCE_FUTURES_TESTNET_API_KEY") and env("BINANCE_FUTURES_TESTNET_SECRET"))


def _firmar(params: dict[str, Any]) -> str:
    """Query string firmada HMAC-SHA256 (esquema estándar de Binance)."""
    params = {**params, "timestamp": int(time.time() * 1000), "recvWindow": 5000}
    q = urlencode(params)
    firma = hmac.new(env("BINANCE_FUTURES_TESTNET_SECRET").encode(), q.encode(), hashlib.sha256).hexdigest()
    return f"{q}&signature={firma}"


def _binance_request(metodo: str, ruta: str, params: dict[str, Any]) -> Any:
    """Petición firmada al testnet de opciones; lanza con el mensaje del exchange si falla."""
    url = f"{_binance_base()}/eapi/v1/{ruta}?{_firmar(params)}"
    r = requests.request(metodo, url, headers={"X-MBX-APIKEY": env("BINANCE_FUTURES_TESTNET_API_KEY")}, timeout=20)
    if r.status_code >= 400:
        cuerpo = r.text.strip()
        if cuerpo.startswith("<"):
            cuerpo = "el host no expone la API de opciones (/eapi) — configura `ejecucion.binance_options_testnet_url` o deja el respaldo en paper"
        raise RuntimeError(f"Binance options testnet {r.status_code}: {cuerpo[:300]}")
    return r.json()


def _redondear_tick(precio: float, tick: float) -> float:
    """Redondea el precio al tick del contrato."""
    if tick <= 0:
        return round(precio, 4)
    return round(round(precio / tick) * tick, 8)


def ordenar_binance(estrategia: dict[str, Any]) -> dict[str, Any]:
    """Envía cada pata como orden LIMIT al precio de ejecución (ask al comprar, bid al vender).

    Si alguna pata es rechazada se cancelan las ya enviadas y se lanza RuntimeError.
    """
    enviadas: list[dict[str, Any]] = []
    try:
        for leg in estrategia["legs"]:
            params = {"symbol": leg["symbol"], "side": "BUY" if leg["side"] > 0 else "SELL", "type": "LIMIT", "timeInForce": "GTC",
                      "quantity": f"{leg['qty']:.2f}", "price": f"{_redondear_tick(leg['price'], float(leg.get('tick_size') or 0)):.4f}"}
            res = _binance_request("POST", "order", params)
            enviadas.append(res)
    except Exception as e:
        for o in enviadas:
            try:
                _binance_request("DELETE", "order", {"symbol": o["symbol"], "orderId": o["orderId"]})
            except Exception:
                pass
        raise RuntimeError(f"pata rechazada por Binance ({e})") from e
    return {"venue": "binance_options_testnet", "order_ids": [str(o.get("orderId")) for o in enviadas],
            "detalle": f"{len(enviadas)} órdenes LIMIT enviadas al testnet de opciones de Binance", "estado": "enviada"}


def posiciones_binance() -> list[dict[str, Any]]:
    """Posiciones abiertas en el testnet de opciones (para reconciliar)."""
    if not binance_disponible():
        return []
    try:
        return _binance_request("GET", "position", {})
    except Exception as e:
        log_app.warning("Posiciones del testnet de opciones no disponibles: %s", e)
        return []


# ---------- Alpaca Paper ----------

def ordenar_alpaca(estrategia: dict[str, Any]) -> dict[str, Any]:
    """Orden multipata en Alpaca Paper con precio límite neto (débito/crédito de la estructura).

    Respeta el horario de mercado según `config.ejecucion.mercado_cerrado` (`cola` | `cancelar`).
    """
    abierto = alpaca.mercado_abierto()
    politica = str(cfg("ejecucion", "mercado_cerrado", default="cola"))
    if abierto is False and politica == "cancelar":
        raise RuntimeError("mercado de EE. UU. cerrado y la política es cancelar")
    # Alpaca exige ratios enteros: dividimos entre el máximo común divisor de las cantidades.
    import math

    qtys = [int(round(l["qty"])) for l in estrategia["legs"]]
    if any(q <= 0 for q in qtys):
        raise RuntimeError("cantidades no enteras para Alpaca")
    g = math.gcd(*qtys)
    legs = [{"symbol": l["symbol"], "side": l["side"], "ratio": q // g} for l, q in zip(estrategia["legs"], qtys)]
    limite = sum(l["side"] * (q // g) * l["price"] for l, q in zip(estrategia["legs"], qtys))
    res = alpaca.enviar_orden_multipata(legs, g, limite)
    ids = [str(res.get("id"))] + [str(x.get("id")) for x in res.get("legs", []) or []]
    estado = "en_cola" if abierto is False else str(res.get("status", "enviada"))
    return {"venue": "alpaca_paper", "order_ids": ids, "estado": estado,
            "detalle": f"Orden {'multipata ' if len(legs) > 1 else ''}en Alpaca Paper (límite neto {limite:+.2f})" + (" · mercado cerrado: queda en cola hasta la apertura" if abierto is False else "")}


# ---------- Venue previsto (sin enviar nada) ----------

def filtrar_cadena_deribit(cadena: Any, activo: str) -> tuple[Any, int]:
    """Deja en la cadena de Binance solo los contratos que también existen en el testnet de Deribit.

    Deribit es el único venue cripto con contraparte que llena órdenes, así que conviene que las candidatas se
    armen con contratos que pueda ejecutar. Devuelve la cadena original si Deribit no aplica o no responde.

    Returns:
        (cadena filtrada o la original, número de contratos que existen en Deribit; −1 si no se filtró).
    """
    if venue_cripto() != "deribit_testnet" or str(activo).upper() not in deribit.MONEDAS or cadena is None or len(cadena) == 0:
        return cadena, -1
    try:
        idx = deribit.instrumentos(activo)
    except Exception as e:
        log_app.warning("No se pudieron leer los instrumentos de Deribit: %s", e)
        return cadena, -1
    claves = cadena["expiry"].astype(str) + "|" + cadena["strike"].astype(float).map(lambda k: f"{k:g}") + "|" + cadena["type"].astype(str)
    mascara = claves.isin(set(idx))
    return cadena[mascara], int(mascara.sum())


def venue_previsto(estrategia: dict[str, Any], venue_objetivo: str) -> dict[str, str]:
    """A qué venue irá la estrategia al confirmar, revisando el mapeo de cada pata sin enviar órdenes.

    Returns:
        `{"venue", "texto"}` con una explicación corta para mostrar antes de confirmar.
    """
    if str(cfg("ejecucion", "forzar_paper", default=False)).lower() == "true":
        return {"venue": "paper", "texto": "Paper interno: la ejecución está forzada a paper en config.yaml."}
    if venue_objetivo not in ("binance_options_testnet", "deribit_testnet", "cripto"):
        return {"venue": venue_objetivo, "texto": ""}
    activo = str(estrategia.get("activo", "")).upper()
    motivo = ""
    if venue_cripto() == "deribit_testnet" and deribit.disponible():
        if activo in deribit.MONEDAS:
            try:
                for leg in estrategia["legs"]:
                    deribit.mapear_pata(activo, leg)
                return {"venue": "deribit_testnet", "texto": "Todas las patas existen en el testnet de Deribit."}
            except Exception as e:
                motivo = str(e)
        else:
            motivo = f"Deribit testnet no tiene opciones de {activo}"
    if gate_respaldo_activo():
        try:
            for leg in estrategia["legs"]:
                gate_options.mapear_pata(activo, leg)
            minutos = int(cfg("ejecucion", "gate_minutos_espera", default=10))
            return {"venue": "gate_options_testnet",
                    "texto": f"{motivo}. Irá al testnet de Gate.io; su libro no tiene contraparte, así que si nadie la llena en {minutos} min pasará a paper interno."}
        except Exception as e:
            motivo = f"{motivo}; Gate.io: {e}" if motivo else str(e)
    return {"venue": "paper", "texto": f"Irá a paper interno: {motivo}."}


# ---------- Enrutador ----------

def ejecutar(estrategia: dict[str, Any], venue_objetivo: str, user_id: str | None = None) -> dict[str, Any]:
    """Enruta la estrategia al venue; ante cualquier fallo cae a paper y lo explica en `sistema`."""
    if str(cfg("ejecucion", "forzar_paper", default=False)).lower() == "true":
        return {"venue": "paper", "order_ids": [], "estado": "simulada", "detalle": "Paper interno (forzado en config.yaml)"}
    if venue_objetivo in ("binance_options_testnet", "deribit_testnet", "cripto"):
        venue_objetivo = venue_cripto()
    try:
        if venue_objetivo == "deribit_testnet":
            if not deribit.disponible():
                raise RuntimeError("faltan DERIBIT_TESTNET_CLIENT_ID/SECRET")
            return ordenar_deribit(estrategia)
        if venue_objetivo == "binance_options_testnet":
            if not binance_disponible():
                raise RuntimeError("faltan BINANCE_FUTURES_TESTNET_API_KEY/SECRET")
            return ordenar_binance(estrategia)
        if venue_objetivo == "alpaca_paper":
            if not alpaca.disponible():
                raise RuntimeError("faltan ALPACA_API_KEY/ALPACA_SECRET_KEY")
            return ordenar_alpaca(estrategia)
        raise RuntimeError(f"venue desconocido {venue_objetivo}")
    except Exception as e:
        motivo = str(e)[:300]
        if venue_objetivo in ("deribit_testnet", "binance_options_testnet") and gate_respaldo_activo():
            try:
                res = ordenar_gate(estrategia)
                emit("sistema", "info", f"Orden de {estrategia['nombre']} enviada al respaldo Gate.io testnet",
                     {"venue_intentado": venue_objetivo, "motivo": motivo, "activo": estrategia.get("activo", ""), "ordenes": res["order_ids"]}, user_id)
                return res
            except Exception as e_gate:
                motivo = f"{motivo} · respaldo Gate.io: {str(e_gate)[:200]}"
        log_app.warning("Ejecución en %s falló, respaldo paper: %s", venue_objetivo, motivo)
        emit("sistema", "warning", f"Orden de {estrategia['nombre']} cayó a paper interno",
             {"venue_intentado": venue_objetivo, "motivo": motivo, "activo": estrategia.get("activo", "")}, user_id)
        return {"venue": "paper", "order_ids": [], "estado": "simulada", "detalle": f"Paper interno (respaldo): {motivo}"}
