"""Cobertura de delta de las posiciones de opciones del copiloto con el subyacente.

Cuando la delta de una posición se desvía, la alerta sugiere comprar o vender unidades del subyacente.
Este módulo calcula esa sugerencia y ejecuta la orden desde la app:

- BTC y ETH: perpetuo inverso en el testnet de Deribit (`BTC-PERPETUAL`, `ETH-PERPETUAL`).
- Acciones y ETFs de EE. UU.: acciones en Alpaca Paper. La cuenta guarda una sola posición neta por
  símbolo, así que la orden mueve la cuenta a la suma de las posiciones del bot más las coberturas
  (`bot.executor.mover_cuenta_alpaca`), y el bot cuenta las coberturas al netear sus propias órdenes.
- Lo demás (otras cripto, o si `ejecucion.forzar_paper`): registro en paper interno al precio actual.

Si el venue real rechaza la orden, no se cae a paper en silencio: se devuelve el error a la app.
La cobertura se guarda en `option_hedges` y su delta se suma a la de la posición.
"""
from __future__ import annotations

from typing import Any

from core import db
from core.config import cfg
from core.events import emit, log_app
from data_sources.market import es_cripto

NOMBRE_VENUE = {"deribit_testnet": "Deribit testnet", "alpaca_paper": "Alpaca Paper", "paper": "Paper interno"}


def _forzar_paper() -> bool:
    """True si la ejecución del copiloto está forzada a paper en `config.yaml`."""
    return str(cfg("ejecucion", "forzar_paper", default=False)).lower() == "true"


def abiertas(pos_id: int | None = None) -> list[dict[str, Any]]:
    """Coberturas abiertas (de una posición o de todas)."""
    if pos_id is None:
        return db.consultar("SELECT * FROM option_hedges WHERE status='abierta' ORDER BY id")
    return db.consultar("SELECT * FROM option_hedges WHERE status='abierta' AND position_id=? ORDER BY id", (pos_id,))


def delta_cobertura(pos_id: int) -> float:
    """Unidades del subyacente que suman las coberturas abiertas de la posición (con signo)."""
    return float(sum(float(h["qty"]) for h in abiertas(pos_id)))


def pnl(cobertura: dict[str, Any], S: float) -> float:
    """P&L en USD de una cobertura al precio `S`: unidades × (S − precio de entrada).

    Para el perpetuo inverso de Deribit es exacto en dólares: un monto A comprado a P0 gana A·(S/P0 − 1) = (A/P0)·(S − P0).
    """
    return float(cobertura["qty"]) * (float(S) - float(cobertura["entry_price"]))


def destino(asset: str) -> dict[str, Any]:
    """Dónde se cubre un activo: venue, instrumento y si se opera en unidades enteras."""
    a = asset.upper()
    if _forzar_paper():
        return {"venue": "paper", "instrumento": a, "enteras": not es_cripto(a)}
    if a in ("BTC", "ETH"):
        from data_sources import deribit

        if deribit.disponible():
            return {"venue": "deribit_testnet", "instrumento": f"{a}-PERPETUAL", "enteras": False}
    elif not es_cripto(a):
        from data_sources import alpaca

        if alpaca.disponible():
            return {"venue": "alpaca_paper", "instrumento": a, "enteras": True}
    return {"venue": "paper", "instrumento": a, "enteras": not es_cripto(a)}


def redondear(qty: float, dest: dict[str, Any], S: float) -> float:
    """Redondea la cantidad a lo que el venue acepta: acciones enteras, contratos de 10 USD (BTC) o 1 USD (ETH) en Deribit."""
    if dest["enteras"]:
        return float(round(qty))
    if dest["venue"] == "deribit_testnet":
        contrato = 10.0 if dest["instrumento"].startswith("BTC") else 1.0
        return round(qty * S / contrato) * contrato / S
    return round(qty, 4)


def sugerencia(pos: dict[str, Any], delta_opciones: float, S: float) -> dict[str, Any]:
    """Orden que deja la posición con delta neta ≈ 0.

    Args:
        pos: Posición (necesita `id` y `asset`).
        delta_opciones: Delta de las opciones en unidades del subyacente (ya multiplicada por qty y multiplicador).
        S: Precio actual del subyacente.

    Returns:
        `{qty, lado, delta_opciones, delta_cobertura, delta_neta, destino, texto}`; `qty` con signo (+ compra, − vende),
        0 si la delta ya está cubierta dentro de lo que permite el tamaño mínimo.
    """
    dc = delta_cobertura(pos["id"])
    neta = delta_opciones + dc
    dest = destino(pos["asset"])
    qty = redondear(-neta, dest, S)
    lado = "comprar" if qty > 0 else "vender"
    unidad = "acciones" if dest["enteras"] else pos["asset"]
    texto = (f"{lado.capitalize()} {abs(qty):,.4g} {unidad} ({abs(qty) * S:,.0f} USD) en {NOMBRE_VENUE[dest['venue']]} para dejar la delta neta en cero"
             if qty else "La delta ya está cubierta (lo que falta es menor que el tamaño mínimo de orden)")
    return {"qty": qty, "lado": lado, "delta_opciones": delta_opciones, "delta_cobertura": dc, "delta_neta": neta, "destino": dest, "texto": texto}


def _precio_actual(asset: str) -> float:
    """Precio actual del subyacente (el mismo que usa la valuación de posiciones)."""
    from data_sources.market import cadena_para

    _, S, _ = cadena_para(asset)
    if not S or S != S:
        raise RuntimeError(f"sin precio actual para {asset}")
    return float(S)


def _ejecutar(dest: dict[str, Any], asset: str, qty: float, S: float, cierre: dict[str, Any] | None = None) -> dict[str, Any]:
    """Envía la orden al venue y devuelve `{precio, qty, order_ids, extra}`; lanza si el venue la rechaza.

    Args:
        cierre: Al cerrar, la cobertura original (para deshacer exactamente el mismo monto en Deribit).
    """
    if dest["venue"] == "deribit_testnet":
        from data_sources import deribit

        lado = "buy" if qty > 0 else "sell"
        if cierre:
            monto = float((db.de_json(cierre.get("order_ids_json"), {}) or {}).get("monto_usd") or 0)
            r = deribit.ordenar_perpetuo(asset, lado, monto_usd=monto, reduce_only=True)
        else:
            r = deribit.ordenar_perpetuo(asset, lado, qty_base=qty)
        return {"precio": r["precio"], "qty": r["qty_base"], "order_ids": [r["order_id"]], "extra": {"monto_usd": r["monto_usd"]}}
    if dest["venue"] == "alpaca_paper":
        from bot.executor import mover_cuenta_alpaca, neta_alpaca_simbolo

        objetivo = neta_alpaca_simbolo(asset) + qty
        r = mover_cuenta_alpaca(asset, objetivo, S)
        return {"precio": r["precio"], "qty": qty, "order_ids": r["ids"] or ["neteada"], "extra": {"neto_antes": r["actual"], "neto_despues": objetivo}}
    return {"precio": S, "qty": qty, "order_ids": [], "extra": {}}


def abrir(pos_id: int, qty: float, user_id: str | None = None) -> dict[str, Any]:
    """Abre una cobertura de `qty` unidades del subyacente (+ compra, − vende) para la posición.

    Returns:
        La fila guardada en `option_hedges` más `detalle`.

    Raises:
        ValueError: si la posición no existe o la cantidad redondeada es cero.
        RuntimeError: si el venue rechaza la orden (no se registra nada).
    """
    pos = db.consultar_uno("SELECT id, asset, status FROM option_positions WHERE id=?", (pos_id,))
    if not pos or pos["status"] != "abierta":
        raise ValueError(f"la posición #{pos_id} no está abierta")
    asset = pos["asset"]
    S = _precio_actual(asset)
    dest = destino(asset)
    qty = redondear(float(qty), dest, S)
    if qty == 0:
        raise ValueError("la cantidad es menor que el tamaño mínimo de orden")
    r = _ejecutar(dest, asset, qty, S)
    detalle = f"{'Compra' if r['qty'] > 0 else 'Venta'} de {abs(r['qty']):,.4g} {dest['instrumento']} en {NOMBRE_VENUE[dest['venue']]} a {r['precio']:,.2f}"
    hid = db.ejecutar("INSERT INTO option_hedges(position_id, user_id, ts, venue, instrumento, qty, entry_price, order_ids_json, status, detalle) VALUES (?,?,?,?,?,?,?,?,'abierta',?)",
                      (pos_id, user_id, db.ahora_iso(), dest["venue"], dest["instrumento"],
                       float(r["qty"]), float(r["precio"]), db.a_json({"order_ids": r["order_ids"], **r["extra"]}), detalle))
    emit("alertas", "ok", f"Posición #{pos_id} {asset}: cobertura de delta abierta",
         {"orden": detalle, "venue": dest["venue"], "unidades": round(r["qty"], 6), "ordenes": r["order_ids"]}, user_id)
    return {**db.consultar_uno("SELECT * FROM option_hedges WHERE id=?", (hid,)), "detalle": detalle}


def cerrar(hedge_id: int, user_id: str | None = None, razon: str = "manual") -> dict[str, Any]:
    """Cierra una cobertura con la orden contraria en el mismo venue y registra su P&L.

    Raises:
        RuntimeError: si el venue rechaza la orden; la cobertura sigue abierta.
    """
    h = db.consultar_uno("SELECT * FROM option_hedges WHERE id=?", (hedge_id,))
    if not h or h["status"] != "abierta":
        raise ValueError(f"la cobertura #{hedge_id} no está abierta")
    pos = db.consultar_uno("SELECT asset FROM option_positions WHERE id=?", (h["position_id"],))
    asset = pos["asset"] if pos else h["instrumento"].split("-")[0]
    S = _precio_actual(asset)
    dest = {"venue": h["venue"], "instrumento": h["instrumento"], "enteras": h["venue"] == "alpaca_paper"}
    r = _ejecutar(dest, asset, -float(h["qty"]), S, cierre=h)
    salida = float(r["precio"])
    ganancia = float(h["qty"]) * (salida - float(h["entry_price"]))
    previo = db.de_json(h.get("order_ids_json"), {}) or {}
    previo["cierre"] = {"order_ids": r["order_ids"], **r["extra"]}
    db.ejecutar("UPDATE option_hedges SET status='cerrada', exit_ts=?, exit_price=?, pnl=?, order_ids_json=? WHERE id=?",
                (db.ahora_iso(), salida, ganancia, db.a_json(previo), hedge_id))
    emit("alertas", "ok" if ganancia >= 0 else "rojo", f"Posición #{h['position_id']} {asset}: cobertura cerrada ({razon}), P&L {ganancia:+,.2f} USD",
         {"venue": h["venue"], "unidades": round(float(h["qty"]), 6), "entrada": round(float(h["entry_price"]), 2), "salida": round(salida, 2)}, user_id)
    return {"id": hedge_id, "pnl": ganancia, "precio": salida}


def cerrar_de_posicion(pos_id: int, user_id: str | None = None, razon: str = "posición cerrada") -> list[dict[str, Any]]:
    """Cierra todas las coberturas abiertas de una posición; si una falla lo avisa en `sistema` y sigue con las demás."""
    hechas = []
    for h in abiertas(pos_id):
        try:
            hechas.append(cerrar(int(h["id"]), user_id, razon))
        except Exception as e:
            log_app.warning("No se pudo cerrar la cobertura #%s: %s", h["id"], e)
            emit("sistema", "warning", f"Cobertura #{h['id']} de la posición #{pos_id} sigue abierta",
                 {"motivo": str(e), "venue": h["venue"], "accion": "Ciérrala desde Mis posiciones cuando el venue lo permita"}, user_id)
    return hechas
