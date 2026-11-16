"""Posiciones de opciones en paper trading: apertura, valuación, alertas e historial.

Lo usan la app (pestaña Mis posiciones) y el bot (refresco periódico de P&L y griegas).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core import db
from core.config import cfg
from core.events import emit, log_app
from data_sources.market import cadena_para, es_cripto, normalizar_activo
from quant.strategies_options import griegas_posicion, valor_actual


def abrir_posicion(user_id: str | None, recommendation_id: int | None, activo: str, estrategia: dict[str, Any], ejecucion: dict[str, Any] | None = None) -> int:
    """Crea una posición en `option_positions` con los precios reales de ejecución (ask/bid) y el venue usado.

    Args:
        ejecucion: Resultado de `core.execution.ejecutar` (`venue`, `order_ids`, `detalle`, `estado`); None = paper.

    Returns:
        Id de la posición.
    """
    legs = estrategia["legs"]
    ahora = db.ahora_iso()
    ej = ejecucion or {"venue": "paper", "order_ids": [], "detalle": "Paper interno", "estado": "simulada"}
    pos_id = db.ejecutar(
        "INSERT INTO option_positions(user_id, recommendation_id, asset, legs_json, entry_ts, entry_cost, status, last_value, greeks_json, last_update_ts, venue, venue_order_ids_json) VALUES (?,?,?,?,?,?,'abierta',?,?,?,?,?)",
        (user_id, recommendation_id, normalizar_activo(activo), db.a_json({"legs": legs, "multiplicador": estrategia["multiplicador"], "nombre": estrategia["nombre"], "tipo": estrategia["tipo"],
                                                                        "max_ganancia": estrategia["max_ganancia"], "max_perdida": estrategia["max_perdida"], "ganancia_ilimitada": estrategia.get("ganancia_ilimitada", False),
                                                                        "breakevens": estrategia["breakevens"], "spot_entrada": estrategia["spot"], "expiry": estrategia["expiry"]}),
         ahora, float(estrategia["costo_neto"]), float(estrategia["costo_neto"]), db.a_json(estrategia.get("griegas", {})), ahora, ej["venue"], db.a_json({"order_ids": ej.get("order_ids", []), "estado": ej.get("estado"), "detalle": ej.get("detalle")})),
    )
    emit("alertas", "ok", f"Posición abierta: {estrategia['nombre']} en {activo} [{ej['venue']}]",
         {"costo_neto": round(estrategia["costo_neto"], 2), "max_perdida": round(estrategia["max_perdida"], 2), "vencimiento": estrategia["expiry"], "venue": ej["venue"], "ordenes": ej.get("order_ids", []), "detalle": ej.get("detalle")}, user_id)
    return pos_id


def listar(user_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """Posiciones (todas o del usuario), con `legs` ya deserializadas."""
    sql, params = "SELECT * FROM option_positions WHERE 1=1", []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    filas = db.consultar(sql + " ORDER BY id DESC", tuple(params))
    for f in filas:
        d = db.de_json(f.pop("legs_json"), {})
        f.update({k: v for k, v in d.items()})
        f["greeks"] = db.de_json(f.pop("greeks_json"), {})
        f["ejecucion"] = db.de_json(f.pop("venue_order_ids_json", None), {})
        f["dias_transcurridos"] = _dias_desde(f["entry_ts"])
    return filas


def _dias_desde(ts: str) -> float:
    """Días transcurridos desde un ISO timestamp."""
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 86400
    except ValueError:
        return 0.0


def _precio_liquidacion(pos: dict[str, Any], cadena, S: float, r: float) -> float:
    """Valor de cierre con precios reales de la cadena (bid para lo que se vende, ask para recomprar); si falta, Black-Scholes."""
    total = 0.0
    for leg in pos["legs"]:
        fila = None
        if cadena is not None and not cadena.empty:
            m = cadena[(cadena["expiry"] == leg["expiry"]) & (cadena["strike"] == leg["strike"]) & (cadena["type"] == leg["type"])]
            fila = m.iloc[0] if len(m) else None
        if fila is not None and fila["bid"] > 0 and fila["ask"] > 0:
            px = float(fila["bid"]) if leg["side"] > 0 else float(fila["ask"])
            total += leg["side"] * leg["qty"] * pos["multiplicador"] * px
        else:
            total += valor_actual([leg], S, r, pos["dias_transcurridos"], 0.0, pos["multiplicador"])
    return total


def valuar(pos: dict[str, Any], cadena, S: float, r: float) -> dict[str, Any]:
    """Valor actual, P&L y griegas de una posición."""
    valor = _precio_liquidacion(pos, cadena, S, r)
    g = griegas_posicion(pos["legs"], S, r, pos["multiplicador"], pos["dias_transcurridos"])
    dte_rest = max(pos["legs"][0]["dte"] - pos["dias_transcurridos"], 0.0)
    return {"valor": valor, "pnl": valor - pos["entry_cost"], "griegas": g, "delta_dolares": g["delta"] * S, "dte_restante": dte_rest, "spot": S}


def cerrar_posicion(pos_id: int, valor_salida: float, user_id: str | None = None, razon: str = "manual") -> None:
    """Cierra la posición y registra P&L."""
    pos = db.consultar_uno("SELECT * FROM option_positions WHERE id = ?", (pos_id,))
    if not pos or pos["status"] != "abierta":
        return
    pnl = valor_salida - pos["entry_cost"]
    db.ejecutar("UPDATE option_positions SET status='cerrada', exit_ts=?, exit_value=?, pnl=? WHERE id=?", (db.ahora_iso(), valor_salida, pnl, pos_id))
    emit("alertas", "ok" if pnl >= 0 else "rojo", f"Posición #{pos_id} cerrada ({razon}): P&L {pnl:+,.2f} USD", {"asset": pos["asset"], "valor_salida": round(valor_salida, 2)}, user_id)


def refrescar_todas() -> list[dict[str, Any]]:
    """Revalúa todas las posiciones abiertas, guarda P&L/griegas y emite alertas de cobertura."""
    r = float(cfg("app", "tasa_libre_riesgo", default=0.045))
    c = cfg("posiciones") or {}
    abiertas = listar(status="abierta")
    cadenas: dict[str, tuple[Any, float]] = {}
    resultados = []
    for pos in abiertas:
        a = pos["asset"]
        if a not in cadenas:
            cad, S, _ = cadena_para(a)
            cadenas[a] = (cad, S)
        cad, S = cadenas[a]
        if S != S:
            continue
        v = valuar(pos, cad, S, r)
        db.ejecutar("UPDATE option_positions SET last_value=?, greeks_json=?, last_update_ts=? WHERE id=?", (v["valor"], db.a_json(v["griegas"]), db.ahora_iso(), pos["id"]))
        _alertas(pos, v, c)
        if v["dte_restante"] <= 0:
            cerrar_posicion(pos["id"], v["valor"], pos.get("user_id"), "vencimiento")
        resultados.append({"id": pos["id"], **v})
    return resultados


def _alertas(pos: dict[str, Any], v: dict[str, Any], c: dict[str, Any]) -> None:
    """Emite alertas de delta desviada, 50 % de ganancia máxima y vencimiento cercano (una vez por tipo y día)."""
    hoy = datetime.now(timezone.utc).date().isoformat()
    def ya_emitida(tipo: str) -> bool:
        return bool(db.consultar_uno("SELECT id FROM events WHERE channel='alertas' AND title LIKE ? AND ts >= ?", (f"%#{pos['id']}%{tipo}%", hoy)))
    uid = pos.get("user_id")
    if abs(v["delta_dolares"]) > float(c.get("umbral_delta_dolares", 500)) and not ya_emitida("delta"):
        emit("alertas", "warning", f"Posición #{pos['id']} {pos['asset']}: delta desviada ({v['delta_dolares']:+,.0f} USD)",
             {"sugerencia": f"Cubrir vendiendo/comprando {abs(v['griegas']['delta']):.3f} unidades del subyacente", "delta": round(v["griegas"]["delta"], 4)}, uid)
    if not pos.get("ganancia_ilimitada") and pos.get("max_ganancia", 0) > 0 and v["pnl"] >= float(c.get("pct_toma_ganancia", 0.5)) * pos["max_ganancia"] and not ya_emitida("ganancia"):
        emit("alertas", "ok", f"Posición #{pos['id']} {pos['asset']}: 50% de la ganancia máxima alcanzada", {"pnl": round(v["pnl"], 2), "max_ganancia": round(pos["max_ganancia"], 2), "sugerencia": "Considerar tomar ganancia"}, uid)
    if 0 < v["dte_restante"] <= float(c.get("dias_vencimiento_alerta", 3)) and not ya_emitida("vencimiento"):
        emit("alertas", "warning", f"Posición #{pos['id']} {pos['asset']}: vencimiento en {v['dte_restante']:.1f} días", {"pnl": round(v["pnl"], 2)}, uid)


def historial_copiloto(user_id: str | None = None) -> dict[str, Any]:
    """Recomendación vs resultado real: tasa de acierto sobre posiciones cerradas."""
    cerradas = [p for p in listar(user_id, "cerrada")]
    abiertas = [p for p in listar(user_id, "abierta")]
    aciertos = sum(1 for p in cerradas if (p.get("pnl") or 0) > 0)
    return {"cerradas": len(cerradas), "abiertas": len(abiertas), "aciertos": aciertos, "tasa_acierto": aciertos / len(cerradas) if cerradas else None,
            "pnl_realizado": sum(p.get("pnl") or 0 for p in cerradas), "pnl_abierto": sum((p.get("last_value") or p["entry_cost"]) - p["entry_cost"] for p in abiertas)}
