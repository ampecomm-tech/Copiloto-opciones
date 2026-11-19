"""Loop principal del bot (proceso systemd): corre 24/7 sobre los activos activos en la DB.

Cada ciclo:
1. Lee `bot_assets` y `bot_strategy_state` (el usuario activa/pausa desde la app, sin reinicio).
2. Por activo descarga velas; por estrategia activa genera la señal y la publica en `bot_senales`.
3. Gestiona posiciones: SL/TP por ATR, kill-switch diario, entradas/salidas en `bot_operaciones`.
4. Refresca P&L y griegas de las posiciones de opciones y emite alertas.
"""
from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from bot import risk
from bot.executor import _forzar_paper, ejecutar_neta_alpaca, ejecutar_orden, neta_de_estrategias, reconciliar
from bot.features import atr
from bot.strategies import crear
from core import db
from core.config import cfg
from core.events import emit, log_bot
from data_sources import alpaca
from data_sources.market import velas

_corriendo = True


def _parar(*_: Any) -> None:
    """Manejador de SIGTERM/SIGINT para salir limpiamente."""
    global _corriendo
    _corriendo = False


def registrar_activo(symbol: str, display_name: str, market_type: str, venue: str) -> int:
    """Crea (o recupera) el activo en `bot_assets` y su estado por estrategia con capital inicial."""
    fila = db.consultar_uno("SELECT id FROM bot_assets WHERE symbol = ?", (symbol,))
    if fila:
        asset_id = int(fila["id"])
        db.ejecutar("UPDATE bot_assets SET active = 1 WHERE id = ?", (asset_id,))
    else:
        asset_id = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts) VALUES (?,?,?,?,1,?)",
                               (symbol, display_name, market_type, venue, db.ahora_iso()))
    capital = float(cfg("bot", "capital_inicial_por_estrategia", default=1000))
    for est in cfg("bot", "estrategias_por_defecto", default=[]) or []:
        db.ejecutar("INSERT OR IGNORE INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,?,1,?,NULL,0,?)",
                    (asset_id, est, capital, datetime.now(timezone.utc).date().isoformat()))
    emit("sistema", "ok", f"Activo agregado al bot: {display_name}", {"symbol": symbol, "venue": venue, "market_type": market_type})
    return asset_id


def _explicar_async(titulo: str, datos: dict[str, Any], user_id: str | None = None) -> None:
    """Pide a Gemini una explicación en lenguaje simple de la operación y la emite a `bot_operaciones` (en hilo aparte)."""
    from agents import gemini_client as gem

    if not gem.disponible():
        return

    def _trabajo() -> None:
        texto = gem.explicar(f"esta operación del bot de trading: {titulo}", datos, ttl_cache_s=0)
        if texto:
            emit("bot_operaciones", "info", f"💬 {titulo}", {"explicacion": texto}, user_id)

    import threading

    threading.Thread(target=_trabajo, daemon=True).start()


def _sentimiento(symbol: str) -> float | None:
    """Último puntaje de sentimiento para el activo (BTC/USDT → BTC)."""
    from agents.sentiment import ultimo_score

    base = symbol.split("/")[0].replace("-USD", "")
    return ultimo_score(base)


def venue_de_cierre(asset: dict[str, Any], pos: dict[str, Any]) -> str:
    """Venue donde debe cerrarse la posición: el mismo donde se abrió.

    Si la apertura cayó a paper interno, el cierre también va a paper; así el broker nunca recibe
    el cierre de una posición que no tiene. Para posiciones anteriores a este campo se usa el id de
    orden guardado: sin id, la apertura fue en paper.
    """
    modo = pos.get("modo")
    if modo is None:
        fila = db.consultar_uno("SELECT venue_order_id FROM bot_trades WHERE id = ?", (pos.get("trade_id"),))
        modo = asset["venue"] if fila and fila.get("venue_order_id") else "paper"
    return "paper" if modo == "paper" else asset["venue"]


def _cerrar(asset: dict[str, Any], estado: dict[str, Any], pos: dict[str, Any], precio: float, razon: str) -> None:
    """Cierra la posición abierta en el venue donde se abrió, registra el trade y actualiza capital y P&L diario."""
    lado_orden = "sell" if pos["side"] == "long" else "buy"
    venue = venue_de_cierre(asset, pos)
    if venue == "alpaca_paper":
        objetivo = neta_de_estrategias(asset["id"], venue, excluir=estado["strategy"])
        orden = ejecutar_neta_alpaca(asset, estado["strategy"], objetivo, pos["qty"], lado_orden, precio)
    else:
        orden = ejecutar_orden(asset["symbol"], venue, lado_orden, pos["qty"], precio)
    px = orden["precio"]
    pnl = (px - pos["entry_price"]) * pos["qty"] if pos["side"] == "long" else (pos["entry_price"] - px) * pos["qty"]
    nuevo_capital = estado["capital"] + pnl
    db.ejecutar("UPDATE bot_trades SET exit_price=?, exit_ts=?, pnl=?, exit_reason=? WHERE id=?", (px, db.ahora_iso(), pnl, razon, pos["trade_id"]))
    db.ejecutar("UPDATE bot_strategy_state SET capital=?, position_json=NULL, daily_pnl=daily_pnl+? WHERE asset_id=? AND strategy=?",
                (nuevo_capital, pnl, asset["id"], estado["strategy"]))
    datos = {"precio_entrada": pos["entry_price"], "precio_salida": px, "qty": pos["qty"], "pnl": round(pnl, 2), "capital": round(nuevo_capital, 2), "modo": orden["modo"]}
    titulo = f"Cierre {pos['side']} {asset['symbol']} [{estado['strategy']}] por {razon}: P&L {pnl:+.2f} USD"
    emit("bot_operaciones", "entrada" if pnl >= 0 else "salida", titulo, datos)
    _explicar_async(titulo, {**datos, "estrategia": estado["strategy"], "razon_cierre": razon})


def _actualizar_stop(asset: dict[str, Any], estado: dict[str, Any], pos: dict[str, Any], alto: float, bajo: float, atr_valor: float, ts_vela: str) -> dict[str, Any]:
    """Aplica el trailing con la vela cerrada, marca la vela como revisada y guarda la posición."""
    nueva = risk.actualizar_trailing(pos, alto, bajo, atr_valor)
    nueva["vela_revisada"] = ts_vela
    db.ejecutar("UPDATE bot_strategy_state SET position_json=? WHERE asset_id=? AND strategy=?", (db.a_json(nueva), asset["id"], estado["strategy"]))
    if nueva["sl"] != pos["sl"]:
        db.ejecutar("UPDATE bot_trades SET sl=? WHERE id=?", (nueva["sl"], pos["trade_id"]))
        log_bot.info("Trailing %s [%s]: stop %.4f -> %.4f", asset["symbol"], estado["strategy"], pos["sl"], nueva["sl"])
    if nueva.get("trailing_activo") and not pos.get("trailing_activo"):
        emit("bot_operaciones", "info", f"Trailing activado en {asset['symbol']} [{estado['strategy']}]: stop en {nueva['sl']:,.4f}",
             {"lado": pos["side"], "precio_entrada": pos["entry_price"], "stop_anterior": round(pos["sl"], 4), "stop_nuevo": round(nueva["sl"], 4), "mejor_precio": round(nueva["extremo"], 4)})
    return nueva


def _abrir(asset: dict[str, Any], estado: dict[str, Any], lado: str, precio: float, atr_valor: float, detalles: dict[str, Any], ts_vela: str) -> None:
    """Abre una posición con tamaño por volatilidad y SL/TP por ATR; la vela de entrada queda marcada como revisada."""
    paso = 0.001 if asset["market_type"] == "cripto" else 1.0 if asset["venue"] == "paper" and asset["market_type"] != "cripto" else 1e-4
    qty = risk.tamano_posicion(estado["capital"], precio, atr_valor, paso=paso)
    if asset["venue"] == "alpaca_paper" and lado == "short":
        qty = float(int(qty))  # Alpaca solo acepta cortos en acciones enteras
    if qty <= 0:
        return
    if asset["venue"] == "alpaca_paper":
        objetivo = neta_de_estrategias(asset["id"], "alpaca_paper", excluir=estado["strategy"]) + (qty if lado == "long" else -qty)
        orden = ejecutar_neta_alpaca(asset, estado["strategy"], objetivo, qty, "buy" if lado == "long" else "sell", precio)
    else:
        orden = ejecutar_orden(asset["symbol"], asset["venue"], "buy" if lado == "long" else "sell", qty, precio, abre_corto=lado == "short")
    px = orden["precio"]
    sl, tp = risk.niveles_sl_tp(px, atr_valor, lado)
    trade_id = db.ejecutar("INSERT INTO bot_trades(asset_id, strategy, side, qty, entry_price, sl, tp, entry_ts, venue_order_id) VALUES (?,?,?,?,?,?,?,?,?)",
                           (asset["id"], estado["strategy"], lado, orden["qty"], px, sl, tp, db.ahora_iso(), orden["venue_order_id"]))
    pos = {"side": lado, "qty": orden["qty"], "entry_price": px, "sl": sl, "tp": tp, "trade_id": trade_id, "entry_ts": db.ahora_iso(),
           "extremo": px, "trailing_activo": False, "vela_revisada": ts_vela, "modo": orden["modo"]}
    db.ejecutar("UPDATE bot_strategy_state SET position_json=? WHERE asset_id=? AND strategy=?", (db.a_json(pos), asset["id"], estado["strategy"]))
    datos = {"qty": orden["qty"], "sl": round(sl, 4), "tp": round(tp, 4), "atr": round(atr_valor, 4), "modo": orden["modo"], "razon": detalles.get("razon", "")}
    titulo = f"Apertura {lado} {asset['symbol']} [{estado['strategy']}] @ {px:,.4f}"
    emit("bot_operaciones", "entrada", titulo, datos)
    _explicar_async(titulo, {**datos, "estrategia": estado["strategy"], "indicadores": {k: v for k, v in detalles.items() if k not in ("razon", "por_estrategia")}})


def procesar_activo(asset: dict[str, Any]) -> None:
    """Procesa todas las estrategias activas de un activo sobre su última vela cerrada."""
    df = velas(asset["symbol"], asset["venue"])
    if len(df) < 60:
        log_bot.warning("Sin velas suficientes para %s", asset["symbol"])
        return
    df_cerradas = df.iloc[:-1]           # la última vela puede estar en formación
    ultima = df_cerradas.iloc[-1]
    ts_vela = str(df_cerradas.index[-1])
    precio = float(ultima["close"])
    atr_valor = float(atr(df_cerradas, int(cfg("bot", "riesgo", "atr_periodo", default=14))).iloc[-1])
    hoy = datetime.now(timezone.utc).date().isoformat()
    sentimiento = _sentimiento(asset["symbol"])
    solo_cripto = asset["market_type"] == "cripto"
    # Acciones con el mercado de EE. UU. cerrado: se registran señales, pero no se abre ni se cierra nada.
    # Antes esas órdenes caían a paper interno; ahora esperan a que abra el mercado.
    cerrado = asset["venue"] == "alpaca_paper" and not _forzar_paper() and alpaca.mercado_abierto() is False
    if cerrado:
        log_bot.info("Mercado de EE. UU. cerrado: %s solo registra señales hasta la apertura", asset["symbol"])

    for estado in db.consultar("SELECT * FROM bot_strategy_state WHERE asset_id = ? AND active = 1", (asset["id"],)):
        if estado.get("daily_date") != hoy:
            db.ejecutar("UPDATE bot_strategy_state SET daily_pnl = 0, daily_date = ? WHERE asset_id=? AND strategy=?", (hoy, asset["id"], estado["strategy"]))
            estado["daily_pnl"] = 0.0
        pos = db.de_json(estado.get("position_json"))

        # 1) Gestión de la posición abierta: cada vela cerrada se revisa una sola vez y nunca la vela de entrada.
        #    Primero se evalúa el stop vigente y después se mueve, porque dentro de la vela no se sabe qué ocurrió antes.
        if pos and pos.get("vela_revisada") != ts_vela and not cerrado:
            salida = risk.evaluar_salida(pos, float(ultima["high"]), float(ultima["low"]), precio)
            if salida:
                px = pos["tp"] if salida == "TP" else pos["sl"]
                _cerrar(asset, estado, pos, px, salida)
                pos = None
                estado = db.consultar_uno("SELECT * FROM bot_strategy_state WHERE asset_id=? AND strategy=?", (asset["id"], estado["strategy"])) or estado
            else:
                pos = _actualizar_stop(asset, estado, pos, float(ultima["high"]), float(ultima["low"]), atr_valor, ts_vela)

        # 2) Señal solo una vez por vela cerrada
        if estado.get("last_candle_ts") == ts_vela:
            continue
        db.ejecutar("UPDATE bot_strategy_state SET last_candle_ts=? WHERE asset_id=? AND strategy=?", (ts_vela, asset["id"], estado["strategy"]))
        try:
            s = crear(estado["strategy"], asset["id"], sentimiento).generate_signal(df_cerradas)
        except Exception as e:
            log_bot.error("Estrategia %s falló en %s: %s", estado["strategy"], asset["symbol"], e)
            continue
        db.ejecutar("INSERT INTO bot_signals(ts, asset_id, strategy, action, confidence, details_json) VALUES (?,?,?,?,?,?)",
                    (db.ahora_iso(), asset["id"], estado["strategy"], s.action, s.confidence, db.a_json(s.details)))
        if s.action != "HOLD" or estado["strategy"] == "random_forest":
            emit("bot_senales", "senal", f"{asset['symbol']} [{estado['strategy']}] → {s.action} ({s.confidence:.0%}) @ {precio:,.4f}",
                 {"vela": ts_vela, **{k: v for k, v in s.details.items() if k != "por_estrategia"}, **({"por_estrategia": s.details["por_estrategia"]} if "por_estrategia" in s.details else {})})

        if cerrado:
            continue

        # 3) Kill-switch diario
        if risk.kill_switch_activo(float(estado["capital"]) - float(estado["daily_pnl"]), float(estado["daily_pnl"])):
            if pos:
                _cerrar(asset, estado, pos, precio, "kill-switch")
            emit("alertas", "rojo", f"Kill-switch diario activado en {asset['symbol']} [{estado['strategy']}]", {"pnl_dia": round(float(estado["daily_pnl"]), 2)})
            continue

        # 4) Entradas y salidas por señal (solo largos en spot; cortos solo en paper)
        if s.action == "BUY" and not pos:
            _abrir(asset, estado, "long", precio, atr_valor, s.details, ts_vela)
        elif s.action == "SELL" and pos and pos["side"] == "long":
            _cerrar(asset, estado, pos, precio, "señal")
        elif s.action == "SELL" and not pos and asset["venue"] in ("paper", "alpaca_paper"):
            _abrir(asset, estado, "short", precio, atr_valor, s.details, ts_vela)
        elif s.action == "BUY" and pos and pos["side"] == "short":
            _cerrar(asset, estado, pos, precio, "señal")


def refrescar_posiciones_opciones() -> None:
    """Actualiza valor, P&L y griegas de las posiciones de opciones abiertas y emite alertas."""
    try:
        from core.positions import refrescar_todas

        refrescar_todas()
    except Exception as e:
        log_bot.error("No se pudieron refrescar las posiciones de opciones: %s", e)


def resumen_diario(forzar: bool = False) -> str | None:
    """Resumen diario con Gemini en `sistema`: desempeño del bot, posiciones abiertas y alertas del día.

    Se emite una vez al día a la hora UTC de `config.bot.resumen_diario_hora_utc`.
    """
    from agents import gemini_client as gem

    hoy = datetime.now(timezone.utc).date().isoformat()
    hora = int(cfg("bot", "resumen_diario_hora_utc", default=23))
    if not forzar:
        if datetime.now(timezone.utc).hour != hora:
            return None
        if db.consultar_uno("SELECT id FROM events WHERE channel='sistema' AND title LIKE 'Resumen diario%' AND ts >= ?", (hoy,)):
            return None
    trades = db.consultar("SELECT a.symbol, t.strategy, t.side, t.pnl, t.exit_reason FROM bot_trades t JOIN bot_assets a ON a.id=t.asset_id WHERE t.exit_ts >= ?", (hoy,))
    abiertas_bot = db.consultar("SELECT a.symbol, s.strategy, s.capital, s.position_json FROM bot_strategy_state s JOIN bot_assets a ON a.id=s.asset_id WHERE s.position_json IS NOT NULL")
    opciones = db.consultar("SELECT asset, entry_cost, last_value, venue FROM option_positions WHERE status='abierta'")
    alertas = db.consultar("SELECT title FROM events WHERE channel='alertas' AND ts >= ? ORDER BY id DESC LIMIT 15", (hoy,))
    senales = db.consultar_uno("SELECT COUNT(*) n FROM bot_signals WHERE ts >= ?", (hoy,))
    datos = {"fecha": hoy, "operaciones_cerradas_hoy": trades, "pnl_hoy": round(sum(t["pnl"] or 0 for t in trades), 2), "senales_hoy": (senales or {}).get("n", 0),
             "posiciones_bot_abiertas": [{"symbol": a["symbol"], "estrategia": a["strategy"], "capital": a["capital"]} for a in abiertas_bot],
             "posiciones_opciones": [{"activo": o["asset"], "costo": o["entry_cost"], "valor": o["last_value"], "venue": o["venue"]} for o in opciones],
             "alertas_hoy": [a["title"] for a in alertas]}
    texto = ""
    if gem.disponible():
        try:
            texto = gem.generar("Redacta el resumen diario (5-8 frases, español, tono claro, sin inventar) de este sistema de paper trading para un canal de Discord:\n" + db.a_json(datos),
                                system="Eres el asistente que reporta el estado diario de un bot de trading educativo.", max_tokens=600)
        except Exception as e:
            log_bot.warning("Resumen diario con Gemini falló: %s", e)
    if not texto:
        texto = f"{len(trades)} operaciones cerradas hoy (P&L {datos['pnl_hoy']:+.2f} USD), {len(abiertas_bot)} posiciones del bot y {len(opciones)} de opciones abiertas, {len(alertas)} alertas."
    emit("sistema", "info", f"Resumen diario {hoy}", {"resumen": texto, "pnl_hoy": datos["pnl_hoy"], "operaciones": len(trades), "alertas": len(alertas)})
    return texto


def ciclo() -> None:
    """Un ciclo completo sobre todos los activos activos."""
    for asset in db.consultar("SELECT * FROM bot_assets WHERE active = 1 AND training_status = 'listo'"):
        try:
            procesar_activo(asset)
        except Exception as e:
            log_bot.exception("Error procesando %s: %s", asset["symbol"], e)
            emit("sistema", "error", f"Error del bot en {asset['symbol']}", {"error": str(e)[:300]})


def main() -> None:
    """Punto de entrada del servicio `copiloto-bot`."""
    signal.signal(signal.SIGTERM, _parar)
    signal.signal(signal.SIGINT, _parar)
    db.inicializar()
    emit("sistema", "ok", "Bot iniciado", {"timeframe": cfg("bot", "timeframe"), "intervalo_s": cfg("bot", "intervalo_ciclo_s"), "forzar_paper": cfg("bot", "forzar_paper", default=False)})
    try:
        rec = reconciliar()
        emit("sistema", "info", "Reconciliación al arrancar", {k: v for k, v in rec.items() if v})
    except Exception as e:
        log_bot.error("Reconciliación falló: %s", e)
    intervalo = int(cfg("bot", "intervalo_ciclo_s", default=60))
    ultimo_refresco = 0.0
    while _corriendo:
        inicio = time.time()
        ciclo()
        if time.time() - ultimo_refresco > float(cfg("posiciones", "intervalo_refresco_min", default=5)) * 60:
            refrescar_posiciones_opciones()
            ultimo_refresco = time.time()
        try:
            resumen_diario()
        except Exception as e:
            log_bot.error("Resumen diario falló: %s", e)
        time.sleep(max(1.0, intervalo - (time.time() - inicio)))
    emit("sistema", "info", "Bot detenido", {})


if __name__ == "__main__":
    sys.exit(main())
