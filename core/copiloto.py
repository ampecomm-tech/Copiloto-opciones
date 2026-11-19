"""Servicio del copiloto: encadena datos → selector → Monte Carlo → semáforo → mesa de agentes.

`analizar()` devuelve todo lo que la pestaña necesita y guarda la recomendación.
`confirmar()` abre la posición en paper trading.
"""
from __future__ import annotations

from typing import Any, Callable

from agents import mesa
from agents.sentiment import sentimiento as sentimiento_x
from core import db
from core.config import cfg
from core.events import emit
from core import execution
from core.positions import abrir_posicion
from data_sources import news
from data_sources.market import cadena_para, cierres_diarios, normalizar_activo
from quant.selector import Opinion, seleccionar
from quant.viability import evaluar
from quant.volatility import iv_atm, ratio_iv_hv, volatilidad_historica


def senal_bot_para(activo: str) -> dict[str, Any] | None:
    """Última señal del RF del bot sobre el activo (si está registrado)."""
    base = normalizar_activo(activo)
    fila = db.consultar_uno(
        "SELECT s.ts, s.action, s.confidence, s.details_json FROM bot_signals s JOIN bot_assets a ON a.id = s.asset_id "
        "WHERE s.strategy = 'random_forest' AND (a.symbol LIKE ? OR a.symbol LIKE ?) ORDER BY s.id DESC LIMIT 1", (f"{base}/%", f"{base}-%"))
    if not fila:
        return None
    d = db.de_json(fila["details_json"], {})
    return {"ts": fila["ts"], "accion": fila["action"], "confianza": round(fila["confidence"], 3), "prob_subida": d.get("prob_subida")}


def analizar(opinion: Opinion, user_id: str | None = None, progreso: Callable[[str, float], None] | None = None, con_mesa: bool = True) -> dict[str, Any]:
    """Pipeline completo del copiloto para una opinión.

    Returns:
        Diccionario con `spot`, `cadena_resumen`, `iv`, `hv`, `candidatas` (con semáforo),
        `mesa` (opiniones + veredicto), `sentimiento`, `recommendation_id`, `meta`.
    """
    def avisar(m: str, f: float) -> None:
        if progreso:
            progreso(m, f)

    r = float(cfg("app", "tasa_libre_riesgo", default=0.045))
    activo = normalizar_activo(opinion.activo)
    opinion.activo = activo
    avisar("Descargando cadena de opciones…", 0.1)
    cadena, S, meta = cadena_para(activo, r, opinion.horizonte_dias)
    if cadena.empty or S != S:
        raise RuntimeError(f"No hay cadena de opciones disponible para {activo}")
    avisar("Calculando volatilidad implícita e histórica…", 0.25)
    hv = volatilidad_historica(cierres_diarios(activo), 30, meta["periodos_por_anio"])
    iv = iv_atm(cadena, S, opinion.horizonte_dias)
    ratio = ratio_iv_hv(iv, hv)
    avisar("Generando candidatas y corriendo Monte Carlo…", 0.4)
    # Para BTC y ETH se prefieren contratos que existen en el testnet de Deribit, el único venue cripto que llena
    # órdenes; si con ellos no sale ninguna candidata, se usa la cadena completa (respaldo Gate.io o paper).
    cadena_ej, n_deribit = execution.filtrar_cadena_deribit(cadena, activo)
    candidatas = seleccionar(cadena_ej, S, opinion, r, meta["multiplicador"], meta["paso_qty"]) if n_deribit > 0 else []
    meta["filtro_deribit"] = {"contratos": n_deribit, "total": int(len(cadena)), "aplicado": bool(candidatas)}
    if not candidatas:
        candidatas = seleccionar(cadena, S, opinion, r, meta["multiplicador"], meta["paso_qty"])
    for c in candidatas:
        c["venue_previsto"] = execution.venue_previsto({**c, "activo": activo}, meta.get("venue_ejecucion", "paper"))
    if not candidatas:
        raise RuntimeError("No se encontraron estrategias líquidas dentro del presupuesto. Prueba con más presupuesto u otro horizonte.")
    avisar("Consultando sentimiento de X y noticias…", 0.55)
    sent = sentimiento_x(activo, user_id=user_id)
    titulares = news.noticias(activo)
    eventos = news.eventos_proximos(activo)
    for e in candidatas:
        e["semaforo"] = evaluar(e, ratio if ratio == ratio else None, sent.get("score") if sent.get("disponible", True) else None, opinion.model_dump()).to_dict()
    resultado_mesa: dict[str, Any] = {"opiniones": {}, "veredicto": None, "modo": "omitida"}
    if con_mesa:
        avisar("Deliberando en la mesa de agentes…", 0.7)
        contexto = mesa.construir_contexto(activo, S, opinion.model_dump(), candidatas, iv, hv, sent, titulares, senal_bot_para(activo), eventos)
        resultado_mesa = mesa.deliberar(contexto, candidatas, user_id)
    avisar("Guardando recomendación…", 0.95)
    verdict = (resultado_mesa.get("veredicto") or {}).get("veredicto") or candidatas[0]["semaforo"]["color"]
    rec_id = db.ejecutar(
        "INSERT INTO recommendations(ts, user_id, asset, view_json, candidates_json, agents_json, verdict) VALUES (?,?,?,?,?,?,?)",
        (db.ahora_iso(), user_id, activo, db.a_json(opinion.model_dump()), db.a_json([_sin_histograma(c) for c in candidatas]), db.a_json(resultado_mesa), verdict),
    )
    avisar("Listo", 1.0)
    return {"activo": activo, "spot": S, "meta": meta, "iv": iv, "hv": hv, "iv_hv": ratio, "candidatas": candidatas, "mesa": resultado_mesa,
            "sentimiento": sent, "titulares": titulares, "eventos": eventos, "recommendation_id": rec_id, "r": r,
            "cadena_resumen": {"contratos": int(len(cadena)), "vencimientos": sorted(cadena["expiry"].unique().tolist())}}


def _sin_histograma(c: dict[str, Any]) -> dict[str, Any]:
    """Copia de la candidata sin el histograma (para no inflar la DB)."""
    d = dict(c)
    if "montecarlo" in d:
        d["montecarlo"] = {k: v for k, v in d["montecarlo"].items() if k not in ("histograma", "histograma_precios", "cuantiles_pnl")}
    return d


def confirmar(resultado: dict[str, Any], indice: int, user_id: str | None = None) -> tuple[int, dict[str, Any]]:
    """Abre la candidata `indice` (0-based): envía la orden al venue (Binance options testnet o
    Alpaca Paper, con respaldo paper) y registra la posición y la elección en la recomendación.

    Returns:
        (id de la posición, resultado de la ejecución).
    """
    est = dict(resultado["candidatas"][indice])
    est["activo"] = resultado["activo"]
    db.ejecutar("UPDATE recommendations SET chosen_strategy_json = ? WHERE id = ?", (db.a_json(_sin_histograma(est)), resultado["recommendation_id"]))
    ejecucion = execution.ejecutar(est, resultado["meta"].get("venue_ejecucion", "paper"), user_id)
    pos_id = abrir_posicion(user_id, resultado["recommendation_id"], resultado["activo"], est, ejecucion)
    return pos_id, ejecucion
