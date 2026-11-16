"""Orquestación de la mesa de agentes: una sola ronda (3 en paralelo) y luego el juez.

Entrada: contexto cuantitativo ya calculado (top 3 con Monte Carlo, griegas,
semáforo de cada candidata, IV/HV, sentimiento, noticias, señal del bot).
Salida: opiniones de cada agente + veredicto, todo emitido al canal `mesa`.

Proveedores: Together AI por defecto; cualquier rol puede correr con Gemini si en
`config.yaml` su modelo es `gemini` (por defecto, el analista de mercado), y Gemini
es el respaldo de toda la mesa si Together falla o no está configurado.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents import gemini_client as gem, roles, together_client as tc
from core.config import cfg
from core.events import emit, log_app

ROLES_PARALELOS = {"analista": roles.ANALISTA, "sentimiento": roles.SENTIMIENTO, "abogado_diablo": roles.ABOGADO_DIABLO}


def _modelo(rol: str) -> str:
    """Modelo configurado para el rol en `config.yaml` (`gemini` = usar Gemini)."""
    return str(cfg("agentes", "modelos", rol, default="meta-llama/Llama-3.3-70B-Instruct-Turbo"))


def _usa_gemini(rol: str) -> bool:
    """True si el rol está configurado con Gemini, o si Together no está y Gemini sí (respaldo)."""
    m = _modelo(rol).lower()
    if m.startswith("gemini"):
        return gem.disponible()
    return not tc.disponible() and gem.disponible()


def _llamar_json(rol: str, system: str, user: str, esquema, max_tokens: int | None = None):
    """Llama al proveedor del rol; si Together falla y Gemini está disponible, reintenta con Gemini."""
    if _usa_gemini(rol):
        return gem.generar_json(user, esquema, system=system), f"gemini:{gem.modelo()}"
    try:
        return tc.chat_json(_modelo(rol), system, user, esquema, max_tokens), _modelo(rol)
    except Exception as e:
        if gem.disponible():
            log_app.warning("Together falló para %s (%s); respaldo con Gemini", rol, str(e)[:100])
            return gem.generar_json(user, esquema, system=system), f"gemini:{gem.modelo()} (respaldo)"
        raise


def hay_proveedor() -> bool:
    """True si al menos un proveedor de la mesa está configurado."""
    return tc.disponible() or gem.disponible()


def _resumen_candidata(i: int, e: dict[str, Any]) -> dict[str, Any]:
    """Versión compacta de una candidata para no desperdiciar tokens."""
    mc, g, sem = e.get("montecarlo", {}), e.get("griegas", {}), e.get("semaforo", {})
    return {
        "indice": i,
        "nombre": e["nombre"], "tipo": e["tipo"], "descripcion": e.get("descripcion", ""),
        "patas": [f"{'compra' if l['side'] > 0 else 'vende'} {l['qty']:g} {l['type']} K={l['strike']:g} @ {l['price']:.2f}" for l in e["legs"]],
        "vencimiento": e["expiry"], "dte": round(e["dte"], 1),
        "costo_neto_usd": round(e["costo_neto"], 2), "max_ganancia_usd": ("ilimitada" if e.get("ganancia_ilimitada") else round(e["max_ganancia"], 2)),
        "max_perdida_usd": round(e["max_perdida"], 2), "breakevens": e["breakevens"],
        "griegas": {k: round(v, 4) for k, v in g.items()},
        "montecarlo": {"prob_ganancia": round(mc.get("prob_ganancia", 0), 3), "valor_esperado_usd": round(mc.get("valor_esperado", 0), 2),
                        "p5": round(mc.get("pnl_p5", 0), 2), "p95": round(mc.get("pnl_p95", 0), 2),
                        "escenario_neutral": mc.get("neutral", {})},
        "semaforo": {"color": sem.get("color"), "puntaje": sem.get("puntaje"), "advertencias": sem.get("advertencias", []), "razones": sem.get("razones", [])[:4]},
    }


def construir_contexto(activo: str, spot: float, opinion: dict[str, Any], candidatas: list[dict[str, Any]], iv: float, hv: float,
                       sentimiento: dict[str, Any], titulares: list[dict[str, Any]], senal_bot: dict[str, Any] | None, eventos: list[dict[str, str]]) -> dict[str, Any]:
    """Arma el diccionario de contexto que ven todos los agentes."""
    return {
        "activo": activo, "spot": spot,
        "opinion_usuario": opinion,
        "volatilidad": {"iv_atm": round(iv, 4) if iv == iv else None, "hv_30d": round(hv, 4) if hv == hv else None, "iv_sobre_hv": round(iv / hv, 3) if hv and hv == hv and iv == iv else None},
        "sentimiento": {"score": sentimiento.get("score"), "proveedor": sentimiento.get("proveedor"), "resumen": sentimiento.get("resumen"), "fuentes": sentimiento.get("fuentes", [])[:5]},
        "titulares": [{"titulo": t["titulo"], "sentimiento": t["sentimiento"]} for t in titulares[:8]],
        "eventos_proximos": eventos[:5],
        "senal_bot_rf": senal_bot,
        "candidatas": [_resumen_candidata(i + 1, e) for i, e in enumerate(candidatas)],
    }


def _llamar_agente(rol: str, contexto: dict[str, Any]) -> dict[str, Any]:
    """Llama a un agente paralelo; devuelve dict con `ok`, `respuesta` o `error`."""
    try:
        r, modelo_usado = _llamar_json(rol, ROLES_PARALELOS[rol], "CONTEXTO:\n" + json.dumps(contexto, ensure_ascii=False), roles.OpinionAgente)
        r.rol = rol
        return {"ok": True, "modelo": modelo_usado, "respuesta": r.model_dump()}
    except Exception as e:
        log_app.error("Agente %s falló: %s", rol, e)
        return {"ok": False, "modelo": _modelo(rol), "error": str(e)[:300]}


def _veredicto_sin_llm(candidatas: list[dict[str, Any]], opiniones: dict[str, Any]) -> roles.Veredicto:
    """Veredicto de respaldo basado solo en el semáforo (cuando el juez o Together no está)."""
    orden = {"verde": 0, "amarillo": 1, "rojo": 2}
    mejor = min(range(len(candidatas)), key=lambda i: (orden.get(candidatas[i]["semaforo"]["color"], 3), -candidatas[i].get("score", 0)))
    sem = candidatas[mejor]["semaforo"]
    return roles.Veredicto(
        veredicto=sem["color"], estrategia_elegida=mejor + 1, confianza=float(sem.get("puntaje", 0.5)),
        justificacion="Veredicto automático por reglas del semáforo (el juez LLM no estaba disponible). " + " ".join(sem.get("razones", [])[:3]),
        riesgos_principales=sem.get("advertencias", []), condiciones=[f"Alternativa sugerida: {sem['alternativa']}"] if sem.get("alternativa") else [],
        agentes_considerados=[r for r, o in opiniones.items() if o.get("ok")],
    )


def deliberar(contexto: dict[str, Any], candidatas: list[dict[str, Any]], user_id: str | None = None) -> dict[str, Any]:
    """Corre la mesa completa y emite cada opinión y el veredicto al canal `mesa`.

    Returns:
        `{"opiniones": {rol: {...}}, "veredicto": {...}, "modo": "llm" | "reglas"}`.
    """
    activo = contexto["activo"]
    emit("mesa", "info", f"Mesa convocada para {activo}", {"opinion": contexto["opinion_usuario"], "candidatas": [c["nombre"] for c in candidatas]}, user_id)
    opiniones: dict[str, Any] = {}
    if hay_proveedor():
        with ThreadPoolExecutor(max_workers=3) as ex:
            futuros = {rol: ex.submit(_llamar_agente, rol, contexto) for rol in ROLES_PARALELOS}
            for rol, f in futuros.items():
                opiniones[rol] = f.result()
    else:
        for rol in ROLES_PARALELOS:
            opiniones[rol] = {"ok": False, "modelo": "-", "error": "Ni TOGETHER_API_KEY ni GEMINI_API_KEY configuradas"}

    for rol, o in opiniones.items():
        if o["ok"]:
            r = o["respuesta"]
            tipo = {"favorable": "verde", "desfavorable": "rojo"}.get(r["postura"], "amarillo")
            emit("mesa", tipo, f"[{rol}] {r['postura']} ({r['confianza']:.0%}) — {activo}", {"modelo": o["modelo"], "argumento": r["argumento"], "puntos_clave": r["puntos_clave"], "estrategia_preferida": r.get("estrategia_preferida")}, user_id)
        else:
            emit("mesa", "error", f"[{rol}] no respondió — {activo}", {"error": o.get("error")}, user_id)

    modo = "reglas"
    veredicto: roles.Veredicto
    if hay_proveedor():
        try:
            entrada = {"contexto": contexto, "opiniones_agentes": {r: (o["respuesta"] if o["ok"] else {"error": o.get("error")}) for r, o in opiniones.items()}}
            veredicto, _ = _llamar_json("juez", roles.JUEZ, "ENTRADA:\n" + json.dumps(entrada, ensure_ascii=False), roles.Veredicto)
            # El semáforo manda: nunca mejor color que el de la candidata elegida.
            orden = {"verde": 0, "amarillo": 1, "rojo": 2}
            sem_color = candidatas[veredicto.estrategia_elegida - 1]["semaforo"]["color"]
            if orden.get(veredicto.veredicto, 2) < orden.get(sem_color, 2):
                veredicto.veredicto = sem_color
                veredicto.condiciones.append("Veredicto ajustado al semáforo determinista.")
            modo = "llm"
        except Exception as e:
            log_app.error("El juez falló: %s", e)
            veredicto = _veredicto_sin_llm(candidatas, opiniones)
    else:
        veredicto = _veredicto_sin_llm(candidatas, opiniones)

    elegida = candidatas[veredicto.estrategia_elegida - 1]["nombre"]
    emit("mesa", veredicto.veredicto, f"[juez] {veredicto.veredicto.upper()} → {elegida} ({activo})",
         {"modo": modo, "confianza": veredicto.confianza, "justificacion": veredicto.justificacion, "riesgos": veredicto.riesgos_principales, "condiciones": veredicto.condiciones}, user_id)
    return {"opiniones": opiniones, "veredicto": veredicto.model_dump(), "modo": modo}


def formular_opinion(activo: str, contexto: dict[str, Any], texto_libre: str | None = None, presupuesto: float = 500.0) -> dict[str, Any]:
    """Pide una opinión estructurada a partir de noticias/sentimiento o de texto libre.

    Orden de proveedores: **Gemini** (tarea de apoyo), luego Together, y si ninguno
    está disponible una heurística determinista basada en el sentimiento y la IV.
    """
    user = ("TEXTO DEL USUARIO:\n" + texto_libre + "\n\n" if texto_libre else "") + f"PRESUPUESTO POR DEFECTO: {presupuesto}\nCONTEXTO:\n" + json.dumps(contexto, ensure_ascii=False)
    if gem.disponible():
        try:
            r = gem.generar_json(user, roles.OpinionEstructurada, system=roles.FORMULAR_OPINION)
            return r.model_dump() | {"fuente": "gemini"}
        except Exception as e:
            log_app.error("formular_opinion con Gemini falló: %s", e)
    if tc.disponible():
        try:
            r = tc.chat_json(_modelo("opinion"), roles.FORMULAR_OPINION, user, roles.OpinionEstructurada, max_tokens=500)
            return r.model_dump() | {"fuente": "together"}
        except Exception as e:
            log_app.error("formular_opinion con Together falló: %s", e)
    score = float((contexto.get("sentimiento") or contexto.get("sentimiento_x") or {}).get("score") or 0.0)
    iv = (contexto.get("volatilidad") or {}).get("iv_atm") or 0.5
    direccion = "alcista" if score > 0.2 else "bajista" if score < -0.2 else "neutral"
    mag = round(iv * (30 / 365) ** 0.5 * 100, 1)   # movimiento 1σ a 30 días
    return {"direccion": direccion, "magnitud_pct": mag, "horizonte_dias": 30, "presupuesto": presupuesto,
            "razonamiento": f"Heurística sin LLM: sentimiento {score:+.2f} → {direccion}; magnitud = 1σ de la IV a 30 días ({mag}%).", "fuente": "heuristica"}
