"""Pestaña 2: copiloto de opciones (opinión → top 3 → mesa de agentes → semáforo → confirmar)."""
from __future__ import annotations

from typing import Any

import streamlit as st

from agents import gemini_client as gem
from agents.mesa import formular_opinion
from app import auth, ui
from core import copiloto
from core.config import cfg
from quant.selector import Opinion
from quant.strategies_options import griegas_posicion, pnl_actual

DIRECCIONES = ["alcista", "bajista", "neutral", "volatil"]
NOMBRE_DIRECCION = {"alcista": "Alcista", "bajista": "Bajista", "neutral": "Neutral", "volatil": "Volátil"}
NOMBRE_VENUE = {"deribit_testnet": "Deribit testnet", "gate_options_testnet": "Gate.io testnet", "binance_options_testnet": "Binance Options testnet",
                "alpaca_paper": "Alpaca Paper", "paper": "Paper interno"}
NOMBRE_FUENTE = {"binance_options": "Binance Options", "alpaca": "Alpaca", "yfinance": "yfinance"}
ROLES = {
    "analista": (":material/query_stats:", "Analista de mercado"),
    "sentimiento": (":material/forum:", "Agente de sentimiento"),
    "abogado_diablo": (":material/report:", "Abogado del diablo"),
}


# --- Formulario --------------------------------------------------------------------------------

def _formulario() -> Opinion | None:
    """Formulario de opinión (precargado desde Noticias si existe). Devuelve la opinión al enviar."""
    pre = st.session_state.get("opinion_precargada", {})
    defaults = cfg("app", "activos_por_defecto", default=["BTC", "NVDA"])
    with st.form("opinion", border=True):
        c1, c2 = st.columns([1, 3], vertical_alignment="top")
        activo = c1.text_input("Activo", value=pre.get("activo", defaults[0]), help="Ticker o cripto: BTC, ETH, SOL, NVDA, AAPL…")
        direccion = c2.segmented_control("Dirección", DIRECCIONES, default=pre.get("direccion") if pre.get("direccion") in DIRECCIONES else "alcista",
                                         format_func=NOMBRE_DIRECCION.get, required=True)
        c = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
        mag = c[0].number_input("Magnitud esperada (%)", 0.5, 100.0, float(pre.get("magnitud_pct", 5.0)), 0.5)
        hor = c[1].number_input("Horizonte (días)", 1, 365, int(pre.get("horizonte_dias", 30)))
        pres = c[2].number_input("Pérdida máxima aceptada (USD)", 50.0, 1_000_000.0, float(pre.get("presupuesto", 500.0)), 50.0)
        enviado = c[3].form_submit_button("Analizar", type="primary", icon=":material/arrow_forward:", use_container_width=True)
        if pre.get("razonamiento"):
            st.caption(f"Propuesta de Gemini: {pre['razonamiento']}")
    with st.expander("Describir la idea en texto libre", icon=":material/edit_note:"):
        texto = st.text_area("Tu idea", placeholder="Creo que NVDA sube un 10% antes de resultados, arriesgo 800 USD", key="texto_libre", label_visibility="collapsed")
        if st.button("Convertir a opinión", icon=":material/auto_awesome:") and texto.strip():
            with st.spinner("Interpretando tu idea…"):
                op = formular_opinion(pre.get("activo", defaults[0]), {"activo": pre.get("activo", defaults[0])}, texto_libre=texto, presupuesto=float(pre.get("presupuesto", 500)))
            st.session_state["opinion_precargada"] = {"activo": pre.get("activo", defaults[0]), **op}
            st.rerun()
    if enviado:
        return Opinion(activo=activo, direccion=direccion or "alcista", magnitud_pct=mag, horizonte_dias=int(hor), presupuesto=pres)
    return None


# --- Contexto y veredicto ----------------------------------------------------------------------

def _contexto(res: dict[str, Any]) -> None:
    """Franja con los datos del subyacente, la volatilidad, el sentimiento y el venue de ejecución."""
    score = float(res["sentimiento"].get("score", 0))
    tono = "verde" if score > 0.15 else "rojo" if score < -0.15 else "gris"
    ratio = res["iv_hv"]
    lectura_ratio = "opciones caras" if ratio > 1.2 else "opciones baratas" if ratio < 0.85 else "precio razonable"
    fuente = NOMBRE_FUENTE.get(res["meta"]["fuente"], res["meta"]["fuente"])
    venue = res["meta"].get("venue_ejecucion", "paper")
    ui.franja_stats([
        (f"{res['activo']} spot", f"{res['spot']:,.2f}", ""),
        ("IV ATM", f"{res['iv']:.1%}", f"HV 30d {res['hv']:.1%}"),
        ("IV / HV", f"{ratio:.2f}", lectura_ratio),
        ("Sentimiento", ui.pill(f"{score:+.2f}", tono), res["sentimiento"].get("proveedor") or "sin proveedor"),
        ("Cadena", f"{res['cadena_resumen']['contratos']} contratos", fuente),
        ("Ejecución", ui.esc(NOMBRE_VENUE.get(venue, venue)), "al confirmar"),
    ])


def _veredicto(res: dict[str, Any]) -> None:
    """Tarjeta principal: qué recomienda el gestor de riesgo y por qué."""
    mesa = res.get("mesa") or {}
    v = mesa.get("veredicto")
    if not v:
        return
    idx = v["estrategia_elegida"] - 1
    elegida = res["candidatas"][idx] if 0 <= idx < len(res["candidatas"]) else None
    with st.container(border=True):
        st.html("<div class='cp-eyebrow'>Veredicto del gestor de riesgo</div>"
                "<div class='cp-titulo'>"
                f"<span class='cp-titulo__texto'>#{v['estrategia_elegida']} · {ui.esc(elegida['nombre'] if elegida else 'Sin estrategia')}</span>"
                f"{ui.semaforo_pill(v['veredicto'])}"
                f"{ui.pill('Confianza {:.0%}'.format(v['confianza']), 'gris', punto=False)}"
                "</div>")
        if mesa.get("modo") == "reglas":
            st.caption("Ni Together AI ni Gemini respondieron: el veredicto sale solo de las reglas del semáforo.")
        st.html(f"<div class='cp-texto'>{ui.esc(v['justificacion'])}</div>")
        riesgos, condiciones = v.get("riesgos_principales") or [], v.get("condiciones") or []
        if riesgos or condiciones:
            c1, c2 = st.columns(2, gap="large")
            if riesgos:
                c1.html("<div class='cp-eyebrow'>Riesgos principales</div>" + ui.lista_puntos(riesgos, "rojo"))
            if condiciones:
                c2.html("<div class='cp-eyebrow'>Condiciones para entrar</div>" + ui.lista_puntos(condiciones, "acento"))
        if v.get("agentes_considerados"):
            st.caption("Decidió con: " + ", ".join(ROLES.get(a, ("", a))[1] for a in v["agentes_considerados"]))


# --- Candidatas --------------------------------------------------------------------------------

def _mostrar_candidata(i: int, e: dict[str, Any], res: dict[str, Any], recomendada: bool) -> None:
    """Tarjeta completa de una candidata: patas, cifras, what-if, payoff, Monte Carlo, griegas y semáforo."""
    sem, mc = e["semaforo"], e["montecarlo"]
    st.html("<div class='cp-titulo'>"
            f"<span class='cp-titulo__texto'>{ui.esc(e['nombre'])}</span>{ui.semaforo_pill(sem['color'])}"
            + (ui.pill("Elegida por la mesa", "acento", punto=False) if recomendada else "") + "</div>")
    if e.get("descripcion"):
        st.caption(e["descripcion"])
    ui.tabla_patas(e["legs"])
    st.caption(f"Vence el {e['expiry']} · {e['dte']:.0f} días · precios de ejecución reales: ask al comprar, bid al vender")

    m = st.columns(6)
    m[0].metric("Costo neto (USD)", f"{e['costo_neto']:+,.2f}", help="Positivo = débito pagado; negativo = crédito recibido", border=True)
    m[1].metric("Pérdida máxima", f"{e['max_perdida']:,.2f}", border=True)
    m[2].metric("Ganancia máxima", "Ilimitada" if e.get("ganancia_ilimitada") else f"{e['max_ganancia']:,.2f}", border=True)
    m[3].metric("Prob. ganancia", f"{mc['prob_ganancia']:.0%}", help=f"Escenario neutral: {mc['neutral']['prob_ganancia']:.0%}", border=True)
    m[4].metric("Valor esperado", f"{mc['valor_esperado']:+,.2f}", help=f"Escenario neutral: {mc['neutral']['valor_esperado']:+,.2f}", border=True)
    m[5].metric("Breakeven", " / ".join(f"{b:,.0f}" for b in e["breakevens"]) or "—", border=True)

    with st.container(border=True):
        st.markdown("**Escenario what-if**")
        w = st.columns(3, gap="large")
        spot_w = w[0].slider("Precio del subyacente", float(e["spot"] * 0.7), float(e["spot"] * 1.3), float(e["spot"]), key=f"spot_{i}")
        iv_w = w[1].slider("Cambio de IV (puntos)", -0.3, 0.3, 0.0, 0.01, key=f"iv_{i}")
        dias_w = w[2].slider("Días transcurridos", 0, int(max(e["dte"] - 1, 1)), 0, key=f"dias_{i}")
        pnl_w = pnl_actual(e, spot_w, res["r"], dias_w, iv_w)
        g_w = griegas_posicion(e["legs"], spot_w, res["r"], e["multiplicador"], dias_w)
        r = st.columns(3, gap="large")
        r[0].metric("P&L teórico (USD)", f"{pnl_w:+,.2f}")
        r[1].metric("Delta", f"{g_w['delta']:+.3f}")
        r[2].metric("Theta por día", f"{g_w['theta']:+.2f}")

    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(ui.grafica_payoff(e, res["r"], dias_w, iv_w, spot_w), use_container_width=True, key=f"payoff_{i}")
    g2.plotly_chart(ui.grafica_histograma(mc), use_container_width=True, key=f"hist_{i}")
    _riesgo_candidata(i, e, mc)

    patas_txt = ", ".join(f"{'Compra' if l['side'] > 0 else 'Vende'} {l['qty']:g} {l['type']} K={l['strike']:g} @ {l['price']:.2f}" for l in e["legs"])
    c_g, c_s = st.columns(2, gap="medium")
    with c_g, st.container(border=True, height="stretch"):
        st.markdown("**Griegas de la posición hoy**")
        g = e["griegas"]
        f1 = st.columns(2)
        f1[0].metric("Delta", f"{g['delta']:+.4f}")
        f1[1].metric("Gamma", f"{g['gamma']:+.6f}")
        f2 = st.columns(2)
        f2[0].metric("Theta por día", f"{g['theta']:+.2f}")
        f2[1].metric("Vega", f"{g['vega']:+.2f}")
        _boton_educativo(f"griegas_{i}", "qué significan estas griegas para esta posición concreta (delta, gamma, theta, vega) y qué le conviene que pase al precio y a la volatilidad",
                         {"estrategia": e["nombre"], "griegas": e["griegas"], "spot": e["spot"], "dte": e["dte"], "costo_neto": e["costo_neto"]})
    with c_s, st.container(border=True, height="stretch"):
        st.html("<div class='cp-titulo'><strong>Semáforo de viabilidad</strong>"
                f"{ui.semaforo_pill(sem['color'])}{ui.pill('Puntaje {:.2f}'.format(sem['puntaje']), 'gris', punto=False)}</div>")
        if sem["razones"]:
            st.html(ui.lista_puntos(sem["razones"], "verde"))
        if sem["advertencias"]:
            st.html(ui.lista_puntos(sem["advertencias"], "amarillo"))
        if sem.get("alternativa"):
            st.html(ui.lista_puntos([f"Alternativa sugerida: {sem['alternativa']}"], "acento"))
        _boton_educativo(f"sem_{i}", f"por qué el semáforo de esta estrategia salió {sem['color']} y qué tendría que cambiar para mejorar",
                         {"estrategia": e["nombre"], "color": sem["color"], "razones": sem["razones"], "advertencias": sem["advertencias"], "reglas": sem["reglas"]})

    _boton_educativo(f"est_{i}", f"cómo funciona la estrategia '{e['nombre']}' ({e['tipo']}), cuándo gana, cuándo pierde y qué riesgo tiene",
                     {"tipo": e["tipo"], "patas": patas_txt, "max_ganancia": "ilimitada" if e.get("ganancia_ilimitada") else e["max_ganancia"], "max_perdida": e["max_perdida"], "breakevens": e["breakevens"], "prob_ganancia": mc["prob_ganancia"]},
                     etiqueta="¿Cómo funciona esta estrategia?")
    _confirmar(i, res, recomendada)


def _riesgo_candidata(i: int, e: dict[str, Any], mc: dict[str, Any]) -> None:
    """Riesgo de cola de la candidata: VaR y CVaR del Monte Carlo, distribución del precio al horizonte y probabilidad acumulada."""
    if mc.get("var_95") is None:
        return  # análisis guardado antes de que existieran estas métricas
    with st.container(border=True):
        st.markdown("**Riesgo de la estrategia (Monte Carlo)**")
        st.caption("Pérdidas en USD al horizonte. VaR: pérdida que solo se supera en el peor 5 % (o 1 %) de las trayectorias. "
                   "CVaR: pérdida promedio dentro de ese peor 5 % (o 1 %). Un valor negativo significa que incluso esos escenarios terminan en ganancia.")
        m = st.columns(5)
        m[0].metric("VaR 95 %", f"{mc['var_95']:,.2f}", border=True)
        m[1].metric("CVaR 95 %", f"{mc['cvar_95']:,.2f}", border=True)
        m[2].metric("VaR 99 %", f"{mc['var_99']:,.2f}", border=True)
        m[3].metric("CVaR 99 %", f"{mc['cvar_99']:,.2f}", border=True)
        m[4].metric("CVaR 99 % / pérdida máx.", f"{mc['cvar_99'] / max(e['max_perdida'], 1e-9):.0%}", border=True,
                    help="Qué fracción de la pérdida máxima se pierde en promedio en el peor 1 % de los escenarios")
        t1, t2 = st.tabs(["Precio al horizonte", "Probabilidad acumulada"])
        with t1:
            st.plotly_chart(ui.grafica_precio_final(mc, e), use_container_width=True, key=f"precio_{i}")
        with t2:
            st.plotly_chart(ui.grafica_cdf(mc), use_container_width=True, key=f"cdf_{i}")
        _boton_educativo(f"var_{i}", "qué significan el VaR y el CVaR de esta estrategia y qué tan grave es su peor escenario comparado con la pérdida máxima",
                         {"estrategia": e["nombre"], "var_95": mc["var_95"], "cvar_95": mc["cvar_95"], "var_99": mc["var_99"], "cvar_99": mc["cvar_99"],
                          "max_perdida": e["max_perdida"], "prob_ganancia": mc["prob_ganancia"]})


def _confirmar(i: int, res: dict[str, Any], recomendada: bool) -> None:
    """Fila final de la candidata: a dónde va la orden y botón para confirmarla."""
    usuario = auth.usuario_actual() or {}
    venue = res["meta"].get("venue_ejecucion", "paper")
    previsto = res["candidatas"][i].get("venue_previsto") or {}
    st.divider()
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    if previsto.get("venue") in ("gate_options_testnet", "paper"):
        c1.html(f"<div class='cp-titulo'>{ui.pill('Va a ' + NOMBRE_VENUE.get(previsto['venue'], previsto['venue']), 'amarillo')}</div>")
        c1.caption(previsto.get("texto", ""))
    else:
        destino = previsto.get("venue") or venue
        c1.caption(f"Al confirmar se envía la orden a **{NOMBRE_VENUE.get(destino, destino)}** con los precios de la tabla. "
                   + (previsto.get("texto", "") + " " if previsto.get("texto") else "")
                   + "Si el venue rechaza una pata, se intenta el respaldo y, si tampoco puede, la posición se abre en paper interno con aviso en #sistema.")
    if c2.button(f"Confirmar #{i + 1}", key=f"conf_{i}", type="primary" if recomendada else "secondary", icon=":material/check:", use_container_width=True):
        with st.spinner("Enviando la orden…"):
            pos_id, ej = copiloto.confirmar(res, i, usuario.get("user_id"))
        if ej["venue"] == "paper":
            st.warning(f"Posición #{pos_id} abierta en paper interno. {ej['detalle']}", icon=":material/info:")
        else:
            st.success(f"Posición #{pos_id} abierta en {NOMBRE_VENUE.get(ej['venue'], ej['venue'])} · órdenes {', '.join(ej['order_ids'])} · {ej['detalle']}. "
                       "Puedes seguirla en Mis posiciones.", icon=":material/check_circle:")


def _boton_educativo(clave: str, que: str, contexto: dict[str, Any], etiqueta: str = "¿Qué significa esto?") -> None:
    """Modo educativo: botón que pide a Gemini una explicación del caso concreto en lenguaje simple."""
    if not gem.disponible():
        return
    if st.button(etiqueta, key=f"edu_{clave}", icon=":material/school:", type="tertiary"):
        with st.spinner("Gemini está preparando la explicación…"):
            texto = gem.explicar(que, contexto)
        st.info(texto or "No se pudo generar la explicación.", icon=":material/lightbulb:")


# --- Mesa de agentes ---------------------------------------------------------------------------

def _mostrar_mesa(res: dict[str, Any]) -> None:
    """Opinión de cada agente (modelo, postura, confianza y argumento)."""
    mesa = res["mesa"]
    st.subheader("Mesa de agentes")
    st.caption("Tres agentes de distintos modelos opinan en paralelo; el gestor de riesgo lee sus argumentos y el semáforo y decide.")
    cols = st.columns(3, gap="medium")
    for col, (rol, o) in zip(cols, mesa["opiniones"].items()):
        icono, nombre = ROLES.get(rol, (":material/person:", rol))
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"{icono} **{nombre}**")
            st.html(f"<div class='cp-modelo'>{ui.esc(o.get('modelo', ''))}</div>")
            if o.get("ok"):
                r = o["respuesta"]
                tono = ui.TONO_POSTURA.get(r["postura"], "gris")
                pref = r.get("estrategia_preferida")
                st.html("<div class='cp-titulo' style='margin-top:0.25rem'>"
                        f"{ui.pill(r['postura'].capitalize(), tono)}"
                        + (ui.pill(f'Prefiere #{pref}', 'gris', punto=False) if pref else "") + "</div>"
                        f"<div style='font-size:0.75rem;color:#8A94A6;margin-top:0.6rem'>Confianza {r['confianza']:.0%}</div>"
                        + ui.barra_confianza(r["confianza"], tono)
                        + f"<div class='cp-texto' style='margin-top:0.75rem'>{ui.esc(r['argumento'])}</div>")
                if r.get("puntos_clave"):
                    with st.expander("Puntos clave"):
                        st.html(ui.lista_puntos(r["puntos_clave"], tono))
            else:
                st.html(ui.pill("No respondió", "rojo") + f"<div class='cp-modelo' style='margin-top:0.5rem'>{ui.esc(o.get('error', ''))}</div>")


# --- Página ------------------------------------------------------------------------------------

def render() -> None:
    """Dibuja la pestaña."""
    st.title("Copiloto de opciones")
    st.caption("Describe tu opinión sobre un activo. El copiloto arma las estrategias que mejor encajan, las simula y una mesa de agentes decide cuál conviene.")
    usuario = auth.usuario_actual() or {}
    opinion = _formulario()
    if opinion:
        barra = st.progress(0, text="Iniciando…")
        try:
            res = copiloto.analizar(opinion, usuario.get("user_id"), lambda m, f: barra.progress(min(int(f * 100), 100), text=m))
            st.session_state["resultado_copiloto"] = res
        except Exception as e:
            st.error(f"No se pudo completar el análisis: {e}", icon=":material/error:")
        finally:
            barra.empty()

    res = st.session_state.get("resultado_copiloto")
    if not res:
        st.caption("Tras pulsar Analizar: cadena de opciones → griegas e IV → candidatas → Monte Carlo (10,000 trayectorias) → semáforo → mesa de agentes. Tarda unos 30 s.")
        return

    st.space("small")
    _contexto(res)
    _aviso_mercado(res)
    _veredicto(res)

    v = (res["mesa"] or {}).get("veredicto")
    elegida = (v["estrategia_elegida"] - 1) if v else 0
    etiquetas = [f":{ui.COLOR_MD.get(e['semaforo']['color'], 'gray')}[●] #{i + 1} {e['nombre']}" for i, e in enumerate(res["candidatas"])]
    st.subheader("Estrategias candidatas")
    tabs = st.tabs(etiquetas, default=etiquetas[elegida] if 0 <= elegida < len(etiquetas) else None)
    for i, (tab, e) in enumerate(zip(tabs, res["candidatas"])):
        with tab:
            _mostrar_candidata(i, e, res, recomendada=(i == elegida))
    st.space("medium")
    _mostrar_mesa(res)
    _validacion_binance(res)


def _aviso_mercado(res: dict[str, Any]) -> None:
    """Aviso cuando la ejecución iría a Alpaca y el mercado de EE. UU. está cerrado."""
    if res["meta"].get("venue_ejecucion") != "alpaca_paper":
        return
    from data_sources import alpaca

    abierto = alpaca.mercado_abierto()
    if abierto is False:
        politica = cfg("ejecucion", "mercado_cerrado", default="cola")
        st.warning("El mercado de EE. UU. está cerrado. " + ("Si confirmas, la orden queda en cola en Alpaca hasta la apertura." if politica == "cola" else "Si confirmas, la orden se simula en paper interno (política: cancelar)."),
                   icon=":material/schedule:")


def _validacion_binance(res: dict[str, Any]) -> None:
    """Tabla comparativa de nuestro Black-Scholes contra el mark price, la IV y las griegas de Binance."""
    if res["meta"].get("fuente") != "binance_options":
        return
    st.space("small")
    with st.expander("Validación: nuestro Black-Scholes vs mark price de Binance", icon=":material/fact_check:"):
        from data_sources.binance_options import cadena_opciones
        from validation.bs_vs_binance import comparar, resumen

        cad, S, _ = cadena_opciones(res["activo"])
        tabla = comparar(cad, S)
        if tabla.empty:
            st.caption("Sin datos de mark price.")
            return
        r_ = resumen(tabla)
        m = st.columns(5)
        m[0].metric("Contratos", r_["n"])
        m[1].metric("Error de precio mediano", f"{r_['err_precio_pct_mediano']:.2f}%")
        m[2].metric("MAE de IV", f"{r_['mae_iv']:.4f}")
        m[3].metric("MAE de delta", f"{r_['mae_delta']:.4f}")
        m[4].metric("Correlación de vega", f"{r_['corr_vega']:.3f}")
        st.dataframe(tabla[["symbol", "type", "strike", "dte", "mark_iv", "bs_precio", "mark_price", "err_precio_pct", "iv_recuperada", "delta_bs", "delta_bin", "gamma_bs", "gamma_bin", "theta_bs", "theta_bin", "vega_bs", "vega_bin"]].round(5),
                     hide_index=True, use_container_width=True, height=320)
        st.download_button("Descargar CSV para el informe", tabla.to_csv(index=False).encode(), f"{res['activo']}_bs_vs_binance.csv", "text/csv", icon=":material/download:")
