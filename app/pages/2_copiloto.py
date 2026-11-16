"""Pestaña 2: copiloto de opciones (opinión → top 3 → mesa de agentes → semáforo → confirmar)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from agents import gemini_client as gem
from agents.mesa import formular_opinion
from app import auth, ui
from core import copiloto
from core.config import cfg
from quant.selector import Opinion
from quant.strategies_options import griegas_posicion, pnl_actual

DIRECCIONES = ["alcista", "bajista", "neutral", "volatil"]


def _formulario() -> Opinion | None:
    """Formulario de opinión (precargado desde Noticias si existe). Devuelve la opinión al enviar."""
    pre = st.session_state.get("opinion_precargada", {})
    defaults = cfg("app", "activos_por_defecto", default=["BTC", "NVDA"])
    with st.expander("✍️ Texto libre → opinión estructurada (LLM)", expanded=False):
        texto = st.text_area("Describe tu idea", placeholder="Creo que NVDA sube un 10% antes de resultados, arriesgo 800 USD", key="texto_libre")
        if st.button("Convertir a opinión") and texto.strip():
            with st.spinner("Interpretando…"):
                op = formular_opinion(pre.get("activo", defaults[0]), {"activo": pre.get("activo", defaults[0])}, texto_libre=texto, presupuesto=float(pre.get("presupuesto", 500)))
            st.session_state["opinion_precargada"] = {"activo": pre.get("activo", defaults[0]), **op}
            st.rerun()
    with st.form("opinion"):
        c = st.columns(5)
        activo = c[0].text_input("Activo", value=pre.get("activo", defaults[0]))
        direccion = c[1].selectbox("Dirección", DIRECCIONES, index=DIRECCIONES.index(pre.get("direccion", "alcista")) if pre.get("direccion") in DIRECCIONES else 0)
        mag = c[2].number_input("Magnitud esperada (%)", 0.5, 100.0, float(pre.get("magnitud_pct", 5.0)), 0.5)
        hor = c[3].number_input("Horizonte (días)", 1, 365, int(pre.get("horizonte_dias", 30)))
        pres = c[4].number_input("Presupuesto máx. de pérdida (USD)", 50.0, 1_000_000.0, float(pre.get("presupuesto", 500.0)), 50.0)
        if pre.get("razonamiento"):
            st.caption(f"Sugerido por el LLM: {pre['razonamiento']}")
        enviado = st.form_submit_button("🚀 Analizar", type="primary", use_container_width=True)
    if enviado:
        return Opinion(activo=activo, direccion=direccion, magnitud_pct=mag, horizonte_dias=int(hor), presupuesto=pres)
    return None


def _mostrar_candidata(i: int, e: dict[str, Any], res: dict[str, Any]) -> None:
    """Tarjeta completa de una candidata: métricas, payoff, griegas, Monte Carlo, semáforo y what-if."""
    sem, mc = e["semaforo"], e["montecarlo"]
    st.markdown(f"### {i + 1}. {e['nombre']}  {ui.semaforo_badge(sem['color'])}")
    st.caption(e.get("descripcion", ""))
    patas = ", ".join(f"{'Compra' if l['side'] > 0 else 'Vende'} {l['qty']:g} {l['type']} K={l['strike']:g} @ {l['price']:.2f} ({'ask' if l['side'] > 0 else 'bid'})" for l in e["legs"])
    st.markdown(f"**Patas:** {patas} · vence `{e['expiry']}` ({e['dte']:.0f} días)")
    m = st.columns(6)
    m[0].metric("Costo neto", f"{e['costo_neto']:+,.2f}", help="Positivo = débito pagado; negativo = crédito recibido")
    m[1].metric("Máx. ganancia", "Ilimitada" if e.get("ganancia_ilimitada") else f"{e['max_ganancia']:,.2f}")
    m[2].metric("Máx. pérdida", f"{e['max_perdida']:,.2f}")
    m[3].metric("Prob. ganancia", f"{mc['prob_ganancia']:.0%}", help=f"Escenario neutral: {mc['neutral']['prob_ganancia']:.0%}")
    m[4].metric("Valor esperado", f"{mc['valor_esperado']:+,.2f}", help=f"Escenario neutral: {mc['neutral']['valor_esperado']:+,.2f}")
    m[5].metric("Breakevens", ", ".join(f"{b:,.2f}" for b in e["breakevens"]) or "—")

    st.markdown("**What-if**")
    w = st.columns(3)
    spot_w = w[0].slider("Precio del subyacente", float(e["spot"] * 0.7), float(e["spot"] * 1.3), float(e["spot"]), key=f"spot_{i}")
    iv_w = w[1].slider("Cambio de IV (puntos)", -0.3, 0.3, 0.0, 0.01, key=f"iv_{i}")
    dias_w = w[2].slider("Días transcurridos", 0, int(max(e["dte"] - 1, 1)), 0, key=f"dias_{i}")
    pnl_w = pnl_actual(e, spot_w, res["r"], dias_w, iv_w)
    g_w = griegas_posicion(e["legs"], spot_w, res["r"], e["multiplicador"], dias_w)
    st.markdown(f"P&L teórico en ese escenario: **{pnl_w:+,.2f} USD** · delta {g_w['delta']:+.3f} · theta {g_w['theta']:+.2f}/día")

    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(ui.grafica_payoff(e, res["r"], dias_w, iv_w, spot_w), use_container_width=True, key=f"payoff_{i}")
    g2.plotly_chart(ui.grafica_histograma(mc), use_container_width=True, key=f"hist_{i}")
    st.markdown("**Griegas de la posición (hoy)**")
    st.dataframe(ui.tabla_griegas(e["griegas"]), hide_index=True, use_container_width=True)
    _boton_educativo(f"griegas_{i}", "qué significan estas griegas para esta posición concreta (delta, gamma, theta, vega) y qué le conviene que pase al precio y a la volatilidad",
                     {"estrategia": e["nombre"], "griegas": e["griegas"], "spot": e["spot"], "dte": e["dte"], "costo_neto": e["costo_neto"]})

    with st.container(border=True):
        st.markdown(f"**Semáforo {ui.semaforo_badge(sem['color'])}** · puntaje {sem['puntaje']:.2f}")
        for r_ in sem["razones"]:
            st.markdown(f"- ✅ {r_}")
        for a in sem["advertencias"]:
            st.markdown(f"- ⚠️ {a}")
        if sem.get("alternativa"):
            st.markdown(f"- 💡 Alternativa sugerida: `{sem['alternativa']}`")
        _boton_educativo(f"sem_{i}", f"por qué el semáforo de esta estrategia salió {sem['color']} y qué tendría que cambiar para mejorar",
                         {"estrategia": e["nombre"], "color": sem["color"], "razones": sem["razones"], "advertencias": sem["advertencias"], "reglas": sem["reglas"]})
    _boton_educativo(f"est_{i}", f"cómo funciona la estrategia '{e['nombre']}' ({e['tipo']}), cuándo gana, cuándo pierde y qué riesgo tiene",
                     {"tipo": e["tipo"], "patas": patas, "max_ganancia": "ilimitada" if e.get("ganancia_ilimitada") else e["max_ganancia"], "max_perdida": e["max_perdida"], "breakevens": e["breakevens"], "prob_ganancia": mc["prob_ganancia"]})


def _boton_educativo(clave: str, que: str, contexto: dict[str, Any]) -> None:
    """Modo educativo: botón "¿Qué significa esto?" que pide a Gemini una explicación del caso concreto."""
    if not gem.disponible():
        return
    if st.button("❓ ¿Qué significa esto?", key=f"edu_{clave}"):
        with st.spinner("Gemini está explicando…"):
            texto = gem.explicar(que, contexto)
        st.info(texto or "No se pudo generar la explicación.")


def _mostrar_mesa(res: dict[str, Any]) -> None:
    """Opiniones de los agentes y veredicto del juez."""
    mesa = res["mesa"]
    st.subheader("🧑‍⚖️ Mesa de agentes")
    if mesa["modo"] == "reglas":
        st.warning("Ni Together AI ni Gemini están disponibles: el veredicto se emitió solo con las reglas del semáforo.")
    cols = st.columns(3)
    nombres = {"analista": "📊 Analista de mercado", "sentimiento": "💬 Agente de sentimiento", "abogado_diablo": "😈 Abogado del diablo"}
    for col, (rol, o) in zip(cols, mesa["opiniones"].items()):
        with col, st.container(border=True):
            st.markdown(f"**{nombres.get(rol, rol)}**  \n<small>{o.get('modelo', '')}</small>", unsafe_allow_html=True)
            if o.get("ok"):
                r = o["respuesta"]
                st.markdown(f"Postura: **{r['postura']}** ({r['confianza']:.0%}) · prefiere #{r.get('estrategia_preferida') or '—'}")
                st.write(r["argumento"])
                for p in r.get("puntos_clave", []):
                    st.markdown(f"- {p}")
            else:
                st.error(f"No respondió: {o.get('error', '')}")
    v = mesa.get("veredicto")
    if v:
        with st.container(border=True):
            st.markdown(f"### ⚖️ Veredicto del gestor de riesgo: {ui.semaforo_badge(v['veredicto'])} → estrategia #{v['estrategia_elegida']} ({v['confianza']:.0%})")
            st.write(v["justificacion"])
            if v.get("riesgos_principales"):
                st.markdown("**Riesgos:** " + " · ".join(v["riesgos_principales"]))
            if v.get("condiciones"):
                st.markdown("**Condiciones:** " + " · ".join(v["condiciones"]))
            if v.get("agentes_considerados"):
                st.caption("Agentes considerados: " + ", ".join(v["agentes_considerados"]))


def render() -> None:
    """Dibuja la pestaña."""
    st.header("🧭 Copiloto de opciones")
    usuario = auth.usuario_actual() or {}
    opinion = _formulario()
    if opinion:
        barra = st.progress(0, text="Iniciando…")
        try:
            res = copiloto.analizar(opinion, usuario.get("user_id"), lambda m, f: barra.progress(min(int(f * 100), 100), text=m))
            st.session_state["resultado_copiloto"] = res
        except Exception as e:
            st.error(f"No se pudo completar el análisis: {e}")
        finally:
            barra.empty()

    res = st.session_state.get("resultado_copiloto")
    if not res:
        st.info("Define tu opinión y pulsa **Analizar**. El copiloto baja la cadena, calcula griegas y Monte Carlo, y convoca a la mesa.")
        return

    st.markdown(f"**{res['activo']}** · spot **{res['spot']:,.2f}** · IV ATM {res['iv']:.0%} · HV 30d {res['hv']:.0%} · IV/HV {res['iv_hv']:.2f} · "
                f"{res['cadena_resumen']['contratos']} contratos ({res['meta']['fuente']}) · sentimiento {float(res['sentimiento'].get('score', 0)):+.2f} ({res['sentimiento'].get('proveedor') or 'n/d'}) · "
                f"ejecución al confirmar: `{res['meta'].get('venue_ejecucion', 'paper')}`")
    _aviso_mercado(res)
    v = (res["mesa"] or {}).get("veredicto")
    elegida = (v["estrategia_elegida"] - 1) if v else 0
    tabs = st.tabs([f"{ui.EMOJI.get(e['semaforo']['color'], '⚪')} #{i + 1} {e['tipo']}" + (" ⭐" if i == elegida else "") for i, e in enumerate(res["candidatas"])])
    for i, (tab, e) in enumerate(zip(tabs, res["candidatas"])):
        with tab:
            _mostrar_candidata(i, e, res)
            if st.button(f"✅ Confirmar #{i + 1}", key=f"conf_{i}", type="primary" if i == elegida else "secondary"):
                with st.spinner("Enviando la orden al venue…"):
                    pos_id, ej = copiloto.confirmar(res, i, usuario.get("user_id"))
                if ej["venue"] == "paper":
                    st.warning(f"Posición #{pos_id} abierta en **paper interno**. {ej['detalle']}")
                else:
                    st.success(f"Posición #{pos_id} abierta en **{ej['venue']}** · órdenes {', '.join(ej['order_ids'])} · {ej['detalle']}. Revisa **Mis posiciones**.")
    st.divider()
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
        st.warning("El mercado de EE. UU. está cerrado. " + ("Si confirmas, la orden queda en cola en Alpaca hasta la apertura." if politica == "cola" else "Si confirmas, la orden se simula en paper interno (política: cancelar)."))


def _validacion_binance(res: dict[str, Any]) -> None:
    """Tabla comparativa de nuestro Black-Scholes contra el mark price, la IV y las griegas de Binance."""
    if res["meta"].get("fuente") != "binance_options":
        return
    with st.expander("🧪 Validación: nuestro Black-Scholes vs mark price de Binance", expanded=False):
        from data_sources.binance_options import cadena_opciones
        from validation.bs_vs_binance import comparar, resumen

        cad, S, _ = cadena_opciones(res["activo"])
        tabla = comparar(cad, S)
        if tabla.empty:
            st.info("Sin datos de mark price.")
            return
        r_ = resumen(tabla)
        m = st.columns(5)
        m[0].metric("Contratos", r_["n"])
        m[1].metric("Error precio mediano", f"{r_['err_precio_pct_mediano']:.2f}%")
        m[2].metric("MAE IV", f"{r_['mae_iv']:.4f}")
        m[3].metric("MAE delta", f"{r_['mae_delta']:.4f}")
        m[4].metric("Corr. vega", f"{r_['corr_vega']:.3f}")
        st.dataframe(tabla[["symbol", "type", "strike", "dte", "mark_iv", "bs_precio", "mark_price", "err_precio_pct", "iv_recuperada", "delta_bs", "delta_bin", "gamma_bs", "gamma_bin", "theta_bs", "theta_bin", "vega_bs", "vega_bin"]].round(5),
                     hide_index=True, use_container_width=True, height=320)
        st.download_button("⬇️ CSV para el informe", tabla.to_csv(index=False).encode(), f"{res['activo']}_bs_vs_binance.csv", "text/csv")
