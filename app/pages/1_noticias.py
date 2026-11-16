"""Pestaña 1: noticias, sentimiento de X, IV vs HV, eventos y botón "Formular opinión con esto"."""
from __future__ import annotations

import streamlit as st

from agents.sentiment import sentimiento as sentimiento_x
from agents.mesa import formular_opinion
from app import auth
from core.config import cfg
from data_sources import news
from data_sources.market import cadena_para, cierres_diarios, es_cripto, normalizar_activo
from quant.volatility import iv_atm, ratio_iv_hv, volatilidad_historica

ICONO = {"positivo": "🟢", "negativo": "🔴", "neutral": "⚪"}


@st.cache_data(ttl=300, show_spinner=False)
def _contexto_vol(activo: str) -> dict:
    """IV ATM, HV 30d y spot (cacheado 5 min)."""
    cadena, S, meta = cadena_para(activo)
    hv = volatilidad_historica(cierres_diarios(activo), 30, meta["periodos_por_anio"])
    iv = iv_atm(cadena, S, 30) if not cadena.empty else float("nan")
    return {"spot": S, "iv": iv, "hv": hv, "ratio": ratio_iv_hv(iv, hv)}


def selector_activo(clave: str = "activo_noticias") -> str:
    """Selector de activo compartido (lista por defecto + texto libre)."""
    defaults = cfg("app", "activos_por_defecto", default=["BTC", "NVDA"])
    c1, c2 = st.columns([2, 1])
    with c1:
        elegido = st.selectbox("Activo", defaults, key=f"{clave}_sel")
    with c2:
        otro = st.text_input("…u otro ticker", key=f"{clave}_txt", placeholder="AAPL, ETH, TSLA")
    return normalizar_activo(otro or elegido)


def render() -> None:
    """Dibuja la pestaña."""
    st.header("📰 Noticias y sentimiento")
    usuario = auth.usuario_actual() or {}
    activo = selector_activo()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader(f"Titulares de {activo}")
        titulares = news.noticias(activo)
        st.caption(news.resumen_dia(activo, titulares) + ("  · etiquetas: Gemini" if any(t.get("fuente_etiqueta") == "gemini" for t in titulares) else "  · etiquetas: léxico (sin Gemini)"))
        for t in titulares:
            st.markdown(f"{ICONO.get(t['sentimiento'], '⚪')} [{t['titulo']}]({t['url']}) — *{t['fuente']}* {t['fecha']}")
        if not titulares:
            st.info("Sin titulares recientes.")

    with col_b:
        st.subheader("Sentimiento del mercado")
        forzar = st.button("🔄 Actualizar sentimiento", key="btn_sent")
        with st.spinner("Analizando sentimiento…"):
            sent = sentimiento_x(activo, forzar=forzar, user_id=usuario.get("user_id"))
        score = float(sent.get("score", 0))
        st.metric("Puntaje (-1 a 1)", f"{score:+.2f}", help="Cacheado 15 min")
        st.progress(int((score + 1) / 2 * 100))
        if sent.get("proveedor"):
            st.caption(f"**Fuente: {sent['proveedor']}**" + (f" · desde caché ({sent.get('ts', '')[:16]})" if sent.get("cache") else ""))
        st.write(sent.get("resumen", ""))
        for p in sent.get("fuentes", [])[:6]:
            st.markdown(f"- {p}")

        st.subheader("Volatilidad y contexto")
        try:
            v = _contexto_vol(activo)
            m1, m2, m3 = st.columns(3)
            m1.metric("Spot", f"{v['spot']:,.2f}")
            m2.metric("IV ATM 30d", f"{v['iv']:.0%}" if v["iv"] == v["iv"] else "n/d")
            m3.metric("HV 30d", f"{v['hv']:.0%}" if v["hv"] == v["hv"] else "n/d")
            if v["ratio"] == v["ratio"]:
                st.caption(f"IV/HV = {v['ratio']:.2f} → " + ("opciones caras (favorece vender vol)" if v["ratio"] > 1.2 else "opciones baratas (favorece comprar vol)" if v["ratio"] < 0.85 else "precio de opciones razonable"))
        except Exception as e:
            v = {"iv": float("nan"), "hv": float("nan"), "ratio": float("nan"), "spot": float("nan")}
            st.warning(f"No se pudo calcular IV/HV: {e}")
        if es_cripto(activo):
            fng = news.fear_and_greed()
            if fng:
                st.metric("Fear & Greed (cripto)", f"{fng['valor']} · {fng['clasificacion']}")
        eventos = news.eventos_proximos(activo)
        if eventos:
            st.markdown("**Próximos eventos**")
            for ev in eventos:
                st.markdown(f"- `{ev['fecha']}` {ev['descripcion']}")

    st.divider()
    c1, c2 = st.columns([1, 3])
    with c1:
        presupuesto = st.number_input("Presupuesto máx. de pérdida (USD)", 50.0, 100000.0, 500.0, 50.0, key="pres_noticias")
    with c2:
        st.write("")
        st.write("")
        if st.button("🧭 Formular opinión con esto", type="primary", use_container_width=True):
            contexto = {"activo": activo, "volatilidad": {"iv_atm": v["iv"] if v["iv"] == v["iv"] else None, "hv_30d": v["hv"] if v["hv"] == v["hv"] else None},
                        "sentimiento": {"score": sent.get("score"), "proveedor": sent.get("proveedor"), "resumen": sent.get("resumen")}, "titulares": [{"titulo": t["titulo"], "sentimiento": t["sentimiento"]} for t in titulares[:8]],
                        "eventos_proximos": eventos}
            with st.spinner("El LLM está formulando una opinión…"):
                op = formular_opinion(activo, contexto, presupuesto=presupuesto)
            st.session_state["opinion_precargada"] = {"activo": activo, **op}
            st.success(f"Opinión propuesta: **{op['direccion']}**, {op['magnitud_pct']}% en {op['horizonte_dias']} días. {op['razonamiento']}")
            st.info("Ve a la pestaña **Copiloto de opciones**: la opinión ya está precargada (editable).")
