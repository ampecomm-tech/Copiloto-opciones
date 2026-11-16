"""Pestaña 1: noticias, sentimiento, IV vs HV, eventos y botón "Formular opinión con esto"."""
from __future__ import annotations

import streamlit as st

from agents.mesa import formular_opinion
from agents.sentiment import sentimiento as sentimiento_x
from app import auth, ui
from core.config import cfg
from data_sources import news
from data_sources.market import cadena_para, cierres_diarios, es_cripto, normalizar_activo
from quant.volatility import iv_atm, ratio_iv_hv, volatilidad_historica

NAN = float("nan")


@st.cache_data(ttl=300, show_spinner=False)
def _contexto_vol(activo: str) -> dict:
    """IV ATM, HV 30d y spot (cacheado 5 min)."""
    cadena, S, meta = cadena_para(activo)
    hv = volatilidad_historica(cierres_diarios(activo), 30, meta["periodos_por_anio"])
    iv = iv_atm(cadena, S, 30) if not cadena.empty else NAN
    return {"spot": S, "iv": iv, "hv": hv, "ratio": ratio_iv_hv(iv, hv)}


def _ok(x: float) -> bool:
    """True si el número no es NaN."""
    return x == x


def selector_activo(clave: str = "activo_noticias") -> str:
    """Selector de activo: favoritos de `config.yaml` como botones segmentados + campo para cualquier otro ticker."""
    defaults = cfg("app", "activos_por_defecto", default=["BTC", "NVDA"])
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    elegido = c1.segmented_control("Activo", defaults, default=defaults[0], key=f"{clave}_sel", required=True)
    otro = c2.text_input("Otro ticker", key=f"{clave}_txt", placeholder="Otro: TSLA, SOL…", label_visibility="collapsed")
    return normalizar_activo(otro or elegido or defaults[0])


def render() -> None:
    """Dibuja la pestaña."""
    st.title("Noticias y sentimiento")
    st.caption("Lo que pasa con el activo hoy: titulares, ánimo del mercado, volatilidad y agenda. Con esto puedes formular tu opinión.")
    usuario = auth.usuario_actual() or {}
    activo = selector_activo()

    try:
        v = _contexto_vol(activo)
        error_vol = None
    except Exception as e:
        v = {"iv": NAN, "hv": NAN, "ratio": NAN, "spot": NAN}
        error_vol = str(e)
    with st.spinner("Analizando el sentimiento…"):
        sent = sentimiento_x(activo, forzar=bool(st.session_state.pop("forzar_sent", False)), user_id=usuario.get("user_id"))
    score = float(sent.get("score", 0))
    fng = news.fear_and_greed() if es_cripto(activo) else None

    ratio = v["ratio"]
    lectura = ("opciones caras" if ratio > 1.2 else "opciones baratas" if ratio < 0.85 else "precio razonable") if _ok(ratio) else ""
    stats = [
        (f"{activo} spot", f"{v['spot']:,.2f}" if _ok(v["spot"]) else "n/d", ""),
        ("IV ATM 30d", f"{v['iv']:.1%}" if _ok(v["iv"]) else "n/d", "volatilidad implícita"),
        ("HV 30d", f"{v['hv']:.1%}" if _ok(v["hv"]) else "n/d", "volatilidad histórica"),
        ("IV / HV", f"{ratio:.2f}" if _ok(ratio) else "n/d", lectura),
    ]
    if fng:
        tono_fng = "rojo" if fng["valor"] < 40 else "verde" if fng["valor"] > 60 else "gris"
        stats.append(("Fear & Greed", ui.pill(str(fng["valor"]), tono_fng), fng["clasificacion"]))
    st.space("small")
    ui.franja_stats(stats)
    if error_vol:
        st.caption(f"No se pudo calcular IV/HV: {error_vol}")
    st.space("small")

    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        titulares = news.noticias(activo)
        st.subheader("Titulares")
        etiquetas = "Gemini" if any(t.get("fuente_etiqueta") == "gemini" for t in titulares) else "léxico (sin Gemini)"
        with st.container(border=True):
            st.html("<div class='cp-eyebrow'>Resumen del día</div>"
                    f"<div class='cp-texto'>{ui.esc(news.resumen_dia(activo, titulares))}</div>")
        if titulares:
            ui.lista_titulares(titulares)
            st.caption(f"Color de cada titular: sentimiento asignado por {etiquetas}.")
        else:
            st.caption("Sin titulares recientes.")

    with col_b:
        st.subheader("Sentimiento")
        with st.container(border=True):
            st.html(ui.medidor_sentimiento(score))
            if sent.get("proveedor"):
                st.caption(f"Fuente: {sent['proveedor']}" + (f" · en caché desde {ui.hora_local(sent['ts']):%H:%M}" if sent.get("cache") and sent.get("ts") else ""))
            if sent.get("resumen"):
                st.html(f"<div class='cp-texto'>{ui.esc(sent['resumen'])}</div>")
            if sent.get("fuentes"):
                with st.expander("Fuentes consultadas"):
                    st.html(ui.lista_puntos(sent["fuentes"][:8], "gris"))
            if st.button("Actualizar sentimiento", key="btn_sent", icon=":material/refresh:", type="tertiary", help="Ignora la caché de 15 min"):
                st.session_state["forzar_sent"] = True
                st.rerun()

        eventos = news.eventos_proximos(activo)
        st.subheader("Próximos eventos")
        with st.container(border=True):
            if eventos:
                st.html(ui.agenda(eventos))
            else:
                st.caption("Sin eventos próximos registrados.")

    st.space("medium")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="bottom")
        c1.html("<div class='cp-titulo__texto' style='font-size:1.05rem'>¿Ya tienes una idea?</div>"
                "<div class='cp-texto' style='font-size:0.875rem;color:#A3ACBA'>Gemini propone una opinión (dirección, magnitud y horizonte) con estos datos. "
                "Queda precargada en el Copiloto y puedes editarla.</div>")
        presupuesto = c2.number_input("Pérdida máxima aceptada (USD)", 50.0, 100000.0, 500.0, 50.0, key="pres_noticias")
        formular = c3.button("Formular opinión con esto", type="primary", icon=":material/auto_awesome:", use_container_width=True)
    if formular:
        contexto = {"activo": activo, "volatilidad": {"iv_atm": v["iv"] if _ok(v["iv"]) else None, "hv_30d": v["hv"] if _ok(v["hv"]) else None},
                    "sentimiento": {"score": sent.get("score"), "proveedor": sent.get("proveedor"), "resumen": sent.get("resumen")},
                    "titulares": [{"titulo": t["titulo"], "sentimiento": t["sentimiento"]} for t in titulares[:8]],
                    "eventos_proximos": eventos}
        with st.spinner("Gemini está formulando una opinión…"):
            op = formular_opinion(activo, contexto, presupuesto=presupuesto)
        st.session_state["opinion_precargada"] = {"activo": activo, **op}
        st.success(f"Opinión propuesta: {op['direccion']}, {op['magnitud_pct']}% en {op['horizonte_dias']} días. {op['razonamiento']}  \n"
                   "Ya está precargada en **Copiloto de opciones**.", icon=":material/check_circle:")
