"""Pestaña 5: Centro de control — feed en vivo de `events` con filtros y autorefresco."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import streamlit as st

from app import ui
from core.config import cfg
from core.events import CANALES, leer_eventos

VENTANAS = {1: "Última hora", 6: "Últimas 6 h", 24: "Últimas 24 h", 72: "Últimos 3 días", 24 * 30: "Últimos 30 días"}


def _nombre_dia(d: datetime) -> str:
    """Etiqueta del separador de día: Hoy, Ayer o la fecha."""
    hoy = ui.hora_local(datetime.now(timezone.utc).isoformat()).date()
    if d.date() == hoy:
        return "Hoy"
    if d.date() == hoy - timedelta(days=1):
        return "Ayer"
    return d.strftime("%d/%m/%Y")


def _feed_html(eventos: list[dict], visto: int) -> str:
    """HTML del feed completo con separadores por día; marca como nuevos los eventos con id > `visto`."""
    partes: list[str] = []
    dia_actual = None
    for ev in eventos:
        dia = _nombre_dia(ui.hora_local(ev["ts"]))
        if dia != dia_actual:
            partes.append(f"<div class='cp-dia'>{dia}</div>")
            dia_actual = dia
        partes.append(ui.html_evento(ev, nuevo=visto > 0 and ev["id"] > visto))
    return "".join(partes)


def render() -> None:
    """Dibuja la pestaña."""
    st.title("Centro de control")
    st.caption("Todo lo que razonan la mesa de agentes y el bot, en el mismo orden en que llega a Discord.")

    canales = st.pills("Canales", list(CANALES), selection_mode="multi", default=list(CANALES),
                       format_func=lambda k: ui.CANAL_DISCORD[k], label_visibility="collapsed")
    c = st.columns([2, 1.2, 1, 1], vertical_alignment="center")
    activo = c[0].text_input("Activo", placeholder="Filtrar por activo: BTC, NVDA…", label_visibility="collapsed")
    horas = c[1].selectbox("Periodo", list(VENTANAS), index=2, format_func=VENTANAS.get, label_visibility="collapsed")
    limite = c[2].number_input("Máx. eventos", 20, 500, 100, 20, label_visibility="collapsed", help="Máximo de eventos a mostrar")
    en_vivo = c[3].toggle("En vivo", value=True, help="Actualiza el feed solo. Desactívalo para leer con calma.")
    seg = int(cfg("app", "autorefresco_centro_control_s", default=5))

    @st.fragment(run_every=f"{seg}s" if en_vivo else None)
    def feed() -> None:
        """Fragmento que se refresca solo cada `seg` segundos mientras está en vivo."""
        desde = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat()
        todos = leer_eventos(None, activo or None, desde, int(limite))
        eventos = [e for e in todos if e["channel"] in (canales or [])]

        estado = (f"<span class='cp-vivo'><span class='cp-vivo__punto'></span>En vivo · cada {seg} s</span>" if en_vivo
                  else "<span class='cp-vivo'><span class='cp-vivo__punto cp-vivo__punto--pausa'></span>En pausa</span>")
        st.html(f"<div style='display:flex;justify-content:space-between;align-items:center;margin-top:0.25rem'>{estado}"
                f"<span class='cp-evento__hora'>{len(eventos)} eventos · {ui.hora_local(datetime.now(timezone.utc).isoformat()):%H:%M:%S}</span></div>")

        conteo = Counter(e["channel"] for e in todos)
        ui.franja_stats([(ui.CANAL_DISCORD[k], str(conteo.get(k, 0)), VENTANAS[horas].lower()) for k in CANALES])

        if not eventos:
            st.caption("No hay eventos con estos filtros." if canales else "Elige al menos un canal.")
            return
        visto = int(st.session_state.get("cc_visto", 0))
        st.html(_feed_html(eventos, visto))
        st.session_state["cc_visto"] = max(visto, max(e["id"] for e in eventos))

    feed()
