"""Pestaña 5: Centro de control — feed en vivo de `events` con filtros y autorefresco."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import streamlit as st

from app import ui
from core.config import cfg
from core.events import CANALES, leer_eventos

NOMBRES = {"mesa": "#mesa-agentes", "sentimiento": "#sentimiento-x", "bot_senales": "#bot-señales", "bot_operaciones": "#bot-operaciones", "alertas": "#alertas-posiciones", "sistema": "#sistema"}


def render() -> None:
    """Dibuja la pestaña."""
    st.header("🛰️ Centro de control")
    c = st.columns([2, 1, 1, 1])
    canales = c[0].multiselect("Canales", list(CANALES), default=list(CANALES), format_func=lambda k: NOMBRES[k])
    activo = c[1].text_input("Activo", placeholder="BTC, NVDA…")
    horas = c[2].selectbox("Desde", [1, 6, 24, 72, 24 * 30], index=2, format_func=lambda h: f"últimas {h} h" if h < 48 else f"últimos {h // 24} días")
    limite = c[3].number_input("Máx. eventos", 20, 500, 100, 20)
    seg = int(cfg("app", "autorefresco_centro_control_s", default=5))

    @st.fragment(run_every=f"{seg}s")
    def feed() -> None:
        """Fragmento que se refresca solo cada `seg` segundos."""
        desde = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat()
        eventos = [e for e in leer_eventos(None, activo or None, desde, int(limite)) if e["channel"] in canales]
        st.caption(f"{len(eventos)} eventos · actualizado {datetime.now().strftime('%H:%M:%S')} · autorefresco cada {seg} s")
        for ev in eventos:
            ui.tarjeta_evento(ev)
        if not eventos:
            st.info("Sin eventos con esos filtros.")

    feed()
