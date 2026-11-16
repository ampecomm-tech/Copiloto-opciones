"""Pestaña 3: posiciones de opciones con P&L y griegas en vivo, griegas del portafolio, alertas e historial."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import auth, ui
from core import positions
from core.config import cfg
from core.events import leer_eventos
from quant.portfolio import griegas_portafolio


def render() -> None:
    """Dibuja la pestaña."""
    st.header("📊 Mis posiciones")
    usuario = auth.usuario_actual() or {}
    uid = usuario.get("user_id")
    c1, c2 = st.columns([1, 5])
    if c1.button("🔄 Refrescar P&L y griegas"):
        with st.spinner("Revaluando con la cadena actual…"):
            positions.refrescar_todas()
    c2.caption(f"El bot refresca automáticamente cada {cfg('posiciones', 'intervalo_refresco_min', default=5)} min.")

    abiertas = positions.listar(uid, "abierta")
    if not abiertas:
        st.info("No tienes posiciones abiertas. Confirma una estrategia desde el Copiloto.")
    else:
        spots = {p["asset"]: (p["greeks"] or {}).get("spot") for p in abiertas}
        filas = []
        for p in abiertas:
            g = p["greeks"] or {}
            valor = p.get("last_value") if p.get("last_value") is not None else p["entry_cost"]
            filas.append({"ID": p["id"], "Activo": p["asset"], "Estrategia": p["nombre"], "Venue": p.get("venue", "paper"), "Vence": p["expiry"], "Costo": round(p["entry_cost"], 2),
                          "Valor actual": round(valor, 2), "P&L": round(valor - p["entry_cost"], 2), "Máx. pérdida": round(p["max_perdida"], 2),
                          "Delta": round(g.get("delta", 0), 3), "Theta/día": round(g.get("theta", 0), 2), "Vega": round(g.get("vega", 0), 2),
                          "Actualizado": (p.get("last_update_ts") or "")[:16]})
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

        st.subheader("Griegas del portafolio")
        from data_sources.market import cadena_para  # importación local: solo si hay posiciones
        spots = {}
        for a in {p["asset"] for p in abiertas}:
            try:
                _, S, _ = cadena_para(a)
                spots[a] = S
            except Exception:
                pass
        gp = griegas_portafolio(abiertas, spots, float(cfg("app", "tasa_libre_riesgo", default=0.045)))
        m = st.columns(4)
        m[0].metric("Delta en dólares", f"{gp['delta_dolares']:+,.0f}")
        m[1].metric("Theta diaria", f"{gp['theta_diaria']:+,.2f}")
        m[2].metric("Vega total (por 1%)", f"{gp['vega_total']:+,.2f}")
        m[3].metric("Gamma $ (por 1%)", f"{gp['gamma_dolares']:+,.2f}")
        if gp["por_activo"]:
            st.dataframe(pd.DataFrame(gp["por_activo"]).T.round(2), use_container_width=True)

        st.subheader("Cerrar posición")
        cc = st.columns([1, 1, 2])
        pid = cc[0].selectbox("Posición", [p["id"] for p in abiertas])
        pos = next(p for p in abiertas if p["id"] == pid)
        valor_cierre = cc[1].number_input("Valor de cierre (USD)", value=float(pos.get("last_value") or pos["entry_cost"]))
        if cc[2].button("Cerrar en paper", key="cerrar"):
            positions.cerrar_posicion(pid, valor_cierre, uid)
            st.rerun()

    st.subheader("🔔 Alertas recientes")
    for ev in leer_eventos("alertas", limite=8):
        ui.tarjeta_evento(ev)

    st.subheader("Historial del copiloto")
    h = positions.historial_copiloto(uid)
    m = st.columns(4)
    m[0].metric("Cerradas", h["cerradas"])
    m[1].metric("Tasa de acierto", f"{h['tasa_acierto']:.0%}" if h["tasa_acierto"] is not None else "—")
    m[2].metric("P&L realizado", f"{h['pnl_realizado']:+,.2f}")
    m[3].metric("P&L abierto", f"{h['pnl_abierto']:+,.2f}")
    cerradas = positions.listar(uid, "cerrada")
    if cerradas:
        st.dataframe(pd.DataFrame([{"ID": p["id"], "Activo": p["asset"], "Estrategia": p["nombre"], "Abierta": p["entry_ts"][:10], "Cerrada": (p.get("exit_ts") or "")[:10],
                                    "Costo": round(p["entry_cost"], 2), "Salida": round(p.get("exit_value") or 0, 2), "P&L": round(p.get("pnl") or 0, 2)} for p in cerradas]),
                     hide_index=True, use_container_width=True)
