"""Pestaña 3: posiciones de opciones con P&L y griegas en vivo, griegas del portafolio, alertas e historial."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import auth, ui
from core import positions
from core.config import cfg
from core.events import leer_eventos
from quant.portfolio import griegas_portafolio

NOMBRE_VENUE = {"deribit_testnet": "Deribit testnet", "binance_options_testnet": "Binance Options testnet",
                "alpaca_paper": "Alpaca Paper", "paper": "Paper interno"}


def _color_pnl(x: float) -> str:
    """Estilo CSS para una celda de P&L: verde si gana, rojo si pierde."""
    if not isinstance(x, (int, float)) or x == 0:
        return ""
    return f"color: {ui.TOKENS['verde'] if x > 0 else ui.TOKENS['rojo']}"


def _tono(x: float) -> str:
    """Tono de pastilla según el signo."""
    return "verde" if x > 0 else "rojo" if x < 0 else "gris"


def _usd(x: float, signo: bool = True) -> str:
    """Formato de dólares con signo opcional."""
    return f"{x:+,.2f}" if signo else f"{x:,.2f}"


def render() -> None:
    """Dibuja la pestaña."""
    usuario = auth.usuario_actual() or {}
    uid = usuario.get("user_id")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    c1.title("Mis posiciones")
    c1.caption(f"P&L y griegas revaluados con la cadena actual. El bot los refresca cada {cfg('posiciones', 'intervalo_refresco_min', default=5)} min.")
    if c2.button("Refrescar ahora", icon=":material/refresh:", use_container_width=True):
        with st.spinner("Revaluando con la cadena actual…"):
            positions.refrescar_todas()

    abiertas = positions.listar(uid, "abierta")
    h = positions.historial_copiloto(uid)

    if not abiertas:
        st.space("small")
        with st.container(border=True):
            st.html("<div class='cp-titulo__texto' style='font-size:1.05rem'>No tienes posiciones abiertas</div>"
                    "<div class='cp-texto' style='font-size:0.875rem;color:#A3ACBA'>Analiza una idea en el Copiloto de opciones y confirma la estrategia que te guste; aparecerá aquí con su P&L y sus griegas.</div>")
    else:
        from data_sources.market import cadena_para  # importación local: solo si hay posiciones
        spots = {}
        for a in {p["asset"] for p in abiertas}:
            try:
                _, S, _ = cadena_para(a)
                spots[a] = S
            except Exception:
                pass
        gp = griegas_portafolio(abiertas, spots, float(cfg("app", "tasa_libre_riesgo", default=0.045)))

        st.space("small")
        ui.franja_stats([
            ("Posiciones abiertas", str(len(abiertas)), ", ".join(sorted({p["asset"] for p in abiertas}))),
            ("P&L abierto (USD)", ui.pill(_usd(h["pnl_abierto"]), _tono(h["pnl_abierto"])), "sin realizar"),
            ("Delta en dólares", f"{gp['delta_dolares']:+,.0f}", "exposición al precio"),
            ("Theta diaria", f"{gp['theta_diaria']:+,.2f}", "USD por día"),
            ("Vega total", f"{gp['vega_total']:+,.2f}", "USD por 1% de IV"),
            ("Gamma $", f"{gp['gamma_dolares']:+,.2f}", "por 1% de movimiento"),
        ])

        st.subheader("Posiciones abiertas")
        filas = []
        for p in abiertas:
            g = p["greeks"] or {}
            valor = p.get("last_value") if p.get("last_value") is not None else p["entry_cost"]
            act = p.get("last_update_ts")
            filas.append({"ID": p["id"], "Activo": p["asset"], "Estrategia": p["nombre"], "Venue": NOMBRE_VENUE.get(p.get("venue", "paper"), p.get("venue")),
                          "Vence": p["expiry"], "Costo": p["entry_cost"], "Valor actual": valor, "P&L": valor - p["entry_cost"],
                          "Pérdida máx.": p["max_perdida"], "Delta": g.get("delta", 0.0), "Theta/día": g.get("theta", 0.0), "Vega": g.get("vega", 0.0),
                          "Actualizado": f"{ui.hora_local(act):%d/%m %H:%M}" if act else "—"})
        df = pd.DataFrame(filas)
        st.dataframe(
            df.style.map(_color_pnl, subset=["P&L"]),
            hide_index=True, use_container_width=True,
            column_config={
                "ID": st.column_config.NumberColumn(width="small", format="%d"),
                "Costo": st.column_config.NumberColumn(format="%.2f"),
                "Valor actual": st.column_config.NumberColumn(format="%.2f"),
                "P&L": st.column_config.NumberColumn(format="%+.2f"),
                "Pérdida máx.": st.column_config.NumberColumn(format="%.2f"),
                "Delta": st.column_config.NumberColumn(format="%+.3f"),
                "Theta/día": st.column_config.NumberColumn(format="%+.2f"),
                "Vega": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        if gp["por_activo"]:
            with st.expander("Griegas por activo", icon=":material/table_chart:"):
                st.dataframe(pd.DataFrame(gp["por_activo"]).T.round(2), use_container_width=True)

        with st.expander("Cerrar una posición", icon=":material/close:"):
            cc = st.columns([1, 1, 1], vertical_alignment="bottom")
            pid = cc[0].selectbox("Posición", [p["id"] for p in abiertas],
                                  format_func=lambda i: next(f"#{p['id']} · {p['asset']} · {p['nombre']}" for p in abiertas if p["id"] == i))
            pos = next(p for p in abiertas if p["id"] == pid)
            valor_cierre = cc[1].number_input("Valor de cierre (USD)", value=float(pos.get("last_value") or pos["entry_cost"]))
            if cc[2].button("Cerrar en paper", key="cerrar", use_container_width=True):
                positions.cerrar_posicion(pid, valor_cierre, uid)
                st.rerun()
            st.caption("El cierre se registra en paper con el valor indicado (por defecto, el último valor revaluado).")

    st.space("medium")
    st.subheader("Historial del copiloto")
    st.caption("Lo que recomendó el copiloto contra lo que pasó de verdad.")
    ui.franja_stats([
        ("Cerradas", str(h["cerradas"]), ""),
        ("Tasa de acierto", f"{h['tasa_acierto']:.0%}" if h["tasa_acierto"] is not None else "—", "posiciones con P&L > 0"),
        ("P&L realizado (USD)", ui.pill(_usd(h["pnl_realizado"]), _tono(h["pnl_realizado"])), ""),
        ("P&L abierto (USD)", ui.pill(_usd(h["pnl_abierto"]), _tono(h["pnl_abierto"])), ""),
    ])
    cerradas = positions.listar(uid, "cerrada")
    if cerradas:
        dfc = pd.DataFrame([{"ID": p["id"], "Activo": p["asset"], "Estrategia": p["nombre"], "Abierta": p["entry_ts"][:10], "Cerrada": (p.get("exit_ts") or "")[:10],
                             "Costo": p["entry_cost"], "Salida": p.get("exit_value") or 0.0, "P&L": p.get("pnl") or 0.0} for p in cerradas])
        st.dataframe(dfc.style.map(_color_pnl, subset=["P&L"]), hide_index=True, use_container_width=True,
                     column_config={"Costo": st.column_config.NumberColumn(format="%.2f"), "Salida": st.column_config.NumberColumn(format="%.2f"),
                                    "P&L": st.column_config.NumberColumn(format="%+.2f")})

    st.space("medium")
    st.subheader("Alertas recientes")
    st.caption("Cobertura de delta, toma de ganancia y vencimientos. También llegan a #alertas-posiciones en Discord.")
    alertas = leer_eventos("alertas", limite=8)
    if alertas:
        st.html("".join(ui.html_evento(ev) for ev in alertas))
    else:
        st.caption("Sin alertas.")
