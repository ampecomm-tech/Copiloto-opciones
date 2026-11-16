"""Componentes visuales reutilizables de la app (Plotly y pequeños widgets)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.strategies_options import curva_payoff

COLORES = {"verde": "#2ECC71", "amarillo": "#F1C40F", "rojo": "#E74C3C", "info": "#95A5A6", "ok": "#2ECC71", "warning": "#F1C40F", "error": "#E74C3C",
           "senal": "#3498DB", "agente": "#9B59B6", "entrada": "#2ECC71", "salida": "#E74C3C", "alerta": "#F1C40F"}
EMOJI = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}


def semaforo_badge(color: str, texto: str = "") -> str:
    """Markdown con el círculo de color del semáforo."""
    return f"{EMOJI.get(color, '⚪')} **{color.upper()}** {texto}"


def grafica_payoff(estrategia: dict[str, Any], r: float, dias: float = 0.0, iv_shift: float = 0.0, spot_whatif: float | None = None) -> go.Figure:
    """Payoff al vencimiento y curva actual (Black-Scholes), con breakevens y spot."""
    c = curva_payoff(estrategia, r, dias, iv_shift)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["precios"], y=c["vencimiento"], name="Al vencimiento", line=dict(color="#2C3E50", width=2)))
    fig.add_trace(go.Scatter(x=c["precios"], y=c["actual"], name=f"Hoy +{dias:.0f}d (BS)", line=dict(color="#3498DB", width=2, dash="dot")))
    fig.add_hline(y=0, line=dict(color="#7F8C8D", width=1))
    fig.add_vline(x=estrategia["spot"], line=dict(color="#95A5A6", dash="dash"), annotation_text="Spot")
    if spot_whatif:
        fig.add_vline(x=spot_whatif, line=dict(color="#E67E22", dash="dash"), annotation_text="What-if")
    for be in estrategia["breakevens"]:
        fig.add_vline(x=be, line=dict(color="#E74C3C", width=1, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Precio del subyacente", yaxis_title="P&L (USD)", legend=dict(orientation="h"))
    return fig


def grafica_histograma(mc: dict[str, Any]) -> go.Figure:
    """Histograma del P&L simulado."""
    h = mc.get("histograma", {"x": [], "y": []})
    colores = ["#E74C3C" if x < 0 else "#2ECC71" for x in h["x"]]
    fig = go.Figure(go.Bar(x=h["x"], y=h["y"], marker_color=colores))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="P&L al horizonte (USD)", yaxis_title="Trayectorias", title=f"Monte Carlo ({mc.get('n', 0):,} trayectorias)")
    return fig


def tabla_griegas(g: dict[str, float]) -> pd.DataFrame:
    """Griegas en tabla de una fila."""
    return pd.DataFrame([{k.capitalize(): round(v, 4) for k, v in g.items()}])


def tarjeta_evento(ev: dict[str, Any]) -> None:
    """Un evento del bus en estilo chat."""
    color = COLORES.get(ev.get("event_type", "info"), "#95A5A6")
    with st.container(border=True):
        st.markdown(f"<span style='color:{color}'>●</span> **{ev['title']}**  \n<small>{ev['ts'][:19].replace('T', ' ')} · `{ev['channel']}` · {ev['event_type']}</small>", unsafe_allow_html=True)
        p = ev.get("payload") or {}
        if p:
            with st.expander("Detalles", expanded=False):
                st.json(p, expanded=1)


def grafica_equity(trades: pd.DataFrame, capital_inicial: float) -> go.Figure:
    """Curva de capital a partir de los trades cerrados."""
    fig = go.Figure()
    if not trades.empty:
        t = trades.sort_values("exit_ts")
        eq = capital_inicial + t["pnl"].cumsum()
        fig.add_trace(go.Scatter(x=t["exit_ts"], y=eq, mode="lines+markers", name="Capital"))
    fig.add_hline(y=capital_inicial, line=dict(dash="dot", color="#95A5A6"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="USD")
    return fig


def grafica_velas(df: pd.DataFrame, titulo: str = "") -> go.Figure:
    """Gráfica de velas OHLC."""
    fig = go.Figure(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title=titulo, xaxis_rangeslider_visible=False)
    return fig
