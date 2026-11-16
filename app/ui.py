"""Componentes visuales reutilizables de la app: sistema de diseño (tokens + CSS global), Plotly y widgets."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.strategies_options import curva_payoff

# --- Sistema de diseño -------------------------------------------------------------------------
# Mismos valores que `.streamlit/config.toml`. Un solo acento (azul); verde/rojo/amarillo solo
# para P&L, semáforo y estados.
TOKENS: dict[str, str] = {
    "fondo": "#0B0E13",
    "superficie": "#141922",
    "superficie_2": "#1A202B",
    "borde": "#232A35",
    "texto": "#E7EAEE",
    "texto_2": "#A3ACBA",
    "texto_3": "#6B7486",
    "acento": "#5B8CFF",
    "verde": "#3DD68C",
    "rojo": "#F2555A",
    "amarillo": "#F5C04A",
    "naranja": "#F59E4A",
    "violeta": "#A78BFA",
    "gris": "#8A94A6",
}

COLORES = {"verde": TOKENS["verde"], "amarillo": TOKENS["amarillo"], "rojo": TOKENS["rojo"], "info": TOKENS["gris"], "ok": TOKENS["verde"],
           "warning": TOKENS["amarillo"], "error": TOKENS["rojo"], "senal": TOKENS["acento"], "agente": TOKENS["violeta"],
           "entrada": TOKENS["verde"], "salida": TOKENS["rojo"], "alerta": TOKENS["amarillo"]}

_CSS = """
<style>
:root {
  --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
  --ease-in-out: cubic-bezier(0.65, 0, 0.35, 1);
}

/* Cabecera de Streamlit: sin la franja de color ni fondo propio */
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* Ancho de lectura cómodo y menos aire arriba */
[data-testid="stMainBlockContainer"] {
  padding-top: 2.25rem;
  padding-bottom: 4rem;
  max-width: 1360px;
}

/* Títulos: tracking ajustado y sin cortes feos */
h1, h2, h3 { letter-spacing: -0.015em; text-wrap: balance; }

/* Cifras: números tabulares en métricas, tablas y código */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stDataFrame"], [data-testid="stTable"], code {
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] p { color: #A3ACBA; font-size: 0.8125rem; font-weight: 500; }

/* Entrada de la página: fade + 4 px, ease-out, 220 ms */
[data-testid="stMainBlockContainer"] > div {
  animation: cp-entrada 220ms var(--ease-out) both;
}
@keyframes cp-entrada {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: none; }
}

/* Botones: transición corta y escala 0.97 al presionar */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button, .stLinkButton > a {
  font-weight: 500;
  transition: background-color 150ms var(--ease-out), border-color 150ms var(--ease-out),
              transform 120ms var(--ease-out), color 150ms var(--ease-out);
}
.stButton > button:active, .stFormSubmitButton > button:active,
.stDownloadButton > button:active, .stLinkButton > a:active {
  transform: scale(0.97);
}

/* Foco visible solo con teclado */
button:focus-visible, a:focus-visible {
  outline: 2px solid #5B8CFF;
  outline-offset: 2px;
}

/* Pestañas: subrayado con el acento */
.stTabs [data-baseweb="tab-highlight"] { background-color: #5B8CFF; }

/* Sidebar: navegación compacta */
[data-testid="stSidebarNav"] a { border-radius: 8px; transition: background-color 150ms var(--ease-out); }

/* Pie del sidebar */
.cp-usuario { font-size: 0.8125rem; color: #A3ACBA; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cp-pie { font-size: 0.75rem; color: #6B7486; }

/* Pastilla de estado (semáforo, postura, etiquetas) */
.cp-pill {
  display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.125rem 0.6rem;
  border-radius: 999px; font-size: 0.8125rem; font-weight: 500; line-height: 1.5;
  background: var(--cp-bg); color: var(--cp-fg); white-space: nowrap; vertical-align: middle;
}
.cp-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.cp-pill--sin-punto::before { display: none; }

/* Franja de contexto: datos clave en una línea */
.cp-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1px;
  background: #232A35; border: 1px solid #232A35; border-radius: 10px; overflow: hidden; }
.cp-stat { background: #11151C; padding: 0.7rem 0.9rem; min-width: 0; }
.cp-stat__label { font-size: 0.75rem; color: #8A94A6; margin-bottom: 0.15rem; }
.cp-stat__valor { font-size: 1.05rem; font-weight: 600; color: #E7EAEE; font-variant-numeric: tabular-nums;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cp-stat__sub { font-size: 0.75rem; color: #6B7486; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Tabla de patas de la estrategia */
.cp-patas { width: 100%; border-collapse: collapse; font-size: 0.875rem; font-variant-numeric: tabular-nums; }
.cp-patas th { text-align: left; font-weight: 500; font-size: 0.75rem; color: #8A94A6; padding: 0 0.75rem 0.4rem 0; }
.cp-patas td { padding: 0.45rem 0.75rem 0.45rem 0; border-top: 1px solid #1E2530; color: #E7EAEE; }
.cp-patas td.num, .cp-patas th.num { text-align: right; }
.cp-patas .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.8125rem; color: #A3ACBA; }
.cp-compra { color: #5FE3A3; font-weight: 500; }
.cp-venta { color: #FF7B7F; font-weight: 500; }

/* Encabezado de tarjeta: título + pastillas */
.cp-titulo { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.75rem; }
.cp-titulo__texto { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.015em; color: #E7EAEE; }
.cp-eyebrow { font-size: 0.75rem; font-weight: 500; color: #8A94A6; margin-bottom: 0.25rem; }
.cp-texto { font-size: 0.9375rem; line-height: 1.6; color: #C9D0DA; }
.cp-modelo { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.75rem; color: #6B7486; }

/* Barra de confianza */
.cp-barra { height: 4px; border-radius: 2px; background: #232A35; overflow: hidden; margin: 0.35rem 0 0.15rem; }
.cp-barra > span { display: block; height: 100%; background: var(--cp-fg); border-radius: 2px;
  transform-origin: left; animation: cp-crecer 500ms var(--ease-out) both; }
@keyframes cp-crecer { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* Listas de razones / advertencias */
.cp-lista { list-style: none; padding: 0; margin: 0; display: grid; gap: 0.45rem; }
.cp-lista li { display: grid; grid-template-columns: 1rem 1fr; gap: 0.5rem; font-size: 0.875rem; line-height: 1.5; color: #C9D0DA; }
.cp-lista li::before { content: ""; width: 6px; height: 6px; border-radius: 50%; margin-top: 0.5rem; justify-self: center; background: var(--cp-fg); }

/* Feed de eventos (Centro de control y alertas) */
.cp-evento { position: relative; display: grid; grid-template-columns: 3px 1fr; gap: 0 0.9rem;
  padding: 0.7rem 0.9rem 0.7rem 0; border-bottom: 1px solid #1A202B; }
.cp-evento__barra { border-radius: 2px; background: var(--cp-fg); opacity: 0.85; }
.cp-evento__cab { display: flex; align-items: baseline; gap: 0.6rem; min-width: 0; }
.cp-evento__titulo { font-size: 0.9rem; font-weight: 500; color: #E7EAEE; flex: 1; min-width: 0; line-height: 1.45; }
.cp-evento__hora { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.75rem; color: #6B7486; white-space: nowrap; }
.cp-evento__meta { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.2rem; font-size: 0.75rem; color: #6B7486; }
.cp-canal { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.72rem; color: #8A94A6;
  background: #161C26; border-radius: 4px; padding: 0.05rem 0.4rem; }
.cp-evento__texto { margin-top: 0.4rem; font-size: 0.8125rem; line-height: 1.55; color: #A3ACBA;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.cp-evento details { margin-top: 0.4rem; }
.cp-evento summary { cursor: pointer; font-size: 0.75rem; color: #6B7486; list-style: none; width: fit-content;
  transition: color 150ms var(--ease-out); }
.cp-evento summary::-webkit-details-marker { display: none; }
.cp-evento summary::before { content: "+ "; }
.cp-evento details[open] summary::before { content: "− "; }
.cp-evento summary:hover { color: #E7EAEE; }
.cp-kv { display: grid; grid-template-columns: minmax(110px, max-content) 1fr; gap: 0.3rem 1rem; margin-top: 0.5rem;
  padding: 0.7rem 0.8rem; background: #11151C; border-radius: 8px; font-size: 0.8rem; }
.cp-kv dt { color: #6B7486; }
.cp-kv dd { margin: 0; color: #C9D0DA; word-break: break-word; font-variant-numeric: tabular-nums; }
.cp-kv dd.mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.75rem; white-space: pre-wrap; }
.cp-evento--nuevo { animation: cp-nuevo 900ms var(--ease-out) both; }
@keyframes cp-nuevo {
  0%   { opacity: 0; transform: translateY(-4px); background: rgba(91, 140, 255, 0.10); }
  35%  { opacity: 1; transform: none; background: rgba(91, 140, 255, 0.10); }
  100% { background: transparent; }
}
.cp-dia { font-size: 0.75rem; font-weight: 500; color: #8A94A6; padding: 1rem 0 0.35rem; border-bottom: 1px solid #232A35; }

/* Titulares */
.cp-titular { display: grid; grid-template-columns: 8px 1fr; gap: 0 0.8rem; padding: 0.65rem 0; border-bottom: 1px solid #1A202B; }
.cp-titular:last-child { border-bottom: none; }
.cp-titular__punto { width: 7px; height: 7px; border-radius: 50%; margin-top: 0.5rem; background: var(--cp-fg); }
.cp-titular a { color: #E7EAEE; text-decoration: none; font-size: 0.9rem; font-weight: 500; line-height: 1.45;
  transition: color 150ms var(--ease-out); }
.cp-titular a:hover { color: #8FB0FF; }
.cp-titular__meta { font-size: 0.75rem; color: #6B7486; margin-top: 0.15rem; }

/* Medidor de sentimiento (-1 a 1) */
.cp-medidor { margin: 0.5rem 0 0.35rem; }
.cp-medidor__pista { position: relative; height: 6px; border-radius: 3px;
  background: linear-gradient(90deg, rgba(242,85,90,0.55), rgba(138,148,166,0.35) 50%, rgba(61,214,140,0.55)); }
.cp-medidor__marca { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #E7EAEE;
  border: 3px solid #0B0E13; transform: translate(-50%, -50%); animation: cp-entrada 300ms var(--ease-out) both; }
.cp-medidor__escala { display: flex; justify-content: space-between; font-size: 0.6875rem; color: #6B7486; margin-top: 0.35rem; }
.cp-score { font-size: 2.25rem; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.1; }

/* Próximos eventos */
.cp-agenda { display: grid; gap: 0.5rem; }
.cp-agenda__fila { display: grid; grid-template-columns: 5.5rem 1fr; gap: 0.75rem; font-size: 0.85rem; line-height: 1.45; color: #C9D0DA; }
.cp-agenda__fecha { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 0.75rem; color: #8FB0FF; padding-top: 0.1rem; }

/* Indicador "en vivo" */
.cp-vivo { display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.8125rem; color: #A3ACBA; }
.cp-vivo__punto { width: 8px; height: 8px; border-radius: 50%; background: #3DD68C; box-shadow: 0 0 0 0 rgba(61, 214, 140, 0.5);
  animation: cp-latido 2s ease-out infinite; }
.cp-vivo__punto--pausa { background: #6B7486; animation: none; }
@keyframes cp-latido { 0% { box-shadow: 0 0 0 0 rgba(61, 214, 140, 0.45); } 70%, 100% { box-shadow: 0 0 0 6px rgba(61, 214, 140, 0); } }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def aplicar_estilo() -> None:
    """Inyecta el CSS global de la app (cifras tabulares, animaciones discretas, sidebar, marca)."""
    st.html(_CSS)


def estilo_figura(fig: go.Figure, height: int, **layout: Any) -> go.Figure:
    """Aplica el estilo común a una figura de Plotly: fondo transparente, rejilla sutil, leyenda horizontal.

    Los argumentos extra (`**layout`) se aplican al final y pueden sobrescribir la base (p. ej. `margin`).
    """
    eje = dict(gridcolor=TOKENS["borde"], zerolinecolor=TOKENS["borde"], linecolor=TOKENS["borde"],
               tickfont=dict(color=TOKENS["texto_3"], size=11), title_font=dict(color=TOKENS["texto_2"], size=12))
    base: dict[str, Any] = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TOKENS["texto_2"], size=12),
        title_font=dict(color=TOKENS["texto"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor=TOKENS["superficie_2"], bordercolor=TOKENS["borde"], font=dict(color=TOKENS["texto"])),
        xaxis=eje,
        yaxis=eje,
    )
    base.update(layout)
    fig.update_layout(**base)
    return fig


# --- Widgets -----------------------------------------------------------------------------------

_TONOS: dict[str, tuple[str, str]] = {
    # tono -> (texto, fondo)
    "verde": ("#5FE3A3", "rgba(61, 214, 140, 0.12)"),
    "amarillo": ("#F7CD6B", "rgba(245, 192, 74, 0.12)"),
    "rojo": ("#FF7B7F", "rgba(242, 85, 90, 0.12)"),
    "acento": ("#8FB0FF", "rgba(91, 140, 255, 0.14)"),
    "gris": ("#A3ACBA", "rgba(138, 148, 166, 0.14)"),
}
TONO_POSTURA = {"favorable": "verde", "neutral": "gris", "desfavorable": "rojo"}
NOMBRE_SEMAFORO = {"verde": "Verde", "amarillo": "Amarillo", "rojo": "Rojo"}
COLOR_MD = {"verde": "green", "amarillo": "yellow", "rojo": "red"}
"""Color de Streamlit Markdown (`:green[...]`) para cada color del semáforo."""


def esc(texto: Any) -> str:
    """Escapa texto para insertarlo en HTML (las respuestas de los LLM pueden traer `<`, `$`, etc.)."""
    return html.escape(str(texto))


def pill(texto: str, tono: str = "gris", punto: bool = True) -> str:
    """HTML de una pastilla de estado con el tono dado (verde, amarillo, rojo, acento, gris)."""
    fg, bg = _TONOS.get(tono, _TONOS["gris"])
    clase = "cp-pill" if punto else "cp-pill cp-pill--sin-punto"
    return f"<span class='{clase}' style='--cp-fg:{fg};--cp-bg:{bg}'>{esc(texto)}</span>"


def semaforo_pill(color: str) -> str:
    """Pastilla del semáforo de viabilidad."""
    return pill(NOMBRE_SEMAFORO.get(color, color.capitalize()), color if color in _TONOS else "gris")


def franja_stats(items: list[tuple[str, str, str]]) -> None:
    """Franja horizontal de datos clave. Cada item es `(etiqueta, valor, subtexto)`; el valor puede traer HTML ya escapado."""
    celdas = "".join(f"<div class='cp-stat'><div class='cp-stat__label'>{esc(e)}</div><div class='cp-stat__valor'>{v}</div>"
                     + (f"<div class='cp-stat__sub' title='{esc(s)}'>{esc(s)}</div>" if s else "") + "</div>" for e, v, s in items)
    st.html(f"<div class='cp-stats'>{celdas}</div>")


def tabla_patas(legs: list[dict[str, Any]]) -> None:
    """Tabla de las patas de una estrategia con el precio real de ejecución (ask al comprar, bid al vender)."""
    filas = []
    for l in legs:
        compra = l["side"] > 0
        filas.append(
            "<tr>"
            f"<td class='{'cp-compra' if compra else 'cp-venta'}'>{'Compra' if compra else 'Venta'}</td>"
            f"<td>{esc(str(l['type']).capitalize())}</td>"
            f"<td class='num'>{l['strike']:,.0f}</td>"
            f"<td class='num'>{l['qty']:g}</td>"
            f"<td class='num'>{l['price']:,.2f} <span class='mono'>{'ask' if compra else 'bid'}</span></td>"
            f"<td class='mono'>{esc(l.get('symbol', ''))}</td>"
            "</tr>")
    st.html("<table class='cp-patas'><thead><tr><th>Lado</th><th>Tipo</th><th class='num'>Strike</th><th class='num'>Cantidad</th>"
            f"<th class='num'>Precio</th><th>Contrato</th></tr></thead><tbody>{''.join(filas)}</tbody></table>")


def barra_confianza(valor: float, tono: str = "acento") -> str:
    """HTML de una barra horizontal (0–1) que crece al aparecer."""
    fg, _ = _TONOS.get(tono, _TONOS["acento"])
    v = max(0.0, min(1.0, float(valor)))
    return f"<div class='cp-barra' style='--cp-fg:{fg}'><span style='width:{v:.0%}'></span></div>"


def lista_puntos(items: list[str], tono: str) -> str:
    """HTML de una lista con viñeta de color (razones, advertencias, riesgos)."""
    fg, _ = _TONOS.get(tono, _TONOS["gris"])
    return f"<ul class='cp-lista' style='--cp-fg:{fg}'>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>"


def grafica_payoff(estrategia: dict[str, Any], r: float, dias: float = 0.0, iv_shift: float = 0.0, spot_whatif: float | None = None) -> go.Figure:
    """Payoff al vencimiento y curva actual (Black-Scholes), con breakevens y spot."""
    c = curva_payoff(estrategia, r, dias, iv_shift)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["precios"], y=c["vencimiento"], name="Al vencimiento", line=dict(color=TOKENS["texto"], width=2)))
    fig.add_trace(go.Scatter(x=c["precios"], y=c["actual"], name=f"Hoy +{dias:.0f}d (BS)", line=dict(color=TOKENS["acento"], width=2, dash="dot")))
    fig.add_hline(y=0, line=dict(color=TOKENS["texto_3"], width=1))
    fig.add_vline(x=estrategia["spot"], line=dict(color=TOKENS["gris"], dash="dash", width=1), annotation_text="Spot", annotation_font_color=TOKENS["texto_2"])
    if spot_whatif and abs(spot_whatif / estrategia["spot"] - 1) > 0.002:
        fig.add_vline(x=spot_whatif, line=dict(color=TOKENS["naranja"], dash="dash", width=1), annotation_text="What-if", annotation_font_color=TOKENS["naranja"])
    for be in estrategia["breakevens"]:
        fig.add_vline(x=be, line=dict(color=TOKENS["rojo"], width=1, dash="dot"))
    return estilo_figura(fig, 360, xaxis_title="Precio del subyacente", yaxis_title="P&L (USD)")


def grafica_histograma(mc: dict[str, Any]) -> go.Figure:
    """Histograma del P&L simulado."""
    h = mc.get("histograma", {"x": [], "y": []})
    colores = [TOKENS["rojo"] if x < 0 else TOKENS["verde"] for x in h["x"]]
    fig = go.Figure(go.Bar(x=h["x"], y=h["y"], marker_color=colores, marker_line_width=0))
    return estilo_figura(fig, 260, xaxis_title="P&L al horizonte (USD)", yaxis_title="Trayectorias",
                         title=f"Monte Carlo ({mc.get('n', 0):,} trayectorias)", bargap=0.05)


CANAL_DISCORD = {"mesa": "#mesa-agentes", "sentimiento": "#sentimiento-x", "bot_senales": "#bot-señales",
                 "bot_operaciones": "#bot-operaciones", "alertas": "#alertas-posiciones", "sistema": "#sistema"}
TONO_EVENTO = {"verde": "verde", "ok": "verde", "entrada": "verde", "amarillo": "amarillo", "warning": "amarillo", "alerta": "amarillo",
               "rojo": "rojo", "error": "rojo", "salida": "rojo", "senal": "acento", "agente": "acento", "info": "gris"}
NOMBRE_TIPO = {"verde": "verde", "ok": "ok", "entrada": "entrada", "amarillo": "amarillo", "warning": "aviso", "alerta": "alerta",
               "rojo": "rojo", "error": "error", "salida": "salida", "senal": "señal", "agente": "agente", "info": "info"}
_CLAVES_RESUMEN = ("justificacion", "argumento", "explicacion", "resumen", "razon", "motivo", "sugerencia", "detalle", "error")


def hora_local(ts: str) -> datetime:
    """Convierte un timestamp ISO (UTC) a la zona horaria de la app (`app.zona_horaria`, por defecto Ciudad de México)."""
    from core.config import cfg

    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(ZoneInfo(cfg("app", "zona_horaria", default="America/Mexico_City")))
    except Exception:
        return dt


def _valor_kv(v: Any) -> tuple[str, bool]:
    """Formatea un valor del payload para la lista de detalles. Devuelve (html, es_monoespaciado)."""
    if isinstance(v, float):
        return esc(f"{v:,.4f}".rstrip("0").rstrip(".")), False
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return esc(" · ".join(str(x) for x in v)), False
    if isinstance(v, (dict, list)):
        return esc(json.dumps(v, ensure_ascii=False, indent=1, default=str)), True
    return esc(v), False


def html_evento(ev: dict[str, Any], nuevo: bool = False) -> str:
    """HTML de una fila del feed: barra de color por tipo, título, canal, hora, resumen y detalles plegables."""
    tono = TONO_EVENTO.get(ev.get("event_type", "info"), "gris")
    fg, _ = _TONOS[tono]
    p = ev.get("payload") or {}
    resumen = next((str(p[k]) for k in _CLAVES_RESUMEN if p.get(k)), "")
    hora = hora_local(ev["ts"])
    detalles = ""
    if p:
        filas = []
        for k, v in p.items():
            val, mono = _valor_kv(v)
            filas.append(f"<dt>{esc(k.replace('_', ' '))}</dt><dd{' class=mono' if mono else ''}>{val}</dd>")
        detalles = f"<details><summary>Detalles</summary><dl class='cp-kv'>{''.join(filas)}</dl></details>"
    return (f"<div class='cp-evento{' cp-evento--nuevo' if nuevo else ''}' style='--cp-fg:{fg}'>"
            "<div class='cp-evento__barra'></div><div style='min-width:0'>"
            f"<div class='cp-evento__cab'><span class='cp-evento__titulo'>{esc(ev['title'])}</span>"
            f"<span class='cp-evento__hora' title='{esc(ev['ts'])}'>{hora:%H:%M:%S}</span></div>"
            f"<div class='cp-evento__meta'><span class='cp-canal'>{esc(CANAL_DISCORD.get(ev['channel'], ev['channel']))}</span>"
            f"<span>{esc(NOMBRE_TIPO.get(ev.get('event_type', ''), ev.get('event_type', '')))}</span></div>"
            + (f"<div class='cp-evento__texto'>{esc(resumen)}</div>" if resumen else "")
            + detalles + "</div></div>")


def tarjeta_evento(ev: dict[str, Any], nuevo: bool = False) -> None:
    """Dibuja un evento del bus como fila del feed."""
    st.html(html_evento(ev, nuevo))


TONO_TITULAR = {"positivo": "verde", "negativo": "rojo", "neutral": "gris"}


def lista_titulares(titulares: list[dict[str, Any]]) -> None:
    """Titulares con punto de color según el sentimiento que asignó Gemini (o el léxico), fuente y fecha."""
    filas = []
    for t in titulares:
        fg, _ = _TONOS[TONO_TITULAR.get(t.get("sentimiento", "neutral"), "gris")]
        url = esc(t.get("url") or "#")
        filas.append(f"<div class='cp-titular' style='--cp-fg:{fg}'><span class='cp-titular__punto' title='{esc(t.get('sentimiento', ''))}'></span><div>"
                     f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{esc(t['titulo'])}</a>"
                     f"<div class='cp-titular__meta'>{esc(t.get('fuente', ''))} · {esc(t.get('fecha', ''))}</div></div></div>")
    st.html("".join(filas))


def medidor_sentimiento(score: float) -> str:
    """HTML del puntaje de sentimiento grande y una pista de -1 (rojo) a 1 (verde) con la marca en su posición."""
    s = max(-1.0, min(1.0, float(score)))
    tono = "verde" if s > 0.15 else "rojo" if s < -0.15 else "gris"
    fg, _ = _TONOS[tono]
    etiqueta = "Alcista" if s > 0.15 else "Bajista" if s < -0.15 else "Neutral"
    return (f"<div style='display:flex;align-items:baseline;gap:0.75rem'><span class='cp-score' style='color:{fg}'>{s:+.2f}</span>"
            f"{pill(etiqueta, tono)}</div>"
            f"<div class='cp-medidor'><div class='cp-medidor__pista'><span class='cp-medidor__marca' style='left:{(s + 1) / 2:.1%}'></span></div>"
            "<div class='cp-medidor__escala'><span>-1 bajista</span><span>0</span><span>+1 alcista</span></div></div>")


def agenda(eventos: list[dict[str, Any]]) -> str:
    """HTML de la lista de próximos eventos (fecha en monoespaciada + descripción)."""
    return "<div class='cp-agenda'>" + "".join(
        f"<div class='cp-agenda__fila'><span class='cp-agenda__fecha'>{esc(e.get('fecha', ''))}</span><span>{esc(e.get('descripcion', ''))}</span></div>"
        for e in eventos) + "</div>"


def grafica_equity(trades: pd.DataFrame, capital_inicial: float) -> go.Figure:
    """Curva de capital a partir de los trades cerrados."""
    fig = go.Figure()
    if not trades.empty:
        t = trades.sort_values("exit_ts")
        eq = capital_inicial + t["pnl"].cumsum()
        fig.add_trace(go.Scatter(x=t["exit_ts"], y=eq, mode="lines+markers", name="Capital",
                                 line=dict(color=TOKENS["acento"], width=2), marker=dict(size=5)))
    fig.add_hline(y=capital_inicial, line=dict(dash="dot", color=TOKENS["texto_3"], width=1))
    return estilo_figura(fig, 280, yaxis_title="USD", margin=dict(l=8, r=8, t=20, b=8))


def grafica_velas(df: pd.DataFrame, titulo: str = "") -> go.Figure:
    """Gráfica de velas OHLC."""
    fig = go.Figure(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   increasing=dict(line_color=TOKENS["verde"], fillcolor=TOKENS["verde"]),
                                   decreasing=dict(line_color=TOKENS["rojo"], fillcolor=TOKENS["rojo"])))
    return estilo_figura(fig, 320, title=titulo, xaxis_rangeslider_visible=False)


def grafica_matriz_confusion(cm: list[list[int]]) -> go.Figure:
    """Mapa de calor de la matriz de confusión 2x2 (real vs predicho) con los conteos dentro de cada celda."""
    etiquetas_x, etiquetas_y = ["Pred: baja", "Pred: sube"], ["Real: baja", "Real: sube"]
    fig = go.Figure(go.Heatmap(z=cm, x=etiquetas_x, y=etiquetas_y, colorscale=[[0, TOKENS["superficie"]], [1, TOKENS["acento"]]],
                               showscale=False, text=cm, texttemplate="%{text}", textfont=dict(size=16, color=TOKENS["texto"]),
                               xgap=3, ygap=3, hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return estilo_figura(fig, 260, margin=dict(l=8, r=8, t=8, b=8))


def grafica_importancias(imp: pd.DataFrame, top: int = 12) -> go.Figure:
    """Barras horizontales con la importancia de cada feature del RandomForest (las `top` más importantes)."""
    d = imp.sort_values("importancia", ascending=True).tail(top)
    fig = go.Figure(go.Bar(x=d["importancia"], y=d["feature"], orientation="h", marker_color=TOKENS["acento"], marker_line_width=0,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return estilo_figura(fig, max(220, 24 * len(d) + 40), margin=dict(l=8, r=8, t=8, b=8), bargap=0.35)
