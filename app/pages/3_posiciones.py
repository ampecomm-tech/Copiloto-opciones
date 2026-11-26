"""Pestaña 3: posiciones de opciones con P&L y griegas en vivo, griegas del portafolio, alertas e historial."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import auth, ui
from core import hedging, positions
from core.config import cfg
from core.events import leer_eventos
from quant.portfolio import griegas_portafolio
from quant.strategies_options import griegas_posicion

NOMBRE_VENUE = {"deribit_testnet": "Deribit testnet", "gate_options_testnet": "Gate.io testnet", "binance_options_testnet": "Binance Options testnet",
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


_ROJO_T, _VERDE_T, _GRIS_T = "rgba(242, 85, 90, 0.22)", "rgba(61, 214, 140, 0.22)", "rgba(138, 148, 166, 0.16)"


def _medidores_portafolio(gp: dict, umbral: float) -> None:
    """Medidores del portafolio: delta en dólares contra el umbral de alerta de cobertura y theta diaria."""
    lim = ui.escala_simetrica(gp["delta_dolares"], 2 * umbral)
    th = ui.escala_simetrica(gp["theta_diaria"], 10.0)
    c = st.columns(2, gap="medium")
    with c[0], st.container(border=True):
        st.plotly_chart(ui.medidor(gp["delta_dolares"], -lim, lim, "Delta del portafolio", " USD", 0,
                                   [(-lim, -umbral, _ROJO_T), (-umbral, umbral, _GRIS_T), (umbral, lim, _ROJO_T)]),
                        use_container_width=True, key="med_port_delta", config=ui.SIN_BARRA_PLOTLY)
        st.caption(f"Cuánto gana o pierde el portafolio por cada dólar que se mueve el subyacente, en dólares de exposición. "
                   f"Zona roja: más de {umbral:,.0f} USD en cualquier dirección, se sugiere cubrir.")
    with c[1], st.container(border=True):
        st.plotly_chart(ui.medidor(gp["theta_diaria"], -th, th, "Theta diaria del portafolio", " USD", 2,
                                   [(-th, 0, _ROJO_T), (0, th, _VERDE_T)]),
                        use_container_width=True, key="med_port_theta", config=ui.SIN_BARRA_PLOTLY)
        st.caption("Lo que el portafolio gana o pierde cada día solo por el paso del tiempo. Rojo: el tiempo cuesta dinero; verde: el tiempo paga.")


def _medidores_posicion(p: dict, S: float | None, uid: str | None = None) -> None:
    """Cuatro medidores tipo válvula con las griegas de una posición, normalizadas para que se puedan comparar.

    - Delta por unidad del subyacente, de −1 a 1: dirección de la apuesta.
    - Gamma: cuánto cambia esa delta si el subyacente se mueve 1 %.
    - Theta: USD por día como porcentaje de la pérdida máxima.
    - Vega: USD por un punto más de IV como porcentaje de la pérdida máxima.
    """
    g = p.get("greeks") or {}
    S = S or float(p.get("spot_entrada") or 0) or 1.0
    unidades = max((abs(float(l["qty"])) * float(p.get("multiplicador", 1.0)) for l in p["legs"]), default=1.0) or 1.0
    perdida = max(float(p.get("max_perdida") or 0), 1e-9)
    delta, gamma, theta, vega = (float(g.get(k, 0.0)) for k in ("delta", "gamma", "theta", "vega"))
    du = delta / unidades
    gu = gamma * S * 0.01 / unidades
    sentido = "alcista" if du > 0.1 else "bajista" if du < -0.1 else "casi neutral"
    valor = p.get("last_value") if p.get("last_value") is not None else p["entry_cost"]
    pnl = valor - p["entry_cost"]
    with st.container(border=True):
        st.html("<div class='cp-titulo'>"
                f"<span class='cp-titulo__texto'>#{p['id']} · {ui.esc(p['asset'])} · {ui.esc(p['nombre'])}</span>"
                f"{ui.pill(f'P&L {pnl:+,.2f} USD', _tono(pnl))}{ui.pill(NOMBRE_VENUE.get(p.get('venue', 'paper'), p.get('venue')), 'gris', punto=False)}</div>")
        tp, vp = theta / perdida * 100, vega / perdida * 100
        lg, lt, lv = ui.escala_simetrica(gu, 0.1), ui.escala_simetrica(tp, 5.0), ui.escala_simetrica(vp, 10.0)
        medidores = [
            (ui.medidor(du, -1, 1, "Delta", "", 2, [(-1, -0.1, _ROJO_T), (-0.1, 0.1, _GRIS_T), (0.1, 1, _VERDE_T)]),
             f"Dirección {sentido}: {delta * S:+,.0f} USD de exposición al precio.", "d"),
            (ui.medidor(gu, -lg, lg, "Gamma", "", 3, [(-lg, 0, _ROJO_T), (0, lg, _VERDE_T)]),
             "Cuánto cambia la delta si el precio se mueve 1 %. Verde: los movimientos grandes ayudan.", "g"),
            (ui.medidor(tp, -lt, lt, "Theta", " %", 2, [(-lt, 0, _ROJO_T), (0, lt, _VERDE_T)]),
             f"{theta:+,.2f} USD por día, como % de la pérdida máxima.", "t"),
            (ui.medidor(vp, -lv, lv, "Vega", " %", 2, [(-lv, 0, _GRIS_T), (0, lv, _GRIS_T)]),
             f"{vega:+,.2f} USD si la IV sube 1 punto, como % de la pérdida máxima.", "v"),
        ]
        for fila in (medidores[:2], medidores[2:]):
            cols = st.columns(2, gap="medium")
            for col, (fig, texto, clave) in zip(cols, fila):
                with col:
                    st.plotly_chart(fig, use_container_width=True, key=f"med_{clave}_{p['id']}", config=ui.SIN_BARRA_PLOTLY)
                    st.caption(texto)
        st.divider()
        _cobertura(p, S, uid)


def _cobertura(p: dict, S: float | None, uid: str | None) -> None:
    """Bloque de cobertura de delta de una posición: delta de opciones, coberturas abiertas, sugerencia y botones.

    La sugerencia es la misma de la alerta de Discord, pero con signo y ya descontando lo cubierto. El botón
    "Cubrir delta" abre un formulario con la cantidad editable y envía la orden al venue que corresponde.
    """
    if not S:
        st.caption("Sin precio actual del subyacente: no se puede calcular la cobertura.")
        return
    r = float(cfg("app", "tasa_libre_riesgo", default=0.045))
    d_op = griegas_posicion(p["legs"], S, r, p.get("multiplicador", 1.0), p.get("dias_transcurridos", 0.0))["delta"]
    sug = hedging.sugerencia(p, d_op, S)
    cob = hedging.abiertas(p["id"])
    umbral = float(cfg("posiciones", "umbral_delta_dolares", default=500))
    neta_usd = sug["delta_neta"] * S
    st.html("<div class='cp-titulo' style='margin-top:4px'><span class='cp-titulo__texto'>Cobertura de delta</span>"
            f"{ui.pill(f'Delta neta {neta_usd:+,.0f} USD', 'rojo' if abs(neta_usd) > umbral else 'gris')}</div>")
    c = st.columns(3)
    c[0].metric("Delta de las opciones", f"{d_op:+,.4g}", f"{d_op * S:+,.0f} USD", delta_color="off", delta_arrow="off")
    c[1].metric("Cubierto", f"{sug['delta_cobertura']:+,.4g}", f"{sug['delta_cobertura'] * S:+,.0f} USD", delta_color="off", delta_arrow="off")
    c[2].metric("Delta neta", f"{sug['delta_neta']:+,.4g}", f"{neta_usd:+,.0f} USD", delta_color="off", delta_arrow="off")
    st.caption(f"Sugerencia: {sug['texto']}.")

    dest = sug["destino"]
    enteras = dest["enteras"]
    with st.popover("Cubrir delta", icon=":material/shield:", disabled=sug["qty"] == 0 and not cob, use_container_width=False):
        st.markdown(f"**Orden de cobertura · #{p['id']} {ui.esc(p['asset'])}**")
        st.caption(f"Se envía a {hedging.NOMBRE_VENUE[dest['venue']]} ({dest['instrumento']}). Positivo compra, negativo vende. "
                   "La cantidad sugerida deja la delta neta en cero.")
        qty = st.number_input("Unidades del subyacente", value=float(sug["qty"]), step=1.0 if enteras else 0.001,
                              format="%.0f" if enteras else "%.4f", key=f"cob_qty_{p['id']}")
        st.caption(f"Aprox. {abs(qty) * S:,.0f} USD a {S:,.2f}. Delta neta después: {(sug['delta_neta'] + qty) * S:+,.0f} USD.")
        if st.button("Enviar orden de cobertura", key=f"cob_env_{p['id']}", type="primary", use_container_width=True, disabled=qty == 0):
            try:
                with st.spinner("Enviando la orden…"):
                    h = hedging.abrir(p["id"], qty, uid)
                st.session_state["cob_msg"] = ("ok", f"Cobertura abierta: {h['detalle']}.")
            except Exception as e:
                st.session_state["cob_msg"] = ("error", f"No se envió la cobertura de la posición #{p['id']}: {e}")
            st.rerun()
    for h in cob:
        hc = st.columns([3, 1], vertical_alignment="center")
        g = hedging.pnl(h, S)
        hc[0].html(f"<div class='cp-texto' style='font-size:0.85rem'>{ui.esc(h.get('detalle') or h['instrumento'])} · "
                   f"{ui.pill(f'P&L {g:+,.2f} USD', _tono(g))}</div>")
        if hc[1].button("Cerrar", key=f"cob_cerrar_{h['id']}", icon=":material/close:", use_container_width=True):
            try:
                with st.spinner("Cerrando la cobertura…"):
                    res = hedging.cerrar(int(h["id"]), uid)
                st.session_state["cob_msg"] = ("ok", f"Cobertura #{h['id']} cerrada a {res['precio']:,.2f}: P&L {res['pnl']:+,.2f} USD.")
            except Exception as e:
                st.session_state["cob_msg"] = ("error", f"No se cerró la cobertura #{h['id']}: {e}")
            st.rerun()


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

    msg = st.session_state.pop("cob_msg", None)
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])

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
        for p in abiertas:
            p["delta_cobertura"] = hedging.delta_cobertura(p["id"])
        gp = griegas_portafolio(abiertas, spots, float(cfg("app", "tasa_libre_riesgo", default=0.045)))

        st.space("small")
        ui.franja_stats([
            ("Posiciones abiertas", str(len(abiertas)), ", ".join(sorted({p["asset"] for p in abiertas}))),
            ("P&L abierto (USD)", ui.pill(_usd(h["pnl_abierto"]), _tono(h["pnl_abierto"])), "sin realizar"),
            ("Delta en dólares", f"{gp['delta_dolares']:+,.0f}", "con coberturas"),
            ("Theta diaria", f"{gp['theta_diaria']:+,.2f}", "USD por día"),
            ("Vega total", f"{gp['vega_total']:+,.2f}", "USD por 1% de IV"),
            ("Gamma $", f"{gp['gamma_dolares']:+,.2f}", "por 1% de movimiento"),
        ])
        _medidores_portafolio(gp, float(cfg("posiciones", "umbral_delta_dolares", default=500)))

        st.subheader("Posiciones abiertas")
        filas = []
        for p in abiertas:
            g = p["greeks"] or {}
            valor = p.get("last_value") if p.get("last_value") is not None else p["entry_cost"]
            act = p.get("last_update_ts")
            filas.append({"ID": p["id"], "Activo": p["asset"], "Estrategia": p["nombre"], "Venue": NOMBRE_VENUE.get(p.get("venue", "paper"), p.get("venue")),
                          "Vence": p["expiry"], "Costo": p["entry_cost"], "Valor actual": valor, "P&L": valor - p["entry_cost"],
                          "Pérdida máx.": p["max_perdida"], "Delta": g.get("delta", 0.0), "Cubierto": p.get("delta_cobertura", 0.0), "Theta/día": g.get("theta", 0.0), "Vega": g.get("vega", 0.0),
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
                "Delta": st.column_config.NumberColumn(format="%+.3f", help="Delta de las opciones, en unidades del subyacente"),
                "Cubierto": st.column_config.NumberColumn(format="%+.4f", help="Unidades del subyacente en coberturas abiertas"),
                "Theta/día": st.column_config.NumberColumn(format="%+.2f"),
                "Vega": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        st.subheader("Griegas de cada posición")
        st.caption("La aguja muestra dónde está cada griega dentro de su escala. Delta y gamma van por unidad del subyacente; theta y vega, como porcentaje de la pérdida máxima, "
                   "para comparar posiciones de distinto tamaño. Debajo de cada título va el valor en dólares.")
        for p in abiertas:
            _medidores_posicion(p, spots.get(p["asset"]), uid)

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
            st.caption("El cierre se registra en paper con el valor indicado (por defecto, el último valor revaluado). "
                       "Sus coberturas abiertas se cierran con la orden contraria en su venue.")

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
