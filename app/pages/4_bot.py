"""Pestaña 4: bot de trading (buscador por nombre, entrenamiento, estrategias, desempeño y validación)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import ui
from bot.runner import registrar_activo
from bot.trainer import entrenar_activo
from core import db
from core.config import RAIZ, cfg
from core.tickers import buscar
from data_sources.market import velas
from quant.portfolio import metricas_desempeno
from quant.riesgo import resumen_riesgo

SALIDA = RAIZ / "validation" / "output"
NOMBRE_VENUE = {"binance_spot_testnet": "Binance spot testnet", "alpaca_paper": "Alpaca Paper", "paper": "Paper interno"}
TONO_MODELO = {"listo": "verde", "entrenando": "amarillo", "error": "rojo", "fallido": "rojo"}


def _color_signo(x: object) -> str:
    """Estilo CSS de celda: verde si es positivo, rojo si es negativo."""
    if not isinstance(x, (int, float)) or x != x or x == 0:
        return ""
    return f"color: {ui.TOKENS['verde'] if x > 0 else ui.TOKENS['rojo']}"


def _color_accion(x: object) -> str:
    """Estilo CSS de celda para la acción de una señal (BUY verde, SELL rojo, HOLD gris)."""
    return {"BUY": f"color: {ui.TOKENS['verde']}", "SELL": f"color: {ui.TOKENS['rojo']}"}.get(str(x), f"color: {ui.TOKENS['gris']}")


def _hora(ts: str | None) -> str:
    """Timestamp ISO a `dd/mm HH:MM` en la zona de la app."""
    if not ts:
        return "—"
    try:
        return f"{ui.hora_local(ts):%d/%m %H:%M}"
    except Exception:
        return str(ts)[:16]


# --- Resumen -----------------------------------------------------------------------------------

def _resumen() -> None:
    """Franja con el estado global del bot: activos, estrategias, capital, P&L y operaciones."""
    activos = db.consultar("SELECT active FROM bot_assets")
    estados = db.consultar("SELECT s.active, s.capital FROM bot_strategy_state s JOIN bot_assets a ON a.id = s.asset_id")
    cerr = db.consultar_uno("SELECT COUNT(*) AS n, COALESCE(SUM(pnl), 0) AS pnl FROM bot_trades WHERE exit_ts IS NOT NULL") or {"n": 0, "pnl": 0}
    abiertas = db.consultar_uno("SELECT COUNT(*) AS n FROM bot_trades WHERE exit_ts IS NULL") or {"n": 0}
    cap0 = float(cfg("bot", "capital_inicial_por_estrategia", default=1000))
    capital = sum(float(e["capital"]) for e in estados)
    pnl = float(cerr["pnl"] or 0)
    tono = "verde" if pnl > 0 else "rojo" if pnl < 0 else "gris"
    ui.franja_stats([
        ("Activos", str(len(activos)), f"{sum(1 for a in activos if a['active'])} operando"),
        ("Estrategias", str(len(estados)), f"{sum(1 for e in estados if e['active'])} activas"),
        ("Capital virtual (USD)", f"{capital:,.0f}", f"{cap0:,.0f} por estrategia"),
        ("P&L realizado (USD)", ui.pill(f"{pnl:+,.2f}", tono), f"{cerr['n']} operaciones cerradas"),
        ("Posiciones abiertas", str(abiertas["n"]), ""),
        ("Timeframe", str(cfg("bot", "timeframe")), "señales con datos de Binance real"),
    ])


# --- Buscador ----------------------------------------------------------------------------------

def _buscador() -> None:
    """Buscador por nombre y alta de activos con entrenamiento automático."""
    with st.container(border=True):
        st.markdown("**Agregar un activo**")
        with st.form("busqueda", border=False):
            c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
            q = c1.text_input("Busca por nombre", placeholder="Solana, Apple, oro, NVDA…", key="busq", label_visibility="collapsed")
            if c2.form_submit_button("Buscar", icon=":material/search:", use_container_width=True) and q.strip():
                with st.spinner("Buscando…"):
                    st.session_state["resultados_busqueda"] = buscar(q)
        res = st.session_state.get("resultados_busqueda") or []
        if not res:
            st.caption("El venue se asigna solo: cripto listada → Binance spot testnet, acciones y ETFs de EE. UU. → Alpaca Paper, lo demás → paper interno.")
            return
        df = pd.DataFrame([{"Nombre": r["display_name"], "Símbolo": r["symbol"], "Mercado": r["market_type"],
                            "Venue": NOMBRE_VENUE.get(r["venue"], r["venue"]), "Precio": r.get("price")} for r in res])
        ev = st.dataframe(df, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="tabla_busqueda",
                          column_config={"Precio": st.column_config.NumberColumn(format="%.2f")})
        filas = ev.selection.rows if ev and ev.selection else []
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        if not filas:
            c1.caption("Selecciona una fila para agregarla al bot.")
            return
        r = res[filas[0]]
        c1.caption(f"Se agregará **{r['display_name']}** ({r['symbol']}) en {NOMBRE_VENUE.get(r['venue'], r['venue'])}; "
                   "se descarga el histórico y se entrena el RandomForest antes de operar.")
        if c2.button("Agregar y entrenar", type="primary", icon=":material/add:", use_container_width=True):
            aid = registrar_activo(r["symbol"], r["display_name"], r["market_type"], r["venue"])
            barra = st.progress(0, text="Preparando…")
            try:
                entrenar_activo(aid, r["symbol"], r["venue"], None, lambda m, f: barra.progress(int(f * 100), text=m))
                st.success(f"{r['display_name']} entrenado y activo. El bot empezará a operar en el próximo ciclo.", icon=":material/check_circle:")
            except Exception as e:
                st.error(f"Entrenamiento fallido: {e}", icon=":material/error:")
            finally:
                barra.empty()
            st.session_state.pop("resultados_busqueda", None)
            st.rerun()


# --- Activos y estrategias ---------------------------------------------------------------------

def _texto_posicion(pos: dict | None) -> str:
    """Resumen de la posición abierta de una estrategia."""
    if not pos:
        return "—"
    base = f"{pos['side']} {pos['qty']:g} @ {pos['entry_price']:,.2f}"
    if pos.get("trailing_activo"):
        return f"{base} · SL móvil {pos['sl']:,.2f}"
    return f"{base} · SL {pos['sl']:,.2f} · TP {pos['tp']:,.2f}"


def _activo(a: dict, cap0: float) -> None:
    """Panel de un activo: estado, interruptor, reentrenar, tabla de estrategias editable, velas y curva de capital."""
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="center")
    c1.html("<div class='cp-titulo'>"
            f"<span class='cp-titulo__texto'>{ui.esc(a['display_name'])}</span>"
            f"{ui.pill(NOMBRE_VENUE.get(a['venue'], a['venue']), 'acento', punto=False)}"
            f"{ui.pill('Modelo ' + str(a['training_status']), TONO_MODELO.get(str(a['training_status']), 'gris'))}</div>")
    activo_flag = c2.toggle("Operando", bool(a["active"]), key=f"act_{a['id']}")
    if activo_flag != bool(a["active"]):
        db.ejecutar("UPDATE bot_assets SET active=? WHERE id=?", (int(activo_flag), a["id"]))
    if c3.button("Reentrenar", key=f"re_{a['id']}", icon=":material/model_training:", use_container_width=True):
        barra = st.progress(0)
        try:
            entrenar_activo(a["id"], a["symbol"], a["venue"], None, lambda m, f: barra.progress(int(f * 100), text=m))
        finally:
            barra.empty()
        st.rerun()

    estados = db.consultar("SELECT * FROM bot_strategy_state WHERE asset_id=? ORDER BY strategy", (a["id"],))
    trades = pd.DataFrame(db.consultar("SELECT * FROM bot_trades WHERE asset_id=? AND exit_ts IS NOT NULL", (a["id"],)))
    filas = []
    for e in estados:
        t = trades[trades["strategy"] == e["strategy"]] if not trades.empty else pd.DataFrame()
        eq = pd.Series([cap0] + (cap0 + t.sort_values("exit_ts")["pnl"].cumsum()).tolist()) if len(t) else pd.Series([cap0, e["capital"]])
        met = metricas_desempeno(eq, periodos_por_anio=365)
        wr = (t["pnl"] > 0).mean() if len(t) else None
        filas.append({"Activa": bool(e["active"]), "Estrategia": e["strategy"], "Capital": float(e["capital"]), "ROI": met["roi"] * 100,
                      "Sharpe": met["sharpe"], "DD máx.": met["max_drawdown"] * 100,
                      "VaR 95%": met["var_95"] * 100 if len(t) >= 2 else None, "CVaR 95%": met["cvar_95"] * 100 if len(t) >= 2 else None,
                      "Win rate": f"{wr:.0%}" if wr is not None else "—",
                      "Trades": len(t), "Posición": _texto_posicion(db.de_json(e.get("position_json")))})
    if filas:
        df = pd.DataFrame(filas)
        editado = st.data_editor(
            df, hide_index=True, use_container_width=True, key=f"estr_{a['id']}", disabled=[c for c in df.columns if c != "Activa"],
            column_config={
                "Activa": st.column_config.CheckboxColumn(help="Activa o pausa la estrategia; el bot lo lee en el siguiente ciclo, sin reiniciar"),
                "Capital": st.column_config.NumberColumn(format="%.0f"),
                "ROI": st.column_config.NumberColumn(format="%+.1f%%"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "DD máx.": st.column_config.NumberColumn(format="%.1f%%"),
                "VaR 95%": st.column_config.NumberColumn(format="%.2f%%", help="Pérdida por operación que solo se supera en el 5 % de las peores operaciones (histórico)"),
                "CVaR 95%": st.column_config.NumberColumn(format="%.2f%%", help="Pérdida promedio dentro de ese 5 % de peores operaciones (Expected Shortfall)"),
                "Posición": st.column_config.TextColumn(width="large"),
            })
        for antes, despues in zip(df.itertuples(), editado.itertuples()):
            if bool(antes.Activa) != bool(despues.Activa):
                db.ejecutar("UPDATE bot_strategy_state SET active=? WHERE asset_id=? AND strategy=?", (int(despues.Activa), a["id"], antes.Estrategia))

    try:
        df_v = velas(a["symbol"], a["venue"]).tail(150)
        if len(df_v):
            st.plotly_chart(ui.grafica_velas(df_v, f"{a['symbol']} · velas de {cfg('bot', 'timeframe')}"), use_container_width=True, key=f"velas_{a['id']}")
    except Exception:
        st.caption("No se pudieron descargar las velas.")
    if len(trades):
        st.markdown("**Curva de capital**")
        st.plotly_chart(ui.grafica_equity(trades, cap0 * len(estados)), use_container_width=True, key=f"eq_{a['id']}")


def _activos() -> bool:
    """Una pestaña por activo con sus estrategias. Devuelve True si hay activos."""
    st.subheader("Activos y estrategias")
    activos = db.consultar("SELECT * FROM bot_assets ORDER BY id")
    if not activos:
        st.caption("Aún no hay activos. Busca uno arriba, por ejemplo Solana.")
        return False
    cap0 = float(cfg("bot", "capital_inicial_por_estrategia", default=1000))
    tabs = st.tabs([f"{'' if a['active'] else ':gray['}{a['symbol']}{'' if a['active'] else ']'}" for a in activos])
    for tab, a in zip(tabs, activos):
        with tab:
            _activo(a, cap0)
    return True


# --- Señales y operaciones ---------------------------------------------------------------------

def _senales_y_trades() -> None:
    """Últimas señales y operaciones, en pestañas."""
    st.subheader("Actividad reciente")
    t_sen, t_ops = st.tabs(["Señales", "Operaciones"])
    with t_sen:
        s = db.consultar("SELECT s.ts, a.symbol, s.strategy, s.action, s.confidence FROM bot_signals s JOIN bot_assets a ON a.id=s.asset_id ORDER BY s.id DESC LIMIT 30")
        if not s:
            st.caption("Sin señales todavía.")
        else:
            df = pd.DataFrame([{"Hora": _hora(x["ts"]), "Activo": x["symbol"], "Estrategia": x["strategy"], "Acción": x["action"], "Confianza": x["confidence"]} for x in s])
            st.dataframe(df.style.map(_color_accion, subset=["Acción"]), hide_index=True, use_container_width=True,
                         column_config={"Confianza": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    with t_ops:
        t = db.consultar("SELECT t.entry_ts, a.symbol, t.strategy, t.side, t.qty, t.entry_price, t.exit_price, t.pnl, t.exit_reason FROM bot_trades t JOIN bot_assets a ON a.id=t.asset_id ORDER BY t.id DESC LIMIT 30")
        if not t:
            st.caption("Sin operaciones todavía.")
        else:
            df = pd.DataFrame([{"Entrada": _hora(x["entry_ts"]), "Activo": x["symbol"], "Estrategia": x["strategy"], "Lado": x["side"], "Cantidad": x["qty"],
                                "Precio entrada": x["entry_price"], "Precio salida": x["exit_price"], "P&L": x["pnl"], "Motivo de salida": x["exit_reason"] or "abierta"}
                               for x in t])
            st.dataframe(df.style.map(_color_signo, subset=["P&L"]), hide_index=True, use_container_width=True,
                         column_config={"Cantidad": st.column_config.NumberColumn(format="%g"), "Precio entrada": st.column_config.NumberColumn(format="%.4f"),
                                        "Precio salida": st.column_config.NumberColumn(format="%.4f"), "P&L": st.column_config.NumberColumn(format="%+.2f")})


# --- Validación --------------------------------------------------------------------------------

def _riesgo_activo(aid: int, simbolo: str) -> None:
    """VaR y CVaR del activo: retornos de 15 minutos del subyacente y P&L de las operaciones del bot, con sus distribuciones."""
    st.caption("VaR: pérdida que solo se supera en el peor 5 % (o 1 %) de los casos. CVaR o Expected Shortfall: pérdida promedio dentro de ese peor 5 %. "
               "Se comparan la versión histórica y la normal: si la histórica es mayor, la cola real es más pesada que la de una normal.")
    asset = db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,))
    try:
        df = velas(asset["symbol"], asset["venue"])
        rets = df["close"].pct_change().dropna() * 100
    except Exception:
        rets = pd.Series(dtype=float)
    trades = pd.DataFrame(db.consultar("SELECT strategy, pnl FROM bot_trades WHERE asset_id=? AND exit_ts IS NOT NULL AND qty > 0 AND pnl IS NOT NULL", (aid,)))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"**Subyacente: retornos de {cfg('bot', 'timeframe')}**")
        if len(rets) > 20:
            r = resumen_riesgo(rets.values)
            ui.franja_stats([("VaR 95%", f"{r['var_95']:.3f} %", f"normal {r['var_95_normal']:.3f} %"),
                             ("CVaR 95%", f"{r['cvar_95']:.3f} %", f"normal {r['cvar_95_normal']:.3f} %"),
                             ("CVaR 99%", f"{r['cvar_99']:.3f} %", f"curtosis {r['curtosis']:.1f}")])
            st.caption(f"Distribución de {len(rets):,} velas de {simbolo}. Rojo: el peor 5 %. Punteada: la normal con la misma media y desviación.")
            st.plotly_chart(ui.grafica_var(rets.values, r, "Retorno por vela (%)", ajuste_normal=True),
                            use_container_width=True, key=f"var_sub_{aid}", config=ui.SIN_BARRA_PLOTLY)
        else:
            st.caption("No se pudieron descargar velas suficientes.")
    with c2:
        st.markdown("**Bot: P&L por operación cerrada (USD)**")
        if len(trades) >= 5:
            r = resumen_riesgo(trades["pnl"].values)
            ui.franja_stats([("VaR 95%", f"{r['var_95']:,.2f}", "USD por operación"), ("CVaR 95%", f"{r['cvar_95']:,.2f}", "USD por operación"),
                             ("Operaciones", str(r["n"]), "todas las estrategias")])
            st.caption(f"Distribución de {len(trades)} operaciones cerradas de {simbolo}. Rojo: el peor 5 %.")
            st.plotly_chart(ui.grafica_var(trades["pnl"].values, r, "P&L por operación (USD)"),
                            use_container_width=True, key=f"var_bot_{aid}", config=ui.SIN_BARRA_PLOTLY)
            filas = []
            for est, g in trades.groupby("strategy"):
                rr = resumen_riesgo(g["pnl"].values)
                filas.append({"Estrategia": est, "Operaciones": len(g), "P&L medio": float(g["pnl"].mean()),
                              "VaR 95%": rr.get("var_95", float("nan")), "CVaR 95%": rr.get("cvar_95", float("nan"))})
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("P&L medio", "VaR 95%", "CVaR 95%")})
        else:
            st.caption("Hacen falta al menos 5 operaciones cerradas para estimar la cola.")

    if len(rets) > 400:
        _caviar(aid, simbolo, rets)


@st.cache_data(ttl=900, show_spinner=False)
def _analisis_caviar(simbolo: str, ultima_vela: str, valores: tuple[float, ...], modelo: str) -> dict:
    """Estimación CAViaR en caché por activo, última vela y modelo (el optimizador no se repite en cada recarga)."""
    from quant.caviar import analizar

    return analizar(list(valores), 0.05, modelo)


def _caviar(aid: int, simbolo: str, rets: pd.Series) -> None:
    """CAViaR al 95 %: VaR dinámico estimado por regresión de cuantiles, backtesting y comparación con el VaR histórico."""
    from quant.caviar import MODELOS

    st.space("small")
    st.markdown("**CAViaR: VaR que cambia con el mercado (Engle y Manganelli, 2004)**")
    st.caption("En lugar de una sola cifra para toda la muestra, CAViaR estima el VaR de cada vela a partir del VaR anterior y del último retorno, "
               "por regresión de cuantiles. Se estima con el 70 % inicial de las velas y se evalúa en el 30 % final, que el modelo no vio; "
               "ahí se reestima cada 100 velas usando solo datos pasados, para que se adapte si cambia la volatilidad.")
    modelo = st.segmented_control("Especificación", list(MODELOS), default="sav", format_func=MODELOS.get, key=f"cav_mod_{aid}", required=True) or "sav"
    with st.spinner("Estimando CAViaR…"):
        a = _analisis_caviar(simbolo, str(rets.index[-1]), tuple(float(v) for v in rets.values), modelo)
    m, kf, kd, ch = a["modelo"], a["kupiec_fuera"], a["kupiec_dentro"], a["christoffersen_fuera"]
    dm, kh, pc = a.get("dm_vs_historico") or {}, a.get("kupiec_historico_fuera") or {}, a.get("perdida_cuantil_fuera") or {}
    ui.franja_stats([
        ("VaR CAViaR última vela", f"{-a['q_caviar'][-1]:.3f} %", f"histórico móvil {-a['q_historico'][-1]:.3f} %"),
        ("Violaciones fuera de muestra", f"{kf['tasa']:.1%}", f"{kf['violaciones']} de {kf['n']} velas; objetivo 5 %"),
        ("Kupiec (cobertura)", f"p = {kf['p_valor']:.3f}", "p ≥ 0.05: la tasa es compatible con 5 %"),
        ("Christoffersen (conjunta)", f"p = {ch['p_cc']:.3f}", "p ≥ 0.05: sin rachas de violaciones"),
        ("Diebold-Mariano vs histórico", {"modelo_1": "CAViaR mejor", "modelo_2": "Histórico mejor"}.get(dm.get("mejor"), "Sin diferencia"),
         f"p = {dm['p_valor']:.3f} · pérdida de cuantil" if dm else "—"),
    ])
    st.plotly_chart(ui.grafica_caviar(rets.index, a), use_container_width=True, key=f"cav_{aid}_{modelo}", config=ui.SIN_BARRA_PLOTLY)
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown("**Parámetros estimados**")
        mu = a.get("modelo_ultimo") or m
        st.dataframe(pd.DataFrame({"Parámetro": m["nombres"], "Estimación inicial": m["beta"], "Última reestimación": mu["beta"]}),
                     hide_index=True, use_container_width=True,
                     column_config={"Estimación inicial": st.column_config.NumberColumn(format="%.4f"), "Última reestimación": st.column_config.NumberColumn(format="%.4f")})
        st.caption(f"Cuantil inicial q₀ = {m['q0']:.3f} %. {a.get('reestimaciones', 0)} reestimaciones fuera de muestra. "
                   f"β1 cerca de 1 significa que el VaR tiene mucha memoria; β2 negativo, que un movimiento fuerte aumenta el VaR de la siguiente vela.")
    with c2:
        st.markdown("**Backtesting**")
        filas = [{"Modelo": "CAViaR, dentro de muestra", "Violaciones": kd["violaciones"], "Velas": kd["n"], "Tasa": kd["tasa"] * 100, "Kupiec p": kd["p_valor"], "Pérdida de cuantil": m["perdida_media"]},
                 {"Modelo": "CAViaR, fuera de muestra", "Violaciones": kf["violaciones"], "Velas": kf["n"], "Tasa": kf["tasa"] * 100, "Kupiec p": kf["p_valor"], "Pérdida de cuantil": pc.get("caviar")}]
        if kh:
            filas.append({"Modelo": "Histórico móvil, fuera de muestra", "Violaciones": kh["violaciones"], "Velas": kh["n"], "Tasa": kh["tasa"] * 100, "Kupiec p": kh["p_valor"], "Pérdida de cuantil": pc.get("historico")})
        st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True,
                     column_config={"Tasa": st.column_config.NumberColumn(format="%.1f%%"), "Kupiec p": st.column_config.NumberColumn(format="%.3f"),
                                    "Pérdida de cuantil": st.column_config.NumberColumn(format="%.5f")})
        st.caption("La pérdida de cuantil es la función que minimiza CAViaR; la menor es el mejor pronóstico del VaR. "
                   "Diebold-Mariano compara esas pérdidas vela por vela, como en la validación del Random Forest.")

def _validacion() -> None:
    """Tabla de hiperparámetros, métricas, matriz de confusión y Diebold-Mariano por activo; exporta CSV/PNG."""
    st.subheader("Validación cuantitativa")
    activos = db.consultar("SELECT id, symbol FROM bot_assets ORDER BY id")
    if not activos:
        return
    simbolos = [a["symbol"] for a in activos]
    sel = st.segmented_control("Activo", simbolos, default=simbolos[0], key="val_activo", required=True, label_visibility="collapsed") or simbolos[0]
    aid = next(a["id"] for a in activos if a["symbol"] == sel)
    run = db.consultar_uno("SELECT * FROM model_runs WHERE asset_id=? ORDER BY id DESC LIMIT 1", (aid,))
    if not run:
        st.caption("Este activo aún no tiene entrenamiento.")
        return
    met, dm, hp = db.de_json(run["metrics_json"], {}), db.de_json(run["dm_results_json"], {}), db.de_json(run["hyperparams_json"], {})
    mejor = met.get("mejor_rf") or {}
    ui.franja_stats([
        ("Entrenado", _hora(run["ts"]), ""),
        ("Velas", f"{met.get('n_velas', '—')}", f"{met.get('n_entrenamiento', '—')} entrenamiento · {met.get('n_prueba', '—')} prueba"),
        ("Horizonte", f"{met.get('horizonte_velas', '—')} velas", "de predicción"),
        ("Accuracy (mejor RF)", f"{mejor.get('accuracy', float('nan')):.1%}" if mejor.get("accuracy") is not None else "—", ""),
        ("F1 (mejor RF)", f"{mejor.get('f1', float('nan')):.3f}" if mejor.get("f1") is not None else "—", ""),
        ("Mejores hiperparámetros", " · ".join(f"{v}" for v in hp.values()) or "—", " / ".join(hp.keys())),
    ])

    tabla = pd.DataFrame(met.get("tabla_hiperparametros", []))
    comp = {"random_forest": {k: mejor.get(k) for k in ("accuracy", "precision", "recall", "f1")}}
    comp.update({k: {m: v.get(m) for m in ("accuracy", "precision", "recall", "f1")} for k, v in (met.get("estrategias") or {}).items()})

    t_hp, t_cls, t_dm, t_riesgo, t_imp = st.tabs(["Hiperparámetros", "Clasificación", "Diebold-Mariano", "Riesgo: VaR y CVaR", "Importancia de features"])
    with t_riesgo:
        _riesgo_activo(aid, sel)
    with t_hp:
        if not tabla.empty:
            cols = [c for c in ["n_estimators", "max_depth", "min_samples_leaf", "accuracy", "precision", "recall", "f1", "rmse", "mse", "mae", "mape", "smape", "variance_score"] if c in tabla]
            st.caption("Simulación de hiperparámetros del RandomForest sobre el conjunto de prueba.")
            st.dataframe(tabla[cols].round(4), hide_index=True, use_container_width=True)
    with t_cls:
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.markdown("**Matriz de confusión (mejor RF)**")
            cm = mejor.get("matriz_confusion") or (tabla.iloc[0]["matriz_confusion"] if not tabla.empty and "matriz_confusion" in tabla else None)
            if cm:
                st.plotly_chart(ui.grafica_matriz_confusion(cm), use_container_width=True, key=f"cm_{aid}")
        with c2:
            st.markdown("**Comparativa de modelos y estrategias**")
            st.dataframe(pd.DataFrame(comp).T.round(3), use_container_width=True)
    with t_dm:
        st.caption("Prueba de Diebold-Mariano con corrección de Harvey-Leybourne-Newbold. `modelo_1` = RandomForest; p < 0.05 indica diferencia significativa de precisión.")
        if dm:
            st.dataframe(pd.DataFrame(dm).T[["dm_hln", "p_valor", "n", "mejor"]].round(4), use_container_width=True,
                         column_config={"dm_hln": "Estadístico DM (HLN)", "p_valor": "p-valor", "n": "Observaciones", "mejor": "Mejor modelo"})
    with t_imp:
        imp = pd.DataFrame(met.get("importancias", []))
        if not imp.empty:
            st.plotly_chart(ui.grafica_importancias(imp), use_container_width=True, key=f"imp_{aid}")

    if st.button("Exportar CSV y PNG para el informe", icon=":material/download:"):
        SALIDA.mkdir(parents=True, exist_ok=True)
        tabla.drop(columns=["matriz_confusion"], errors="ignore").to_csv(SALIDA / f"{sel.replace('/', '_')}_hiperparametros.csv", index=False)
        pd.DataFrame(dm).T.to_csv(SALIDA / f"{sel.replace('/', '_')}_diebold_mariano.csv")
        pd.DataFrame(comp).T.to_csv(SALIDA / f"{sel.replace('/', '_')}_comparativa.csv")
        try:
            import plotly.express as px

            fig = px.bar(tabla.assign(combo=tabla[["n_estimators", "max_depth", "min_samples_leaf"]].astype(str).agg("/".join, axis=1)), x="combo", y="f1", title=f"F1 por hiperparámetros · {sel}")
            fig.write_image(str(SALIDA / f"{sel.replace('/', '_')}_f1.png"))
        except Exception as e:
            st.caption(f"PNG omitido (instala kaleido para exportar imágenes): {e}")
        st.success(f"Exportado a {SALIDA}", icon=":material/check_circle:")


# --- Página ------------------------------------------------------------------------------------

def render() -> None:
    """Dibuja la pestaña."""
    st.title("Bot de trading")
    forzado = str(cfg("bot", "forzar_paper", default=False)).lower() == "true"
    st.caption("Opera 24/7 en la VM con varias estrategias por activo y capital virtual propio. "
               + ("Modo forzado: todo en paper interno." if forzado else "Ejecuta en Binance spot testnet, Alpaca Paper o paper interno según el activo."))
    st.space("small")
    _resumen()
    st.space("small")
    _buscador()
    st.space("small")
    if _activos():
        st.space("small")
        _senales_y_trades()
    st.space("small")
    _validacion()
