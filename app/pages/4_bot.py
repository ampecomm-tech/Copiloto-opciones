"""Pestaña 4: bot de trading (buscador por nombre, entrenamiento, estrategias, desempeño y validación)."""
from __future__ import annotations

import io
from pathlib import Path

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

SALIDA = RAIZ / "validation" / "output"


def _buscador() -> None:
    """Buscador por nombre y alta de activos con entrenamiento automático."""
    st.subheader("🔎 Agregar activo")
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Busca por nombre", placeholder="Solana, Apple, oro, NVDA…", key="busq")
    if c2.button("Buscar", use_container_width=True) and q.strip():
        with st.spinner("Buscando…"):
            st.session_state["resultados_busqueda"] = buscar(q)
    res = st.session_state.get("resultados_busqueda") or []
    if res:
        df = pd.DataFrame(res)
        st.dataframe(df.rename(columns={"symbol": "Símbolo", "display_name": "Nombre", "market_type": "Mercado", "venue": "Venue", "price": "Precio"}), hide_index=True, use_container_width=True)
        opciones = [f"{r['display_name']} · {r['symbol']} · {r['venue']}" for r in res]
        elegido = st.selectbox("Selecciona", opciones)
        r = res[opciones.index(elegido)]
        if st.button("➕ Agregar y entrenar", type="primary"):
            aid = registrar_activo(r["symbol"], r["display_name"], r["market_type"], r["venue"])
            barra = st.progress(0, text="Preparando…")
            try:
                entrenar_activo(aid, r["symbol"], r["venue"], None, lambda m, f: barra.progress(int(f * 100), text=m))
                st.success(f"{r['display_name']} entrenado y activo. El bot empezará a operar en el próximo ciclo ({r['venue']}).")
            except Exception as e:
                st.error(f"Entrenamiento fallido: {e}")
            finally:
                barra.empty()
            st.session_state.pop("resultados_busqueda", None)
            st.rerun()


def _tabla_estado() -> list[dict]:
    """Tabla activo × estrategia con capital, posición, métricas y toggles."""
    st.subheader("⚙️ Activos y estrategias")
    activos = db.consultar("SELECT * FROM bot_assets ORDER BY id")
    if not activos:
        st.info("Aún no hay activos. Busca uno arriba (p. ej. *Solana*).")
        return []
    cap0 = float(cfg("bot", "capital_inicial_por_estrategia", default=1000))
    filas = []
    for a in activos:
        with st.expander(f"{a['display_name']} · {a['symbol']} · {a['venue']} · modelo: {a['training_status']}", expanded=True):
            cc = st.columns([1, 1, 4])
            activo_flag = cc[0].toggle("Activo", bool(a["active"]), key=f"act_{a['id']}")
            if activo_flag != bool(a["active"]):
                db.ejecutar("UPDATE bot_assets SET active=? WHERE id=?", (int(activo_flag), a["id"]))
            if cc[1].button("Reentrenar", key=f"re_{a['id']}"):
                barra = st.progress(0)
                try:
                    entrenar_activo(a["id"], a["symbol"], a["venue"], None, lambda m, f: barra.progress(int(f * 100), text=m))
                finally:
                    barra.empty()
                st.rerun()
            estados = db.consultar("SELECT * FROM bot_strategy_state WHERE asset_id=? ORDER BY strategy", (a["id"],))
            trades = pd.DataFrame(db.consultar("SELECT * FROM bot_trades WHERE asset_id=? AND exit_ts IS NOT NULL", (a["id"],)))
            for e in estados:
                t = trades[trades["strategy"] == e["strategy"]] if not trades.empty else pd.DataFrame()
                eq = pd.Series([cap0] + (cap0 + t.sort_values("exit_ts")["pnl"].cumsum()).tolist()) if len(t) else pd.Series([cap0, e["capital"]])
                met = metricas_desempeno(eq, periodos_por_anio=365)
                pos = db.de_json(e.get("position_json"))
                col = st.columns([1.2, 1, 1, 1, 1, 1, 1, 2])
                on = col[0].toggle(e["strategy"], bool(e["active"]), key=f"st_{a['id']}_{e['strategy']}")
                if on != bool(e["active"]):
                    db.ejecutar("UPDATE bot_strategy_state SET active=? WHERE asset_id=? AND strategy=?", (int(on), a["id"], e["strategy"]))
                col[1].metric("Capital", f"{e['capital']:,.0f}")
                col[2].metric("ROI", f"{met['roi']:+.1%}")
                col[3].metric("Sharpe", f"{met['sharpe']:.2f}")
                col[4].metric("DD máx", f"{met['max_drawdown']:.1%}")
                wr = (t["pnl"] > 0).mean() if len(t) else float("nan")
                col[5].metric("Win rate", f"{wr:.0%}" if wr == wr else "—")
                col[6].metric("Trades", len(t))
                col[7].markdown(f"**Posición:** {pos['side']} {pos['qty']:g} @ {pos['entry_price']:,.2f} (SL {pos['sl']:,.2f} / TP {pos['tp']:,.2f})" if pos else "Sin posición")
                filas.append({"asset_id": a["id"], "symbol": a["symbol"], "strategy": e["strategy"], **met, "win_rate": wr, "trades": len(t), "capital": e["capital"]})
            if len(trades):
                st.plotly_chart(ui.grafica_equity(trades, cap0 * len(estados)), use_container_width=True, key=f"eq_{a['id']}")
            try:
                df_v = velas(a["symbol"], a["venue"]).tail(150)
                if len(df_v):
                    st.plotly_chart(ui.grafica_velas(df_v, f"{a['symbol']} · {cfg('bot', 'timeframe')}"), use_container_width=True, key=f"velas_{a['id']}")
            except Exception:
                pass
    return filas


def _senales_y_trades() -> None:
    """Últimas señales y operaciones."""
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📡 Últimas señales")
        s = db.consultar("SELECT s.ts, a.symbol, s.strategy, s.action, s.confidence FROM bot_signals s JOIN bot_assets a ON a.id=s.asset_id ORDER BY s.id DESC LIMIT 30")
        st.dataframe(pd.DataFrame(s), hide_index=True, use_container_width=True, height=300)
    with c2:
        st.subheader("💱 Operaciones")
        t = db.consultar("SELECT t.entry_ts, a.symbol, t.strategy, t.side, t.qty, t.entry_price, t.exit_price, t.pnl, t.exit_reason FROM bot_trades t JOIN bot_assets a ON a.id=t.asset_id ORDER BY t.id DESC LIMIT 30")
        st.dataframe(pd.DataFrame(t), hide_index=True, use_container_width=True, height=300)


def _validacion() -> None:
    """Tabla de hiperparámetros, métricas, matriz de confusión y Diebold-Mariano por activo; exporta CSV/PNG."""
    st.subheader("🧪 Validación cuantitativa")
    activos = db.consultar("SELECT id, symbol FROM bot_assets ORDER BY id")
    if not activos:
        return
    sel = st.selectbox("Activo", [a["symbol"] for a in activos], key="val_activo")
    aid = next(a["id"] for a in activos if a["symbol"] == sel)
    run = db.consultar_uno("SELECT * FROM model_runs WHERE asset_id=? ORDER BY id DESC LIMIT 1", (aid,))
    if not run:
        st.info("Sin entrenamiento todavía.")
        return
    met, dm, hp = db.de_json(run["metrics_json"], {}), db.de_json(run["dm_results_json"], {}), db.de_json(run["hyperparams_json"], {})
    st.markdown(f"Entrenado `{run['ts'][:16]}` · {met.get('n_velas')} velas ({met.get('n_entrenamiento')} entrenamiento / {met.get('n_prueba')} prueba) · horizonte {met.get('horizonte_velas')} velas · mejores hiperparámetros: `{hp}`")
    tabla = pd.DataFrame(met.get("tabla_hiperparametros", []))
    if not tabla.empty:
        cols = [c for c in ["n_estimators", "max_depth", "min_samples_leaf", "accuracy", "precision", "recall", "f1", "rmse", "mse", "mae", "mape", "smape", "variance_score"] if c in tabla]
        st.markdown("**Simulación de hiperparámetros del RandomForest**")
        st.dataframe(tabla[cols].round(4), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matriz de confusión (mejor RF)**")
        cm = (met.get("mejor_rf") or {}).get("matriz_confusion") or tabla.iloc[0]["matriz_confusion"] if not tabla.empty else None
        if cm:
            st.dataframe(pd.DataFrame(cm, index=["Real: baja", "Real: sube"], columns=["Pred: baja", "Pred: sube"]), use_container_width=True)
        st.markdown("**Comparativa de modelos/estrategias (accuracy, F1)**")
        comp = {"random_forest": {k: met["mejor_rf"].get(k) for k in ("accuracy", "precision", "recall", "f1")}}
        comp.update({k: {m: v.get(m) for m in ("accuracy", "precision", "recall", "f1")} for k, v in (met.get("estrategias") or {}).items()})
        st.dataframe(pd.DataFrame(comp).T.round(3), use_container_width=True)
    with c2:
        st.markdown("**Diebold-Mariano (HLN) — RF vs otros**")
        st.dataframe(pd.DataFrame(dm).T[["dm_hln", "p_valor", "n", "mejor"]].round(4), use_container_width=True)
        st.caption("`modelo_1` = RandomForest. p < 0.05 indica diferencia significativa de precisión.")
        imp = pd.DataFrame(met.get("importancias", []))
        if not imp.empty:
            st.bar_chart(imp.set_index("feature")["importancia"])
    if st.button("📁 Exportar CSV/PNG para el informe"):
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
        st.success(f"Exportado a {SALIDA}")


def render() -> None:
    """Dibuja la pestaña."""
    st.header("🤖 Bot de trading")
    modo = "solo paper interno (forzar_paper)" if str(cfg("bot", "forzar_paper", default=False)).lower() == "true" else "Binance spot testnet · Alpaca Paper · paper interno"
    st.caption(f"Timeframe {cfg('bot', 'timeframe')} · capital virtual {cfg('bot', 'capital_inicial_por_estrategia')} USD por estrategia · venues: {modo} · señales con datos del Binance real")
    _buscador()
    filas = _tabla_estado()
    if filas:
        _senales_y_trades()
    _validacion()
