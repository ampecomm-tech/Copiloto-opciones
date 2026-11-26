"""Exporta a CSV y PNG los resultados cuantitativos para el informe técnico y la presentación.

Lee la base de datos del sistema (por defecto `DB_PATH`) y genera en `docs/resultados/`:

- Simulación de hiperparámetros del RandomForest por activo (tabla y gráfica comparativa).
- Métricas de regresión y clasificación del mejor RF, del KNN base y de las estrategias técnicas.
- Matrices de confusión en el tramo de prueba y en vivo (señales reales del bot).
- Prueba de Diebold-Mariano con corrección HLN.
- Desempeño del bot por activo y estrategia (ROI, Sharpe, Sortino, drawdown, win rate).
- Black-Scholes propio contra el mark price, la IV y las griegas de Binance Options (en vivo).
- Correlación de Pearson y frontera eficiente de Markowitz de los activos del bot (en vivo).

Uso:
    python scripts/exportar_resultados.py [--db ruta/app.db] [--salida docs/resultados] [--sin-red]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from quant.portfolio import correlacion_pearson, frontera_eficiente, metricas_desempeno  # noqa: E402

AZUL, GRIS, ROJO, VERDE, TINTA = "#2563eb", "#94a3b8", "#dc2626", "#16a34a", "#0f172a"
NOMBRES = {"random_forest": "Random Forest", "ema50_chandelier": "EMA50 + Chandelier", "rsi_reversion": "Reversión RSI",
           "macd_cross": "Cruce MACD", "voting": "Votación", "knn": "KNN (base)", "donchian_swing": "Donchian (swing, 1d)",
           "ema_pullback_swing": "Retroceso a EMA20 (swing, 1d)", "rsi2_swing": "RSI2 (swing, 1d)"}


def _estilo() -> None:
    """Estilo común de las gráficas: fondo blanco, rejilla suave y tipografía legible."""
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.6, "axes.edgecolor": "#cbd5e1",
                         "axes.titleweight": "bold", "axes.titlesize": 10, "axes.labelcolor": TINTA, "text.color": TINTA})


def _leer(db: Path) -> dict[str, pd.DataFrame]:
    """Lee en solo lectura las tablas necesarias."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = lambda sql: pd.read_sql_query(sql, con)  # noqa: E731
    datos = {
        "runs": q("SELECT m.id, a.symbol, m.ts, m.hyperparams_json, m.metrics_json, m.dm_results_json FROM model_runs m JOIN bot_assets a ON a.id = m.asset_id ORDER BY m.id"),
        "trades": q("SELECT t.*, a.symbol, a.venue FROM bot_trades t JOIN bot_assets a ON a.id = t.asset_id"),
        "senales": q("SELECT s.ts, s.strategy, s.action, s.confidence, s.details_json, a.symbol FROM bot_signals s JOIN bot_assets a ON a.id = s.asset_id ORDER BY s.id"),
        "recs": q("SELECT id, ts, asset, view_json, verdict, chosen_strategy_json FROM recommendations ORDER BY id"),
        "eventos": q("SELECT channel, COUNT(*) AS n FROM events GROUP BY channel"),
        "opciones": q("SELECT id, asset, entry_ts, entry_cost, last_value, venue, status, legs_json FROM option_positions"),
    }
    con.close()
    datos["runs"] = datos["runs"].drop_duplicates("symbol", keep="last")
    return datos


def hiperparametros(runs: pd.DataFrame, salida: Path) -> pd.DataFrame:
    """Tabla y gráfica de la simulación de hiperparámetros del RF por activo."""
    tablas = []
    for _, r in runs.iterrows():
        t = pd.DataFrame(json.loads(r["metrics_json"])["tabla_hiperparametros"]).drop(columns=["matriz_confusion"], errors="ignore")
        t.insert(0, "activo", r["symbol"])
        tablas.append(t)
    tabla = pd.concat(tablas, ignore_index=True)
    tabla.round(4).to_csv(salida / "hiperparametros_rf.csv", index=False)
    activos = list(runs["symbol"])
    fig, ejes = plt.subplots(len(activos), 1, figsize=(10, 2.3 * len(activos)), sharey=True)
    for ax, activo in zip(np.atleast_1d(ejes), activos):
        t = tabla[tabla["activo"] == activo].copy()
        t["etiqueta"] = [f"{int(n)}·{('∞' if pd.isna(d) else int(d))}·{int(l)}" for n, d, l in zip(t["n_estimators"], t["max_depth"], t["min_samples_leaf"])]
        t = t.sort_values(["n_estimators", "max_depth", "min_samples_leaf"])
        x = np.arange(len(t))
        mejor = t["f1"].idxmax()
        ax.bar(x - 0.2, t["f1"], 0.4, color=[AZUL if i == mejor else "#93c5fd" for i in t.index], label="F1")
        ax.bar(x + 0.2, t["accuracy"], 0.4, color=GRIS, label="Accuracy")
        ax.axhline(0.5, color=ROJO, lw=0.8, ls="--")
        ax.set_xticks(x, t["etiqueta"], rotation=90, fontsize=6.5)
        ax.set_ylim(0, 0.8)
        ax.set_title(f"{activo}: F1 y accuracy por combinación (n_estimators·max_depth·min_samples_leaf)", loc="left")
    np.atleast_1d(ejes)[0].legend(loc="upper right", ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(salida / "hiperparametros_rf.png")
    plt.close(fig)
    return tabla


def metricas_modelos(runs: pd.DataFrame, salida: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Métricas del mejor RF, del KNN y de las estrategias técnicas; y resultados de Diebold-Mariano."""
    filas, dm_filas = [], []
    for _, r in runs.iterrows():
        m, dm = json.loads(r["metrics_json"]), json.loads(r["dm_results_json"])
        base = {"activo": r["symbol"], "velas": m["n_velas"], "prueba": m["n_prueba"], "horizonte_velas": m["horizonte_velas"]}
        filas.append({**base, "modelo": "Random Forest", "hiperparametros": r["hyperparams_json"],
                      **{k: v for k, v in m["mejor_rf"].items() if k not in ("n_estimators", "max_depth", "min_samples_leaf")}})
        for nombre, met in m["estrategias"].items():
            filas.append({**base, "modelo": NOMBRES.get(nombre, nombre), **met})
        for comp, res in dm.items():
            rival = comp.replace("rf_vs_", "")
            veredicto = {"modelo_1": "RF mejor", "modelo_2": f"{NOMBRES.get(rival, rival)} mejor", "empate": "Sin diferencia significativa"}[res["mejor"]]
            dm_filas.append({"activo": r["symbol"], "comparacion": f"RF vs {NOMBRES.get(rival, rival)}", "dm": res["dm"], "dm_hln": res["dm_hln"],
                             "p_valor": res["p_valor"], "n": res["n"], "h": res["horizonte"], "resultado": veredicto})
    met = pd.DataFrame(filas)
    met["matriz_confusion"] = met["matriz_confusion"].astype(str)
    met.round(4).to_csv(salida / "metricas_modelos.csv", index=False)
    dmt = pd.DataFrame(dm_filas)
    dmt.round(6).to_csv(salida / "diebold_mariano.csv", index=False)

    activos = list(runs["symbol"])
    fig, ejes = plt.subplots(1, len(activos), figsize=(2.4 * len(activos), 2.6))
    for ax, (_, r) in zip(np.atleast_1d(ejes), runs.iterrows()):
        cm = np.array(json.loads(r["metrics_json"])["mejor_rf"]["matriz_confusion"])
        _matriz(ax, cm, r["symbol"])
    fig.suptitle("Matriz de confusión del mejor RF en el tramo de prueba (30 % final)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida / "matrices_confusion_prueba.png")
    plt.close(fig)

    r = runs[runs["symbol"] == "BTC/USDT"].iloc[0] if (runs["symbol"] == "BTC/USDT").any() else runs.iloc[0]
    imp = pd.DataFrame(json.loads(r["metrics_json"])["importancias"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(imp["feature"], imp["importancia"], color=AZUL)
    ax.set_title(f"Importancia de variables del RF ({r['symbol']})", loc="left")
    ax.set_xlabel("Importancia (reducción media de impureza)")
    fig.savefig(salida / "importancias_rf.png")
    plt.close(fig)
    return met, dmt


def _matriz(ax: Any, cm: np.ndarray, titulo: str) -> None:
    """Dibuja una matriz de confusión 2x2 con conteos y porcentajes."""
    ax.imshow(cm, cmap="Blues")
    total = cm.sum() or 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cm[i, j] / total:.0%}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() * 0.6 else TINTA)
    ax.set_xticks([0, 1], ["Baja", "Sube"])
    ax.set_yticks([0, 1], ["Baja", "Sube"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    ax.grid(False)


def confusion_en_vivo(senales: pd.DataFrame, salida: Path, horizonte: int = 4) -> pd.DataFrame:
    """Matriz de confusión de las señales reales del RF en la VM contra la dirección observada `horizonte` velas después."""
    from validation.metrics import metricas_clasificacion

    rf = senales[senales["strategy"] == "random_forest"].copy()
    rf["det"] = rf["details_json"].map(json.loads)
    rf["close"] = rf["det"].map(lambda d: d.get("close"))
    rf["prob"] = rf["det"].map(lambda d: d.get("prob_subida"))
    filas, figs = [], []
    for activo, g in rf.groupby("symbol"):
        g = g.dropna(subset=["close", "prob"]).reset_index(drop=True)
        if len(g) <= horizonte + 10:
            continue
        futuro = g["close"].shift(-horizonte)
        valido = futuro.notna()
        y = (futuro[valido] > g["close"][valido]).astype(int).values
        pred = (g["prob"][valido] >= 0.5).astype(int).values
        m = metricas_clasificacion(y, pred)
        filas.append({"activo": activo, "senales_evaluadas": int(valido.sum()), **{k: v for k, v in m.items() if k != "matriz_confusion"}, "matriz_confusion": str(m["matriz_confusion"])})
        figs.append((activo, np.array(m["matriz_confusion"])))
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(salida / "confusion_en_vivo.csv", index=False)
    if figs:
        fig, ejes = plt.subplots(1, len(figs), figsize=(2.6 * len(figs), 2.6))
        for ax, (activo, cm) in zip(np.atleast_1d(ejes), figs):
            _matriz(ax, cm, activo)
        fig.suptitle(f"Señales reales del RF en la VM vs dirección {horizonte} velas después", fontweight="bold", fontsize=10)
        fig.tight_layout()
        fig.savefig(salida / "matrices_confusion_vivo.png")
        plt.close(fig)
    return tabla


def desempeno_bot(trades: pd.DataFrame, salida: Path, capital0: float = 1000.0) -> pd.DataFrame:
    """ROI, Sharpe y Sortino por operación, drawdown máximo y win rate por activo y estrategia.

    Las operaciones de Alpaca guardadas con cantidad cero (error corregido el 18 nov 2026) se cuentan aparte y no entran al desempeño.
    """
    cerradas = trades[trades["exit_ts"].notna()].copy()
    cerradas["minutos"] = (pd.to_datetime(cerradas["exit_ts"]) - pd.to_datetime(cerradas["entry_ts"])).dt.total_seconds() / 60
    filas, curvas = [], {}
    for (activo, estrategia, venue), g in cerradas.groupby(["symbol", "strategy", "venue"]):
        validas = g[g["qty"] > 0].sort_values("exit_ts")
        eq = pd.Series([capital0] + (capital0 + validas["pnl"].cumsum()).tolist())
        met = metricas_desempeno(eq, periodos_por_anio=1)
        filas.append({"activo": activo, "estrategia": NOMBRES.get(estrategia, estrategia), "venue": venue, "operaciones": len(g),
                      "operaciones_validas": len(validas), "cantidad_cero": int((g["qty"] <= 0).sum()),
                      "salida_en_vela_de_entrada": int(((g["minutos"] < 2) & g["exit_reason"].isin(["SL", "TP"])).sum()),
                      "pnl_usd": float(validas["pnl"].sum()), "roi_pct": met["roi"] * 100, "sharpe_por_operacion": met["sharpe"],
                      "sortino_por_operacion": met["sortino"], "max_drawdown_pct": met["max_drawdown"] * 100,
                      "win_rate_pct": float((validas["pnl"] > 0).mean() * 100) if len(validas) else np.nan})
        if len(validas) >= 2:
            curvas[f"{activo} · {NOMBRES.get(estrategia, estrategia)}"] = (pd.to_datetime(validas["exit_ts"]), capital0 + validas["pnl"].cumsum())
    tabla = pd.DataFrame(filas).sort_values(["activo", "estrategia"])
    tabla.round(3).to_csv(salida / "desempeno_bot.csv", index=False)
    if curvas:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for nombre, (x, y) in curvas.items():
            ax.step(x, y, where="post", lw=1.3, label=nombre)
        ax.axhline(capital0, color=GRIS, lw=0.8, ls="--")
        ax.set_title("Capital virtual por activo y estrategia (operaciones cerradas en testnet y paper)", loc="left")
        ax.set_ylabel("USD")
        ax.legend(fontsize=6.5, ncols=3, frameon=False, loc="upper left")
        fig.autofmt_xdate()
        fig.savefig(salida / "curvas_capital.png")
        plt.close(fig)
    return tabla


def bs_vs_binance(salida: Path, moneda: str = "BTC") -> pd.DataFrame:
    """Compara en vivo nuestro Black-Scholes con el mark price, la IV y las griegas de Binance Options."""
    from data_sources.binance_options import cadena_opciones
    from validation.bs_vs_binance import comparar, resumen

    cadena, S, _ = cadena_opciones(moneda)
    tabla = comparar(cadena, S)
    if tabla.empty:
        print("Binance Options no respondió: se omite la comparación BS vs Binance.")
        return tabla
    tabla.round(6).to_csv(salida / "bs_vs_binance.csv", index=False)
    res = resumen(tabla)
    pd.DataFrame([{"subyacente": moneda, "spot": S, **res}]).round(6).to_csv(salida / "bs_vs_binance_resumen.csv", index=False)
    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.3))
    ejes[0].scatter(tabla["mark_price"], tabla["bs_precio"], s=8, color=AZUL, alpha=0.7)
    lim = [0, float(max(tabla["mark_price"].max(), tabla["bs_precio"].max()))]
    ejes[0].plot(lim, lim, color=ROJO, lw=0.8)
    ejes[0].set(title="Precio: Black-Scholes propio vs mark", xlabel="Mark price Binance (USDT)", ylabel="Black-Scholes (USDT)")
    d = tabla.dropna(subset=["delta_bin"])
    ejes[1].scatter(pd.to_numeric(d["delta_bin"]), d["delta_bs"], s=8, color=AZUL, alpha=0.7)
    ejes[1].plot([-1, 1], [-1, 1], color=ROJO, lw=0.8)
    ejes[1].set(title="Delta propia vs Binance", xlabel="Delta Binance", ylabel="Delta Black-Scholes")
    ejes[2].hist(tabla["err_precio_pct"].clip(-5, 5), bins=40, color=AZUL)
    ejes[2].set(title="Error relativo del precio", xlabel="Error (%) recortado a ±5 %", ylabel="Contratos")
    fig.suptitle(f"{moneda}: {len(tabla)} contratos, spot {S:,.0f} USDT", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida / "bs_vs_binance.png")
    plt.close(fig)

    from quant.black_scholes import anios, griegas

    exp = sorted(cadena["expiry"].unique())
    venc = next((e for e in exp if 20 <= float(cadena[cadena["expiry"] == e]["dte"].iloc[0]) <= 45), exp[len(exp) // 2])
    c = cadena[(cadena["expiry"] == venc) & (cadena["type"] == "call")].dropna(subset=["mark_iv"]).sort_values("strike")
    c = c[(c["strike"] > S * 0.7) & (c["strike"] < S * 1.3)]
    g = [griegas(S, float(k), anios(float(t)), 0.045, float(v), "call") for k, t, v in zip(c["strike"], c["dte"], c["mark_iv"])]
    fig, ejes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, nombre in zip(ejes, ["delta", "gamma", "theta", "vega"]):
        ax.plot(c["strike"], [getattr(x, nombre) for x in g], color=AZUL, lw=1.5)
        ax.axvline(S, color=GRIS, lw=0.8, ls="--")
        ax.set_title(nombre.capitalize())
        ax.set_xlabel("Strike")
    fig.suptitle(f"Griegas por strike, calls {moneda} con vencimiento {venc}", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(salida / "griegas_por_strike.png")
    plt.close(fig)
    return tabla


def markowitz(salida: Path, tickers: dict[str, str] | None = None) -> dict[str, Any]:
    """Correlación de Pearson y frontera eficiente de Markowitz con un año de cierres diarios (yfinance)."""
    import yfinance as yf

    tickers = tickers or {"BTC": "BTC-USD", "SOL": "SOL-USD", "NVDA": "NVDA", "SPY": "SPY", "INTC": "INTC"}
    precios = yf.download(list(tickers.values()), period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]
    precios = precios.rename(columns={v: k for k, v in tickers.items()})[list(tickers)].dropna()
    rets = precios.pct_change().dropna()
    corr = correlacion_pearson(rets)
    corr.round(4).to_csv(salida / "correlacion_pearson.csv")
    fr = frontera_eficiente(rets, n_carteras=5000, periodos_por_anio=252)
    cart = fr["carteras"]
    pesos = pd.DataFrame([{"cartera": "Máximo Sharpe", **{k: v for k, v in fr["max_sharpe"].items()}},
                          {"cartera": "Mínima volatilidad", **{k: v for k, v in fr["min_vol"].items()}}])
    pesos.round(4).to_csv(salida / "markowitz_carteras.csv", index=False)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    im = ejes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ejes[0].set_xticks(range(len(corr)), corr.columns)
    ejes[0].set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ejes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ejes[0].set_title("Correlación de Pearson (retornos diarios, 1 año)")
    ejes[0].grid(False)
    fig.colorbar(im, ax=ejes[0], fraction=0.046)
    sc = ejes[1].scatter(cart["vol"] * 100, cart["ret"] * 100, c=cart["sharpe"], cmap="viridis", s=4)
    ejes[1].scatter(fr["max_sharpe"]["vol"] * 100, fr["max_sharpe"]["ret"] * 100, marker="*", s=180, color=ROJO, label="Máximo Sharpe")
    ejes[1].scatter(fr["min_vol"]["vol"] * 100, fr["min_vol"]["ret"] * 100, marker="D", s=60, color=TINTA, label="Mínima volatilidad")
    ejes[1].set(title="Frontera eficiente de Markowitz (5,000 carteras)", xlabel="Volatilidad anual (%)", ylabel="Retorno anual esperado (%)")
    ejes[1].legend(frameon=False, fontsize=7)
    fig.colorbar(sc, ax=ejes[1], fraction=0.046, label="Sharpe")
    fig.tight_layout()
    fig.savefig(salida / "markowitz_pearson.png")
    plt.close(fig)
    return {"corr": corr, "pesos": pesos}


def resumen_sistema(datos: dict[str, pd.DataFrame], salida: Path) -> dict[str, Any]:
    """Conteos de actividad del sistema y recomendaciones del copiloto."""
    recs = datos["recs"].copy()
    recs["opinion"] = recs["view_json"].map(lambda v: json.loads(v))
    recs["estrategia"] = recs["chosen_strategy_json"].map(lambda v: (json.loads(v) or {}).get("nombre") if isinstance(v, str) and v else None)
    tabla = pd.DataFrame([{"fecha": r.ts[:16].replace("T", " "), "activo": r.asset, "direccion": r.opinion.get("direccion"), "magnitud_pct": r.opinion.get("magnitud_pct"),
                           "horizonte_dias": r.opinion.get("horizonte_dias"), "presupuesto": r.opinion.get("presupuesto"), "semaforo": r.verdict, "estrategia_elegida": r.estrategia}
                          for r in recs.itertuples()])
    tabla.to_csv(salida / "recomendaciones_copiloto.csv", index=False)
    ops = datos["opciones"].drop(columns=["legs_json"])
    ops.round(2).to_csv(salida / "posiciones_opciones.csv", index=False)
    s = datos["senales"]
    res = {"eventos_por_canal": dict(zip(datos["eventos"]["channel"], datos["eventos"]["n"].astype(int))),
           "senales": int(len(s)), "senales_desde": s["ts"].min(), "senales_hasta": s["ts"].max(),
           "operaciones_bot": int(len(datos["trades"])), "recomendaciones": int(len(recs)), "posiciones_opciones": int(len(ops))}
    (salida / "resumen_sistema.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def _retornos(symbol: str, venue: str) -> pd.Series:
    """Retornos de 15 minutos en % del activo (Binance mainnet para cripto, yfinance para acciones)."""
    from data_sources.market import velas

    return velas(symbol, venue)["close"].pct_change().dropna() * 100


def _histo_var(ax: Any, x: np.ndarray, r: dict[str, Any], titulo: str, eje: str, normal: bool = False) -> None:
    """Histograma con la cola del 5 % en rojo, líneas de VaR y CVaR al 95 % y, opcionalmente, la normal ajustada."""
    from scipy import stats as sst

    if len(x) > 200:
        lo, hi = np.quantile(x, [0.002, 0.998])
        x = x[(x >= min(lo, -r["cvar_95"] * 1.1)) & (x <= hi)]
    h, b = np.histogram(x, bins=int(min(60, max(10, len(x) // 4))))
    c = (b[:-1] + b[1:]) / 2
    ax.bar(c, h, width=(b[1] - b[0]) * 0.9, color=[ROJO if v <= -r["var_95"] else AZUL for v in c])
    if normal:
        m = np.linspace(b[0], b[-1], 200)
        ax.plot(m, sst.norm.pdf(m, r["media"], r["desviacion"]) * len(x) * (b[1] - b[0]), color=GRIS, ls=":", lw=1.4, label="Normal ajustada")
    ax.axvline(-r["var_95"], color="#d97706", ls="--", lw=1.6, label=f"VaR 95 %: {-r['var_95']:.3f}")
    ax.axvline(-r["cvar_95"], color=ROJO, lw=1.6, label=f"CVaR 95 %: {-r['cvar_95']:.3f}")
    ax.set(title=titulo, xlabel=eje, ylabel="Frecuencia")
    ax.legend(frameon=False, fontsize=7)


def riesgo_bot(datos: dict[str, pd.DataFrame], salida: Path) -> pd.DataFrame:
    """VaR y CVaR (históricos y normales, 95 y 99 %) de los retornos de 15 min de cada activo y del P&L de sus operaciones."""
    from quant.riesgo import resumen_riesgo

    activos = datos["trades"][["symbol", "venue"]].drop_duplicates().values.tolist()
    filas, figs = [], []
    for symbol, venue in activos:
        try:
            r = _retornos(symbol, venue)
        except Exception as e:
            print(f"Sin velas para {symbol}: {e}")
            continue
        rr = resumen_riesgo(r.values)
        t = datos["trades"][(datos["trades"]["symbol"] == symbol) & datos["trades"]["exit_ts"].notna() & (datos["trades"]["qty"] > 0)]["pnl"].dropna()
        rt = resumen_riesgo(t.values) if len(t) >= 5 else {}
        filas.append({"activo": symbol, "velas": rr["n"], "var95_hist_pct": rr["var_95"], "cvar95_hist_pct": rr["cvar_95"], "var95_normal_pct": rr["var_95_normal"],
                      "cvar95_normal_pct": rr["cvar_95_normal"], "var99_hist_pct": rr["var_99"], "cvar99_hist_pct": rr["cvar_99"], "curtosis": rr["curtosis"],
                      "operaciones": len(t), "var95_operacion_usd": rt.get("var_95"), "cvar95_operacion_usd": rt.get("cvar_95")})
        if symbol in ("BTC/USDT", "NVDA"):
            figs.append((symbol, r.values, rr))
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(salida / "riesgo_bot.csv", index=False)
    if figs:
        fig, ejes = plt.subplots(1, len(figs), figsize=(5.4 * len(figs), 3.4))
        for ax, (s, x, rr) in zip(np.atleast_1d(ejes), figs):
            _histo_var(ax, x, rr, f"{s}: retornos de 15 min", "Retorno por vela (%)", normal=True)
        fig.tight_layout()
        fig.savefig(salida / "var_cvar_bot.png")
        plt.close(fig)
    return tabla


def caviar_resultados(datos: dict[str, pd.DataFrame], salida: Path) -> pd.DataFrame:
    """CAViaR (SAV, θ = 5 %) por activo: backtesting fuera de muestra y Diebold-Mariano contra el VaR histórico móvil."""
    from quant.caviar import analizar

    filas, figura = [], None
    for symbol, venue in datos["trades"][["symbol", "venue"]].drop_duplicates().values.tolist():
        try:
            r = _retornos(symbol, venue)
            a = analizar(r.values, 0.05, "sav")
        except Exception as e:
            print(f"CAViaR omitido para {symbol}: {e}")
            continue
        kf, kd, ch, dm = a["kupiec_fuera"], a["kupiec_dentro"], a["christoffersen_fuera"], a.get("dm_vs_historico") or {}
        filas.append({"activo": symbol, "beta0": a["modelo_ultimo"]["beta"][0], "beta1": a["modelo_ultimo"]["beta"][1], "beta2": a["modelo_ultimo"]["beta"][2],
                      "violaciones_dentro_pct": kd["tasa"] * 100, "violaciones_fuera_pct": kf["tasa"] * 100, "kupiec_p": kf["p_valor"], "christoffersen_p": ch["p_cc"],
                      "violaciones_historico_pct": (a.get("kupiec_historico_fuera") or {}).get("tasa", np.nan) * 100,
                      "perdida_cuantil_caviar": (a.get("perdida_cuantil_fuera") or {}).get("caviar"), "perdida_cuantil_historico": (a.get("perdida_cuantil_fuera") or {}).get("historico"),
                      "dm_hln": dm.get("dm_hln"), "dm_p": dm.get("p_valor"),
                      "dm_resultado": {"modelo_1": "CAViaR mejor", "modelo_2": "Histórico mejor"}.get(dm.get("mejor"), "Sin diferencia significativa")})
        if symbol == "BTC/USDT":
            figura = (r.index, a)
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(salida / "caviar.csv", index=False)
    if figura:
        x, a = figura
        n = len(a["r"])
        ini = max(0, a["inicio_prueba"] - 250)
        xs = x[ini:n]
        r, q, qh = a["r"][ini:], a["q_caviar"][ini:], a["q_historico"][ini:]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.axvspan(x[a["inicio_prueba"]], x[-1], color=AZUL, alpha=0.06, label="Fuera de muestra")
        ax.scatter(xs, r, s=3, color=GRIS, label="Retorno por vela")
        ax.plot(xs, qh, color=TINTA, lw=1, ls=":", label="VaR histórico móvil")
        ax.plot(xs, q, color=AZUL, lw=1.4, label="VaR CAViaR")
        v = r < q
        ax.scatter(xs[v], r[v], s=16, color=ROJO, marker="x", label="Violación")
        ax.set(title="BTC/USDT: VaR al 95 % de CAViaR (valor absoluto simétrico) contra el VaR histórico", ylabel="Retorno por vela (%)")
        ax.legend(ncols=5, fontsize=7, frameon=False, loc="lower left")
        fig.autofmt_xdate()
        fig.savefig(salida / "caviar_btc.png")
        plt.close(fig)
    return tabla


def riesgo_copiloto(salida: Path) -> pd.DataFrame:
    """Ejemplo en vivo del copiloto en BTC (14 días, 500 USD; opiniones bajista 8 % y neutral 3 %): VaR y CVaR del Monte Carlo de las mejores candidatas."""
    from data_sources.market import cadena_para
    from quant.selector import Opinion, seleccionar

    cadena, S, meta = cadena_para("BTC", 0.045, 14)
    filas, todas = [], []
    for direccion, mag in (("bajista", 8), ("neutral", 3)):
        opinion = Opinion(activo="BTC", direccion=direccion, magnitud_pct=mag, horizonte_dias=14, presupuesto=500)
        for c in seleccionar(cadena, S, opinion, 0.045, meta["multiplicador"], meta["paso_qty"]):
            c["opinion"] = f"{direccion} {mag} %"
            todas.append(c)
    for c in todas:
        mc = c["montecarlo"]
        filas.append({"opinion": c["opinion"], "estrategia": c["nombre"], "costo_neto": c["costo_neto"], "perdida_maxima": c["max_perdida"], "prob_ganancia": mc["prob_ganancia"],
                      "valor_esperado": mc["valor_esperado"], "var95": mc["var_95"], "cvar95": mc["cvar_95"], "var99": mc["var_99"], "cvar99": mc["cvar_99"],
                      "cvar99_sobre_perdida_max": mc["cvar_99"] / max(c["max_perdida"], 1e-9)})
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(salida / "riesgo_copiloto.csv", index=False)
    if todas:
        c = next((x for x in todas if x["montecarlo"]["var_95"] < 0.98 * x["max_perdida"]), todas[0])
        mc = c["montecarlo"]
        fig, ejes = plt.subplots(1, 2, figsize=(11, 3.5))
        h = mc["histograma"]
        ancho = h["x"][1] - h["x"][0]
        ejes[0].bar(h["x"], h["y"], width=ancho * 0.9, color=[ROJO if v <= -mc["var_95"] + ancho / 2 else ("#f59e0b" if v < 0 else VERDE) for v in h["x"]])
        ejes[0].axvline(-mc["var_95"], color="#d97706", ls="--", lw=1.6, label=f"VaR 95 %: {-mc['var_95']:,.0f}")
        ejes[0].axvline(-mc["cvar_95"], color=ROJO, lw=1.6, label=f"CVaR 95 %: {-mc['cvar_95']:,.0f}")
        ejes[0].set(title=f"{c['nombre']}: P&L simulado (10,000 trayectorias)", xlabel="P&L al horizonte (USD)", ylabel="Trayectorias")
        ejes[0].legend(frameon=False, fontsize=7)
        q = mc["cuantiles_pnl"]
        ejes[1].plot(q[:100], range(100), color=AZUL, lw=2)
        ejes[1].axhline(5, color="#d97706", ls="--", lw=1)
        ejes[1].axvline(0, color=GRIS, lw=0.8)
        ejes[1].set(title=f"Probabilidad acumulada · prob. de ganancia {mc['prob_ganancia']:.0%}", xlabel="P&L al horizonte (USD)", ylabel="P(P&L ≤ x) en %")
        fig.tight_layout()
        fig.savefig(salida / "riesgo_copiloto.png")
        plt.close(fig)
    return tabla


def main() -> None:
    """Punto de entrada: exporta todo a la carpeta de salida."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None, help="Ruta a app.db (por defecto DB_PATH)")
    ap.add_argument("--salida", default=str(RAIZ / "docs" / "resultados"))
    ap.add_argument("--sin-red", action="store_true", help="Omite BS vs Binance y Markowitz (requieren internet)")
    args = ap.parse_args()
    from core.config import ruta_db

    db = Path(args.db) if args.db else ruta_db()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    _estilo()
    datos = _leer(db)
    hiperparametros(datos["runs"], salida)
    metricas_modelos(datos["runs"], salida)
    confusion_en_vivo(datos["senales"], salida)
    desempeno_bot(datos["trades"], salida)
    print(json.dumps(resumen_sistema(datos, salida), ensure_ascii=False, default=str))
    if not args.sin_red:
        bs_vs_binance(salida)
        markowitz(salida)
        riesgo_bot(datos, salida)
        caviar_resultados(datos, salida)
        riesgo_copiloto(salida)
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()
