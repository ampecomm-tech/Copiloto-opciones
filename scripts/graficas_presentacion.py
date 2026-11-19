"""Gráficas en tema oscuro, estilo terminal de trading, para la presentación ejecutiva.

Usa las tablas de `docs/resultados/` (de `scripts/exportar_resultados.py`), la base de datos indicada y datos
de mercado en vivo para las velas. Escribe los PNG en `docs/presentacion/`.

Uso:
    python scripts/graficas_presentacion.py --db data/app.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
RES, SAL = RAIZ / "docs" / "resultados", RAIZ / "docs" / "presentacion"
FONDO, PANEL, REJILLA, TEXTO, SUAVE = "#0B0F17", "#0B0F17", "#1E2633", "#E6EDF6", "#8B98AC"
VERDE, ROJO, AZUL, AMBAR, VIOLETA = "#22C55E", "#EF4444", "#3B82F6", "#F59E0B", "#A78BFA"


def _estilo() -> None:
    """Tema oscuro común."""
    plt.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": PANEL, "savefig.facecolor": FONDO, "figure.dpi": 160, "savefig.bbox": "tight",
                         "axes.edgecolor": REJILLA, "axes.labelcolor": SUAVE, "xtick.color": SUAVE, "ytick.color": SUAVE, "text.color": TEXTO,
                         "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": TEXTO, "legend.frameon": False})


def _guardar(fig: plt.Figure, nombre: str) -> None:
    """Guarda y cierra la figura."""
    fig.savefig(SAL / nombre)
    plt.close(fig)


def velas_portada() -> None:
    """Velas de BTC de 15 minutos con EMA 50 y volumen: imagen de fondo de la portada."""
    from bot.features import ema
    from data_sources.market import velas

    df = velas("BTC/USDT", "binance_spot_testnet").tail(140)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0.18, 1, 0.82])
    av = fig.add_axes([0, 0, 1, 0.17], sharex=ax)
    x = np.arange(len(df))
    sube = df["close"] >= df["open"]
    col = np.where(sube, VERDE, ROJO)
    ax.vlines(x, df["low"], df["high"], color=col, lw=1.1, alpha=0.9)
    ax.bar(x, (df["close"] - df["open"]).abs().clip(lower=df["close"].mean() * 2e-5), bottom=np.minimum(df["open"], df["close"]), color=col, width=0.65)
    e = ema(velas("BTC/USDT", "binance_spot_testnet")["close"], 50).tail(140)
    ax.plot(x, e.values, color=AZUL, lw=2.2, alpha=0.9)
    av.bar(x, df["volume"], color=col, width=0.65, alpha=0.45)
    for a in (ax, av):
        a.axis("off")
    ax.set_xlim(-1, len(df))
    _guardar(fig, "portada_velas.png")


def bs_binance() -> None:
    """Precio y delta de Black-Scholes propio contra Binance Options."""
    t = pd.read_csv(RES / "bs_vs_binance.csv")
    fig, ej = plt.subplots(1, 2, figsize=(10, 4))
    ej[0].scatter(t["mark_price"], t["bs_precio"], s=10, color=AZUL, alpha=0.8)
    lim = [0, float(t[["mark_price", "bs_precio"]].max().max())]
    ej[0].plot(lim, lim, color=AMBAR, lw=1)
    ej[0].set(title="Precio: Black-Scholes propio vs mark", xlabel="Mark price Binance (USDT)", ylabel="Black-Scholes (USDT)")
    d = t.dropna(subset=["delta_bin"])
    ej[1].scatter(pd.to_numeric(d["delta_bin"]), d["delta_bs"], s=10, color=VERDE, alpha=0.8)
    ej[1].plot([-1, 1], [-1, 1], color=AMBAR, lw=1)
    ej[1].set(title="Delta propia vs Binance", xlabel="Delta Binance", ylabel="Delta Black-Scholes")
    fig.tight_layout()
    _guardar(fig, "bs_binance.png")


def diebold() -> None:
    """Estadístico DM con corrección HLN por activo y comparación, con las bandas de ±1.96."""
    t = pd.read_csv(RES / "diebold_mariano.csv")
    comps = list(dict.fromkeys(t["comparacion"]))
    activos = list(dict.fromkeys(t["activo"]))
    colores = [AZUL, VERDE, AMBAR, VIOLETA]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ancho = 0.8 / len(comps)
    for j, comp in enumerate(comps):
        v = t[t["comparacion"] == comp].set_index("activo").reindex(activos)["dm_hln"]
        ax.bar(np.arange(len(activos)) + (j - (len(comps) - 1) / 2) * ancho, v.values, ancho * 0.92, color=colores[j % 4], label=comp)
    ax.axhspan(-1.96, 1.96, color=SUAVE, alpha=0.12)
    ax.axhline(0, color=SUAVE, lw=0.8)
    ax.text(len(activos) - 0.45, 2.1, "zona sin diferencia significativa (|DM| < 1.96)", ha="right", fontsize=8, color=SUAVE)
    ax.set_xticks(range(len(activos)), activos)
    ax.set(title="Diebold-Mariano (HLN), h = 4 velas, n = 443 · negativo = el Random Forest es mejor", ylabel="Estadístico DM HLN")
    ax.legend(ncols=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _guardar(fig, "diebold.png")


def curvas(db: Path) -> None:
    """Capital virtual por estrategia en el Binance spot testnet."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    t = pd.read_sql_query("SELECT a.symbol, t.strategy, t.exit_ts, t.pnl, t.qty FROM bot_trades t JOIN bot_assets a ON a.id=t.asset_id "
                          "WHERE t.exit_ts IS NOT NULL AND t.qty > 0 AND a.venue='binance_spot_testnet' ORDER BY t.exit_ts", con)
    fig, ax = plt.subplots(figsize=(10, 4))
    paleta = [AZUL, VERDE, AMBAR, VIOLETA, "#22D3EE", "#F472B6", "#94A3B8", "#FB923C", "#4ADE80", "#60A5FA"]
    for i, ((s, e), g) in enumerate(t.groupby(["symbol", "strategy"])):
        if len(g) < 2:
            continue
        ax.step(pd.to_datetime(g["exit_ts"]), 1000 + g["pnl"].cumsum(), where="post", lw=1.6, color=paleta[i % len(paleta)], label=f"{s} · {e}")
    ax.axhline(1000, color=SUAVE, lw=0.8, ls="--")
    ax.set(title="Capital virtual por estrategia en el Binance spot testnet", ylabel="USD")
    ax.legend(ncols=3, fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    _guardar(fig, "curvas.png")


def _hist_var(ax: plt.Axes, x: np.ndarray, var: float, cvar: float, titulo: str, eje: str, normal: tuple[float, float] | None = None) -> None:
    """Histograma con la cola en rojo y las líneas de VaR y CVaR."""
    from scipy import stats

    if len(x) > 200:
        lo, hi = np.quantile(x, [0.002, 0.998])
        x = x[(x >= min(lo, -cvar * 1.1)) & (x <= hi)]
    h, b = np.histogram(x, bins=int(min(55, max(10, len(x) // 4))))
    c = (b[:-1] + b[1:]) / 2
    ax.bar(c, h, width=(b[1] - b[0]) * 0.9, color=[ROJO if v <= -var + (b[1] - b[0]) / 2 else AZUL for v in c])
    if normal:
        m = np.linspace(b[0], b[-1], 200)
        ax.plot(m, stats.norm.pdf(m, *normal) * len(x) * (b[1] - b[0]), color=SUAVE, ls=":", lw=1.4, label="Normal ajustada")
    ax.axvline(-var, color=AMBAR, ls="--", lw=1.8, label=f"VaR 95 %: {-var:,.3g}")
    ax.axvline(-cvar, color=ROJO, lw=1.8, label=f"CVaR 95 %: {-cvar:,.3g}")
    ax.set(title=titulo, xlabel=eje, ylabel="Frecuencia")
    ax.legend(fontsize=8)


def riesgo_bot() -> None:
    """Retornos de 15 minutos de BTC y NVDA con VaR y CVaR al 95 %."""
    from data_sources.market import velas
    from quant.riesgo import resumen_riesgo

    fig, ej = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (s, v) in zip(ej, (("BTC/USDT", "binance_spot_testnet"), ("NVDA", "alpaca_paper"))):
        r = (velas(s, v)["close"].pct_change().dropna() * 100).values
        rr = resumen_riesgo(r)
        _hist_var(ax, r, rr["var_95"], rr["cvar_95"], f"{s}: retornos de 15 min (curtosis {rr['curtosis']:.0f})", "Retorno por vela (%)", (rr["media"], rr["desviacion"]))
    fig.tight_layout()
    _guardar(fig, "var_bot.png")


def caviar() -> None:
    """VaR de CAViaR contra el VaR histórico móvil en BTC/USDT, con las violaciones."""
    from data_sources.market import velas
    from quant.caviar import analizar

    r = velas("BTC/USDT", "binance_spot_testnet")["close"].pct_change().dropna() * 100
    a = analizar(r.values)
    ini = max(0, a["inicio_prueba"] - 200)
    x = r.index[ini:]
    rr, q, qh = a["r"][ini:], a["q_caviar"][ini:], a["q_historico"][ini:]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axvspan(r.index[a["inicio_prueba"]], r.index[-1], color=AZUL, alpha=0.08)
    ax.text(r.index[a["inicio_prueba"]], float(np.nanmax(rr)) * 0.92, "  fuera de muestra", color=SUAVE, fontsize=8)
    ax.scatter(x, rr, s=4, color=SUAVE, alpha=0.7, label="Retorno por vela")
    ax.plot(x, qh, color=TEXTO, lw=1, ls=":", label="VaR histórico móvil")
    ax.plot(x, q, color=AZUL, lw=1.6, label="VaR CAViaR")
    v = rr < q
    ax.scatter(x[v], rr[v], s=22, color=ROJO, marker="x", label="Violación")
    k = a["kupiec_fuera"]
    ax.set(title=f"CAViaR en BTC/USDT: {k['tasa']:.1%} de violaciones fuera de muestra (objetivo 5 %, Kupiec p = {k['p_valor']:.2f})", ylabel="Retorno por vela (%)")
    ax.legend(ncols=4, fontsize=8, loc="lower left")
    fig.autofmt_xdate()
    _guardar(fig, "caviar.png")


def riesgo_copiloto() -> None:
    """Distribución del P&L de un iron condor y de un long put del copiloto, con VaR y CVaR."""
    from data_sources.market import cadena_para
    from quant.selector import Opinion, seleccionar

    cadena, S, meta = cadena_para("BTC", 0.045, 14)
    fig, ej = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (direc, mag) in zip(ej, (("bajista", 8), ("neutral", 3))):
        c = seleccionar(cadena, S, Opinion(activo="BTC", direccion=direc, magnitud_pct=mag, horizonte_dias=14, presupuesto=500), 0.045, meta["multiplicador"], meta["paso_qty"])[0]
        mc = c["montecarlo"]
        h = mc["histograma"]
        w = h["x"][1] - h["x"][0]
        ax.bar(h["x"], h["y"], width=w * 0.9, color=[ROJO if v <= -mc["var_95"] + w / 2 else (AMBAR if v < 0 else VERDE) for v in h["x"]])
        ax.axvline(-mc["var_95"], color=AMBAR, ls="--", lw=1.8, label=f"VaR 95 %: {-mc['var_95']:,.0f}")
        ax.axvline(-mc["cvar_95"], color=ROJO, lw=1.8, label=f"CVaR 95 %: {-mc['cvar_95']:,.0f}")
        ax.set(title=f"{c['nombre']}\nopinión {direc} · pérdida máx. {c['max_perdida']:,.0f} USD", xlabel="P&L al horizonte (USD)", ylabel="Trayectorias")
        ax.legend(fontsize=8)
    fig.tight_layout()
    _guardar(fig, "riesgo_copiloto.png")


def markowitz() -> None:
    """Correlación de Pearson y frontera eficiente de Markowitz."""
    import yfinance as yf

    from quant.portfolio import correlacion_pearson, frontera_eficiente

    tk = {"BTC": "BTC-USD", "SOL": "SOL-USD", "NVDA": "NVDA", "SPY": "SPY", "INTC": "INTC"}
    p = yf.download(list(tk.values()), period="1y", interval="1d", auto_adjust=True, progress=False)["Close"].rename(columns={v: k for k, v in tk.items()})[list(tk)].dropna()
    r = p.pct_change().dropna()
    corr, fr = correlacion_pearson(r), frontera_eficiente(r, 5000, periodos_por_anio=252)
    fig, ej = plt.subplots(1, 2, figsize=(10.5, 4))
    im = ej[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ej[0].set_xticks(range(len(corr)), corr.columns)
    ej[0].set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ej[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#0B0F17")
    ej[0].grid(False)
    ej[0].set_title("Correlación de Pearson (1 año)")
    c = fr["carteras"]
    sc = ej[1].scatter(c["vol"] * 100, c["ret"] * 100, c=c["sharpe"], cmap="viridis", s=4)
    ej[1].scatter(fr["max_sharpe"]["vol"] * 100, fr["max_sharpe"]["ret"] * 100, marker="*", s=220, color=AMBAR, label="Máximo Sharpe")
    ej[1].set(title="Frontera eficiente de Markowitz", xlabel="Volatilidad anual (%)", ylabel="Retorno anual (%)")
    ej[1].legend(fontsize=8)
    fig.colorbar(sc, ax=ej[1], fraction=0.046, label="Sharpe")
    fig.tight_layout()
    _guardar(fig, "markowitz.png")


def main() -> None:
    """Genera todas las gráficas de la presentación."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="Ruta a app.db")
    args = ap.parse_args()
    SAL.mkdir(parents=True, exist_ok=True)
    _estilo()
    velas_portada()
    bs_binance()
    diebold()
    curvas(Path(args.db))
    riesgo_bot()
    caviar()
    riesgo_copiloto()
    markowitz()
    print(f"Gráficas en {SAL}")


if __name__ == "__main__":
    main()
