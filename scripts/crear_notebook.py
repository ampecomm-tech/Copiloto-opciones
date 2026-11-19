"""Genera y ejecuta `notebooks/validacion_cuantitativa.ipynb` con los cálculos clave del proyecto.

Uso:
    python scripts/crear_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

CELDAS = [
    md("# Validación cuantitativa del Copiloto de Opciones con IA\n\n"
       "Este notebook reproduce, con datos reales descargados en el momento, los cálculos que sostienen el informe técnico:\n\n"
       "1. Black-Scholes y letras griegas contra valores de referencia de la literatura.\n"
       "2. Nuestro Black-Scholes contra el *mark price*, la IV y las griegas que publica Binance Options.\n"
       "3. Monte Carlo (GBM) de una estrategia de opciones.\n"
       "4. Random Forest del bot: simulación de hiperparámetros, métricas de regresión y clasificación, KNN de referencia y prueba de Diebold-Mariano con corrección HLN.\n"
       "5. Resultados obtenidos por el sistema en la VM de Google Cloud (exportados con `scripts/exportar_resultados.py`)."),
    code("import sys, json, warnings\nfrom pathlib import Path\nwarnings.filterwarnings('ignore')\n"
         "RAIZ = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(RAIZ))\n"
         "import numpy as np, pandas as pd\nimport matplotlib.pyplot as plt\npd.set_option('display.width', 160)\npd.set_option('display.max_columns', 20)\n"
         "RES = RAIZ / 'docs' / 'resultados'"),
    md("## 1. Black-Scholes contra valores conocidos\n\n"
       "Caso clásico de Hull (S = 42, K = 40, r = 10 %, σ = 20 %, T = 0.5): call = 4.76 y put = 0.81. "
       "También comprobamos la paridad put-call, C − P = S − K·e^(−rT), y que la volatilidad implícita (Brent + Newton) recupera σ."),
    code("from quant.black_scholes import precio, griegas, volatilidad_implicita\n"
         "S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5\n"
         "c, p = precio(S, K, T, r, sigma, 'call'), precio(S, K, T, r, sigma, 'put')\n"
         "paridad = c - p - (S - K * np.exp(-r * T))\n"
         "iv = volatilidad_implicita(c, S, K, T, r, 'call')\n"
         "g = griegas(S, K, T, r, sigma, 'call')\n"
         "pd.DataFrame([{'call': round(c, 4), 'put': round(p, 4), 'error_paridad': paridad, 'iv_recuperada': iv,\n"
         "               'delta': g.delta, 'gamma': g.gamma, 'theta_dia': g.theta, 'vega_1pct': g.vega, 'rho_1pct': g.rho}])"),
    md("## 2. Black-Scholes propio contra Binance Options (en vivo)\n\n"
       "El *mark price* de Binance también sale de Black-Scholes con su IV de marca, así que las diferencias deben ser mínimas. "
       "Si la API no responde se usa la última exportación guardada en `docs/resultados/`."),
    code("from data_sources.binance_options import cadena_opciones\nfrom validation.bs_vs_binance import comparar, resumen\n"
         "cadena, S_btc, _ = cadena_opciones('BTC')\ntabla = comparar(cadena, S_btc) if not cadena.empty else pd.read_csv(RES / 'bs_vs_binance.csv')\n"
         "print(f'Spot BTC: {S_btc:,.2f} USDT · contratos comparados: {len(tabla)}')\npd.Series(resumen(tabla)).round(6)"),
    code("tabla[['symbol', 'strike', 'dte', 'mark_iv', 'bs_precio', 'mark_price', 'err_precio_pct', 'delta_bs', 'delta_bin']].head(10).round(4)"),
    md("## 3. Monte Carlo de una estrategia (GBM, 10,000 trayectorias)\n\n"
       "Precio final: S_T = S·exp((μ − σ²/2)T + σ√T·Z). Evaluamos un *bull call spread* sintético con strikes al 100 % y 110 % del spot."),
    code("from quant.montecarlo import simular_precios\nfrom quant.strategies_options import payoff_vencimiento\n"
         "S0, sig, T = 100.0, 0.5, 30 / 365\nST = simular_precios(S0, sig, T, 10_000, mu=0.0, semilla=1)\n"
         "legs = [{'type': 'call', 'strike': 100.0, 'side': 1, 'qty': 1, 'price': precio(S0, 100, T, 0.045, sig)},\n"
         "        {'type': 'call', 'strike': 110.0, 'side': -1, 'qty': 1, 'price': precio(S0, 110, T, 0.045, sig)}]\n"
         "costo = legs[0]['price'] - legs[1]['price']\npnl = payoff_vencimiento(legs, ST) - costo\n"
         "print(f'Costo neto {costo:.2f} · prob. de ganancia {np.mean(pnl > 0):.1%} · valor esperado {pnl.mean():.3f} · P5 {np.percentile(pnl, 5):.2f} · P95 {np.percentile(pnl, 95):.2f}')\n"
         "plt.figure(figsize=(6, 2.6)); plt.hist(pnl, bins=50, color='#2563eb'); plt.title('P&L al vencimiento (Monte Carlo)'); plt.xlabel('USD por unidad'); plt.show()"),
    md("## 4. Random Forest del bot: hiperparámetros, métricas y Diebold-Mariano\n\n"
       "Velas de 15 minutos de BTC/USDT (Binance mainnet, datos públicos). Objetivo: si el cierre sube dentro de 4 velas. "
       "Partición temporal 70/30 sin barajar, para no filtrar información del futuro."),
    code("from data_sources.market import velas\nfrom bot.features import construir_features, etiqueta_objetivo, retorno_futuro\n"
         "from validation.hyperparams import simular_grid, comparar_con_knn\nfrom validation.diebold_mariano import diebold_mariano\n"
         "df = velas('BTC/USDT', 'binance_spot_testnet', '15m', 1500)\nX, y, ret = construir_features(df), etiqueta_objetivo(df, 4), retorno_futuro(df, 4)\n"
         "ok = X.notna().all(axis=1) & y.notna()\nX, y, ret = X[ok], y[ok].astype(int), ret[ok]\n"
         "print(f'{len(df)} velas desde {df.index[0]} hasta {df.index[-1]} · {X.shape[1]} variables')\n"
         "tabla_hp, mejor = simular_grid(X, y, ret)\ntabla_hp.drop(columns=['matriz_confusion']).head(10).round(4)"),
    code("fig, ax = plt.subplots(figsize=(10, 3))\nt = tabla_hp.sort_values(['n_estimators', 'max_depth', 'min_samples_leaf'])\n"
         "etq = [f'{a}·{b}·{c}' for a, b, c in zip(t.n_estimators, t.max_depth, t.min_samples_leaf)]\n"
         "ax.bar(np.arange(len(t)) - 0.2, t.f1, 0.4, label='F1', color='#2563eb'); ax.bar(np.arange(len(t)) + 0.2, t.accuracy, 0.4, label='Accuracy', color='#94a3b8')\n"
         "ax.set_xticks(range(len(t)), etq, rotation=90, fontsize=7); ax.axhline(0.5, color='#dc2626', ls='--', lw=0.8); ax.legend(); ax.set_title('Simulación de hiperparámetros del RF'); plt.show()\n"
         "print('Mejor combinación:', mejor['hyperparams'])"),
    code("knn = comparar_con_knn(X, y, ret)\nfilas = [{'modelo': 'Random Forest', **{k: mejor[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'rmse', 'mse', 'mae', 'mape', 'smape', 'variance_score']}},\n"
         "         {'modelo': 'KNN (k=15)', **{k: knn[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'rmse', 'mse', 'mae', 'mape', 'smape', 'variance_score']}}]\n"
         "display(pd.DataFrame(filas).round(4))\nprint('Matriz de confusión RF [[TN, FP], [FN, TP]]:', mejor['matriz_confusion'])\n"
         "dm = diebold_mariano(mejor['_error'], knn['error'], h=4)\npd.Series(dm.to_dict())"),
    md("Interpretación de Diebold-Mariano: H0 es igual precisión predictiva. Un estadístico HLN negativo con p < 0.05 indica que el RF tiene menor pérdida cuadrática; "
       "con p ≥ 0.05 no hay evidencia para preferir un modelo sobre el otro."),
    md("## 5. Resultados registrados por el sistema en la VM\n\nTablas exportadas desde la base de datos de producción con `scripts/exportar_resultados.py`."),
    code("display(pd.read_csv(RES / 'diebold_mariano.csv').round(4))\ndisplay(pd.read_csv(RES / 'confusion_en_vivo.csv').round(4))\npd.read_csv(RES / 'desempeno_bot.csv').round(3)"),
    code("display(pd.read_csv(RES / 'correlacion_pearson.csv', index_col=0).round(3))\npd.read_csv(RES / 'markowitz_carteras.csv').round(4)"),
]


def main() -> None:
    """Crea el notebook, lo ejecuta desde `notebooks/` y lo guarda con sus salidas."""
    nb = nbformat.v4.new_notebook(cells=CELDAS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}})
    destino = RAIZ / "notebooks" / "validacion_cuantitativa.ipynb"
    destino.parent.mkdir(exist_ok=True)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(destino.parent)}}).execute()
    nbformat.write(nb, destino)
    print(f"Notebook ejecutado en {destino}")


if __name__ == "__main__":
    main()
