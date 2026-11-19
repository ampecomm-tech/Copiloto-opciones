# Copiloto de Opciones con IA

Proyecto final de Finanzas Computacionales. App web (Streamlit) donde el usuario se informa, formula una opinión, una **mesa de agentes de IA** le recomienda una estrategia de opciones con **semáforo de viabilidad**, y la posición se abre en **paper trading**. En paralelo un **bot de trading** corre 24/7 (testnet / paper) con varias estrategias y validación cuantitativa. Todo el razonamiento se publica en **Discord** y en la pestaña **Centro de control**.

> Nunca se opera con dinero real. La especificación completa está en [CLAUDE.md](CLAUDE.md).

**App en vivo:** http://34.51.52.155:8501 · VM de Google Cloud en Querétaro, servicios systemd 24/7.

## Entregables

| Entregable | Dónde |
|---|---|
| Informe técnico en PDF | [docs/Informe_Tecnico_Copiloto_Opciones.pdf](docs/Informe_Tecnico_Copiloto_Opciones.pdf) |
| Notebook de validación, ejecutado con datos reales | [notebooks/validacion_cuantitativa.ipynb](notebooks/validacion_cuantitativa.ipynb) |
| Tablas y gráficas de resultados (CSV y PNG) | [docs/resultados/](docs/resultados/) |
| Código documentado con docstrings en español | `app/`, `core/`, `quant/`, `agents/`, `bot/`, `validation/` |
| Pruebas automáticas (35) | `tests/` |

## Opciones del curso que cubre

| Opción | Implementación | Archivos |
|---|---|---|
| 6. App financiera con IA | Streamlit con Firebase Authentication, URL pública, Gemini y Together AI, validación de ML en pantalla y panel del bot de la VM | `app/`, `agents/` |
| 7. Black-Scholes con agentes | Precio, griegas por strike, IV por Brent y Newton, IV contra HV, mesa de agentes y alertas de cobertura | `quant/black_scholes.py`, `agents/mesa.py`, `core/positions.py` |
| 3. Bot en VM de Google Cloud | Random Forest con simulación de 27 hiperparámetros, RMSE, MAE, MAPE, SMAPE, Variance Score, matriz de confusión, F1 y Diebold-Mariano con corrección HLN; SL, TP, trailing y tamaño por ATR; Binance spot testnet y Alpaca Paper | `bot/`, `validation/` |
| 4. Agentes con roles | Analista de mercado, agente de sentimiento, abogado del diablo y gestor de riesgo como juez, con JSON estricto | `agents/roles.py`, `agents/mesa.py` |
| Extra | Correlación de Pearson y frontera eficiente de Markowitz | `quant/portfolio.py` |

## Resultados principales

Datos reales del 17 al 19 de noviembre de 2026, exportados desde la base de la VM.

| Resultado | Valor |
|---|---|
| Error mediano de nuestro Black-Scholes contra el mark price de Binance (400 contratos de BTC) | 0.0032 % |
| Correlación de nuestra delta con la de Binance | 1.0000 |
| F1 del mejor Random Forest en prueba, por activo | 0.467 a 0.575 |
| Diebold-Mariano, RF contra EMA50 + Chandelier | RF mejor en 5 de 5 activos (p < 0.001) |
| Diebold-Mariano, RF contra KNN | Sin diferencia significativa en 5 de 5 activos |
| P&L del bot en el Binance spot testnet (9,000 USD virtuales) | +2.22 USD, ROI de +0.025 % |

## Reproducir los resultados

```bash
pip install -r requirements.txt -r requirements-docs.txt
python scripts/exportar_resultados.py --db data/app.db   # tablas y gráficas en docs/resultados/
python scripts/crear_notebook.py                         # ejecuta el notebook de validación
python scripts/generar_informe.py                        # genera el PDF; portada en docs/portada.yaml
```

`exportar_resultados.py` necesita internet para la comparación con Binance Options y para Markowitz; con `--sin-red` exporta solo lo que está en la base.

## Puesta en marcha local

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # rellena las claves que tengas (todas son opcionales para arrancar)
streamlit run app/main.py   # app en http://localhost:8501
python -m bot.runner        # bot 24/7 (en otra terminal)
pytest                      # tests de Black-Scholes, Diebold-Mariano y semáforo
```

Sin claves la app funciona en **modo degradado**: login demo (sin Firebase), veredicto solo por reglas (sin Together ni Gemini), sentimiento neutro (sin xAI), etiquetas de noticias por léxico (sin Gemini), opciones de acciones por yfinance (sin Alpaca), ejecución y bot en paper interno (sin testnet ni Alpaca) y sin Discord (los eventos igual quedan en la DB y en el Centro de control).

## Estructura

| Carpeta | Contenido |
|---|---|
| `app/` | Streamlit: `main.py` (login + navegación), `auth.py` (Firebase), `ui.py`, `pages/1..5` |
| `core/` | `config.py`, `db.py` (SQLite WAL), `events.py` (bus → DB + Discord + log), `tickers.py`, `positions.py`, `copiloto.py`, `execution.py` (órdenes al confirmar) |
| `scripts/` | `check_apis.py` (prueba cada conexión externa), `exportar_resultados.py`, `crear_notebook.py` y `generar_informe.py` (entregables) |
| `notebooks/` | `validacion_cuantitativa.ipynb`: Black-Scholes, Monte Carlo, RF, KNN y Diebold-Mariano con datos reales |
| `docs/` | Informe técnico en PDF, datos de portada y `resultados/` con tablas CSV y gráficas PNG |
| `data_sources/` | `binance_options.py` (BTC, ETH, BNB, SOL, XRP, DOGE), `alpaca.py` (acciones + órdenes paper), `deribit.py` (órdenes de opciones cripto en testnet), `yf_options.py` (respaldo), `market.py` (velas, IV), `news.py` |
| `quant/` | `black_scholes.py`, `volatility.py`, `strategies_options.py`, `selector.py`, `montecarlo.py`, `viability.py`, `portfolio.py` |
| `agents/` | `together_client.py` (mesa), `gemini_client.py` (apoyo + analista + respaldo), `grok_sentiment.py` (X), `roles.py`, `mesa.py` |
| `bot/` | `runner.py`, `executor.py`, `risk.py`, `features.py`, `trainer.py`, `strategies/` |
| `validation/` | `metrics.py`, `diebold_mariano.py` (con corrección HLN), `hyperparams.py`, `bs_vs_binance.py` (nuestro BS vs mark price de Binance) |
| `deploy/` | servicios systemd y `setup_vm.sh` |

## Flujo del copiloto

1. Cadena de opciones (Binance European Options para cripto; Alpaca para acciones, yfinance de respaldo) → IV por contrato (Brent) y griegas (Black-Scholes), HV del subyacente.
2. `selector.py` genera candidatas coherentes con la opinión (calls/puts, spreads, straddle/strangle, iron condor, butterfly), filtra por liquidez y presupuesto.
3. `montecarlo.py`: GBM con la IV (≥10,000 trayectorias) con la deriva implícita en la opinión del usuario, más un escenario neutral de referencia.
4. `viability.py`: semáforo determinista (VE, prob. de ganancia, IV/HV, liquidez, theta, sentimiento, presupuesto).
5. `agents/mesa.py`: analista, sentimiento y abogado del diablo en paralelo; el juez (gestor de riesgo) emite el veredicto, nunca mejor que el semáforo.
6. Confirmar → orden real en el testnet de Deribit (cripto BTC/ETH; alternativa: testnet de opciones de Binance) o en Alpaca Paper (acciones, multipata); si el venue rechaza una pata o falla, paper interno con aviso en #sistema. La posición queda en `option_positions` con su `venue` y los ids de orden. El bot refresca P&L/griegas y emite alertas.

## Bot

Busca activos por nombre (Yahoo + mercados de Binance vía ccxt) y asigna el venue solo: `binance_spot_testnet` para cripto listada en el testnet, `alpaca_paper` para acciones y ETFs de EE. UU., `paper` para el resto. Las señales se calculan con datos públicos del Binance real; solo la ejecución va al testnet. Entrena un RandomForest por activo con simulación de hiperparámetros, KNN de referencia y Diebold-Mariano contra las estrategias técnicas, y opera cada activo × estrategia con capital virtual propio, SL/TP por ATR y kill-switch diario. Activar/pausar desde la app no requiere reinicio. Al arrancar reconcilia la DB con el venue (el testnet de Binance se reinicia periódicamente) y cada día publica un resumen generado por Gemini en #sistema.

## Reparto de IA

- **Together AI**: la mesa de agentes (sentimiento, abogado del diablo, juez).
- **Gemini** (AI Studio, SDK `google-genai`): analista de mercado de la mesa, etiquetas y resumen de noticias, opinión estructurada, eventos macro con búsqueda en Google, explicación de cada operación del bot, modo educativo ("¿Qué significa esto?"), resumen diario y respaldo de la mesa si Together falla.
- **Grok (xAI)**: solo el sentimiento de X.

Para dejar todo en paper interno (sin órdenes a Binance ni Alpaca) pon `forzar_paper: true` en las secciones `bot` y `ejecucion` de `config.yaml`.

## Despliegue en VM

```bash
sudo bash deploy/setup_vm.sh
```

Instala Python 3.11, crea el venv, copia `copiloto-bot.service` y `copiloto-app.service` (ambos `Restart=always`, `EnvironmentFile=.env`) y los habilita. Abrir el puerto 8501 en el firewall de GCP para la URL pública.

**Región de la VM**: Binance (mainnet, testnet y Options) bloquea las IPs de Estados Unidos con error 451. La VM debe crearse fuera de EE. UU.; la del proyecto está en `northamerica-south1` (Querétaro, México), desde donde todas las APIs responden. Comandos usados:

```bash
gcloud compute instances create copiloto-mx --zone=northamerica-south1-a --machine-type=e2-medium --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=30GB --tags=copiloto
gcloud compute firewall-rules create copiloto-8501 --allow=tcp:8501 --target-tags=copiloto --source-ranges=0.0.0.0/0
```

Para actualizar el código en la VM: `cd /opt/copiloto-opciones && sudo git pull && sudo systemctl restart copiloto-bot copiloto-app`.
