# Copiloto de Opciones con IA

Proyecto final de Finanzas Computacionales. App web (Streamlit) donde el usuario se informa, formula una opinión, una **mesa de agentes de IA** le recomienda una estrategia de opciones con **semáforo de viabilidad**, y la posición se abre en **paper trading**. En paralelo un **bot de trading** corre 24/7 (testnet / paper) con varias estrategias y validación cuantitativa. Todo el razonamiento se publica en **Discord** y en la pestaña **Centro de control**.

> Nunca se opera con dinero real. La especificación completa está en [CLAUDE.md](CLAUDE.md).

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
| `scripts/` | `check_apis.py`: prueba cada conexión externa y muestra una tabla ✅/❌ |
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
