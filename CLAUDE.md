# Copiloto de Opciones con IA — Especificación del proyecto

> Proyecto final de Finanzas Computacionales. Presentación con demo en vivo: **jueves 19 de noviembre de 2026** (20–25 min).
> Este archivo es la fuente de verdad del proyecto. Léelo completo antes de escribir código.

## 1. Resumen

App web (una sola URL, un solo login) donde el usuario:

1. Se informa con noticias y sentimiento de X sobre un activo.
2. Formula una opinión (dirección, magnitud, horizonte, presupuesto).
3. Una **mesa de agentes de IA** le recomienda la estrategia de opciones que mejor encaja, con un **semáforo de viabilidad**.
4. Confirma y la posición se abre en **paper trading**, con seguimiento y alertas de cobertura.

En paralelo, un **bot de trading** corre 24/7 en una VM de Google Cloud, sobre cualquier activo que el usuario busque por nombre, con varias estrategias seleccionables.

Todo el razonamiento interno (agentes y bot) se publica en **Discord** (canales separados) y en una pestaña **Centro de control** dentro de la app.

**Opciones del documento del curso que cubre:** 6 (app interactiva con IA), 7 (Black-Scholes + griegas + agentes), 3 (bot en VM), con arquitectura multiagente estilo Opción 4 y Markowitz/Pearson.

**Nunca se opera con dinero real.** Solo testnet y paper trading.

## 2. Stack

| Capa | Tecnología |
|---|---|
| Lenguaje | Python 3.11 |
| Frontend | Streamlit (+ Plotly para gráficas) |
| Base de datos | SQLite (archivo `data/app.db`), compartida por app y bot |
| Autenticación | Firebase Authentication (email/contraseña vía REST API de Identity Toolkit) |
| Mesa de agentes | Together AI (SDK `together`), un modelo distinto por agente; el analista de mercado corre con Gemini |
| Sentimiento | **Intercambiable** (`agents/sentiment.py`): Gemini + búsqueda en Google (por defecto) o X vía Grok (`x_search`, opcional si hay `XAI_API_KEY`) |
| IA de apoyo | Gemini (API de AI Studio, SDK `google-genai`, **no** Vertex AI): noticias, opinión estructurada, eventos con grounding, explicaciones, resumen diario, respaldo de la mesa |
| Opciones cripto | Binance European Options API (`https://eapi.binance.com`): BTC, ETH, BNB, SOL, XRP, DOGE (liquidación en USDT, estilo europeo) |
| Opciones acciones | API de datos de opciones de Alpaca; respaldo: yfinance (`Ticker.options`, `Ticker.option_chain`) |
| Ejecución copiloto | Cripto: **testnet de Deribit** (`https://test.deribit.com`, BTC y ETH; por defecto) o testnet de opciones de Binance (`config.ejecucion.venue_cripto`); acciones/ETFs: Alpaca Paper (`https://paper-api.alpaca.markets`); respaldo: paper interno |
| Bot (cripto) | ccxt sobre Binance spot testnet (`testnet.binance.vision`) para la ejecución; señales con datos públicos del Binance real (mainnet, sin llave) |
| Bot (acciones/ETFs de EE. UU.) | Alpaca Paper |
| Bot (todo lo demás) | Paper trading interno con precios de yfinance (materias primas, cripto no listada, etc.) |
| Noticias | yfinance `Ticker.news` + RSS (CoinDesk, Cointelegraph) |
| ML | scikit-learn (RandomForest, KNN como base), numpy, pandas, scipy |
| Notificaciones | Discord webhooks (uno por canal) |
| Despliegue | VM Compute Engine e2-medium, Ubuntu 22.04, systemd |

## 3. Estructura de carpetas

```
copiloto-opciones/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml                 # parámetros: modelos por agente, intervalos, riesgo, activos por defecto
├── app/
│   ├── main.py                 # entrada de Streamlit, login y navegación por pestañas
│   ├── auth.py                 # Firebase Auth
│   └── pages/
│       ├── 1_noticias.py
│       ├── 2_copiloto.py
│       ├── 3_posiciones.py
│       ├── 4_bot.py
│       └── 5_centro_control.py
├── core/
│   ├── db.py                   # conexión SQLite, creación de tablas, helpers
│   ├── events.py               # bus de eventos: DB + Discord + log
│   ├── config.py               # carga .env y config.yaml
│   └── tickers.py              # búsqueda por nombre -> ticker
├── data_sources/
│   ├── binance_options.py      # cadena de opciones de Binance (eapi): mark price, IV y griegas publicadas
│   ├── alpaca.py               # datos de opciones de acciones + órdenes en Alpaca Paper
│   ├── deribit.py              # ejecución de opciones cripto en el testnet de Deribit (mapeo de patas Binance → Deribit)
│   ├── yf_options.py           # respaldo para opciones de acciones
│   ├── market.py               # velas OHLCV (ccxt / yfinance)
│   └── news.py
├── quant/
│   ├── black_scholes.py        # precio, griegas, volatilidad implícita (Newton/Brent)
│   ├── volatility.py           # vol histórica, IV vs HV
│   ├── strategies_options.py   # definición de estrategias multi-pata y su payoff
│   ├── selector.py             # genera y rankea candidatas
│   ├── montecarlo.py           # prob. de ganancia, valor esperado
│   ├── viability.py            # semáforo
│   └── portfolio.py            # griegas agregadas, Markowitz, Pearson
├── agents/
│   ├── together_client.py
│   ├── gemini_client.py        # SDK google-genai (AI Studio), reintentos con backoff y caché
│   ├── sentiment.py            # proveedores intercambiables: Gemini + Google Search (defecto) / Grok (opcional)
│   ├── grok_sentiment.py       # GrokSentimentProvider (xAI, x_search)
│   ├── roles.py                # prompts de cada agente
│   └── mesa.py                 # orquestación de la mesa
├── bot/
│   ├── runner.py               # loop principal 24/7 (proceso systemd)
│   ├── executor.py             # enruta órdenes por venue: binance_spot_testnet / alpaca_paper / paper
│   ├── risk.py                 # SL, TP, tamaño por ATR
│   ├── features.py             # indicadores técnicos
│   ├── trainer.py              # entrenamiento automático RF por activo
│   └── strategies/
│       ├── base.py
│       ├── random_forest.py
│       ├── ema50_chandelier.py
│       ├── rsi_reversion.py
│       ├── macd_cross.py
│       └── voting.py
├── validation/
│   ├── metrics.py              # RMSE, MSE, MAE, MAPE, SMAPE, Variance Score, F1, matriz de confusión
│   ├── diebold_mariano.py
│   └── hyperparams.py          # simulación de hiperparámetros RF + tabla comparativa
├── scripts/
│   └── check_apis.py           # prueba cada conexión con una llamada mínima y muestra tabla ✅/❌
├── deploy/
│   ├── copiloto-bot.service
│   ├── copiloto-app.service
│   └── setup_vm.sh
├── logs/                       # trading_bot.log, app.log
└── tests/
```

## 4. Variables de entorno (`.env`, nunca en git)

```
TOGETHER_API_KEY=
XAI_API_KEY=                      # opcional: si está vacía, el sentimiento usa Gemini
BINANCE_SPOT_TESTNET_API_KEY=
BINANCE_SPOT_TESTNET_SECRET=
BINANCE_FUTURES_TESTNET_API_KEY=
BINANCE_FUTURES_TESTNET_SECRET=
DERIBIT_TESTNET_CLIENT_ID=
DERIBIT_TESTNET_CLIENT_SECRET=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite
FIREBASE_API_KEY=
FIREBASE_PROJECT_ID=
DISCORD_WEBHOOK_MESA=
DISCORD_WEBHOOK_SENTIMIENTO=
DISCORD_WEBHOOK_BOT_SENALES=
DISCORD_WEBHOOK_BOT_OPERACIONES=
DISCORD_WEBHOOK_ALERTAS=
DISCORD_WEBHOOK_SISTEMA=
DB_PATH=data/app.db
```

## 5. Bus de eventos (pieza central)

Todo lo que pase en el sistema se registra con **una sola función**:

```python
def emit(channel: str, event_type: str, title: str, payload: dict, user_id: str | None = None) -> None:
    """Guarda el evento en la tabla `events`, lo manda al webhook de Discord del canal y lo escribe en el log."""
```

- Canales válidos: `mesa`, `sentimiento`, `bot_senales`, `bot_operaciones`, `alertas`, `sistema`.
- Discord: embed con título, color por tipo (verde/amarillo/rojo/gris) y campos del payload. Respetar límites de Discord (texto truncado a 1024 por campo, reintento con backoff ante 429).
- Si Discord falla, el evento **igual** queda en la DB (Discord nunca debe tumbar la app ni el bot).
- La pestaña Centro de control lee de `events` con filtros por canal, activo y fecha, autorrefresco cada ~5 s.

| Canal Discord | Contenido |
|---|---|
| #mesa-agentes | Cada recomendación: opinión de cada agente, veredicto del juez |
| #sentimiento-x | Puntaje de sentimiento, fuentes clave y proveedor usado ("Fuente: Gemini + Google Search" o "Fuente: X vía Grok") |
| #bot-señales | Por vela y estrategia: indicadores, probabilidad del RF, decisión |
| #bot-operaciones | Entradas, salidas, SL/TP, P&L, con explicación en lenguaje simple generada por Gemini |
| #alertas-posiciones | Cobertura delta, toma de ganancia, vencimiento |
| #sistema | Arranques, errores, estado de la VM, respaldos de ejecución (por qué una orden cayó a paper), resumen diario de Gemini |

## 6. Esquema de base de datos (mínimo)

- `events(id, ts, channel, event_type, title, payload_json, user_id)`
- `recommendations(id, ts, user_id, asset, view_json, candidates_json, agents_json, verdict, chosen_strategy_json)`
- `option_positions(id, user_id, recommendation_id, asset, legs_json, entry_ts, entry_cost, status, exit_ts, exit_value, pnl, venue, venue_order_ids_json)` — venue: `deribit_testnet` | `binance_options_testnet` | `alpaca_paper` | `paper`
- `sentiment_cache(asset, ts, score, summary, posts_json, provider)` — TTL 15 min
- `bot_assets(id, symbol, display_name, market_type, venue, active, added_ts)` — venue: `binance_spot_testnet` | `alpaca_paper` | `paper`
- `bot_strategy_state(asset_id, strategy, active, capital, position_json)`
- `bot_signals(id, ts, asset_id, strategy, action, confidence, details_json)`
- `bot_trades(id, asset_id, strategy, side, qty, entry_price, exit_price, sl, tp, entry_ts, exit_ts, pnl, venue_order_id)`
- `model_runs(id, asset_id, ts, hyperparams_json, metrics_json, dm_results_json)`

Usar modo WAL en SQLite para que app y bot lean/escriban sin bloquearse.

## 7. Módulos

### 7.1 Noticias y sentimiento (pestaña 1)
- Titulares del activo (yfinance news para acciones; RSS para cripto) con etiqueta de sentimiento por noticia generada por **Gemini**.
- **Sentimiento intercambiable** (`agents/sentiment.py`): dos proveedores con el mismo JSON `{score: -1..1, resumen, fuentes[], proveedor}`.
  - `GeminiSentimentProvider` (**por defecto**): Gemini con búsqueda en Google (grounding) sobre noticias y lo que se comenta del activo en las últimas 24-48 h; para cripto se combina con el índice Fear & Greed.
  - `GrokSentimentProvider` (opcional): xAI con `x_search`; solo se activa si `XAI_API_KEY` existe y no está vacía.
  - Selección automática: Grok si hay llave; si no, Gemini. Si Grok falla en tiempo de ejecución, caer a Gemini y emitir un evento a `sistema`.
  - Caché de 15 min por activo. Emitir a `sentimiento`. La app y el canal muestran el proveedor usado.
- Resumen del día (Gemini), próximos eventos (fecha de earnings de yfinance; eventos macro y contexto con **Gemini + búsqueda en Google (grounding)**, más los manuales en config), IV vs HV y Fear & Greed (cripto).
- Botón **"Formular opinión con esto"**: **Gemini** propone una opinión estructurada que se precarga en el copiloto (editable).

### 7.2 Copiloto de opciones (pestaña 2)
**Entrada:** activo, dirección (alcista/bajista/neutral/volátil), magnitud esperada (%), horizonte (días), presupuesto máx. de pérdida. También acepta texto libre, que **Gemini** convierte a esa estructura.

**Pipeline:**
1. Descargar cadena de opciones (Binance European Options para BTC, ETH, BNB, SOL, XRP, DOGE; Alpaca para acciones/ETFs, con yfinance de respaldo).
2. Calcular IV por contrato si no viene, griegas con Black-Scholes, HV del subyacente.
3. `selector.py`: generar candidatas según la opinión (long call/put, bull/bear spreads, straddle/strangle, iron condor, butterfly), probando strikes y vencimientos cercanos al horizonte. Filtrar por presupuesto y liquidez (spread bid-ask, open interest/volumen).
4. `montecarlo.py`: GBM con la IV, ≥10,000 trayectorias → prob. de ganancia, valor esperado, percentiles de P&L.
5. Rankear y quedarse con el top 3.
6. **Mesa de agentes** (ver 7.5) evalúa el top 3.
7. Mostrar: payoff al vencimiento y curva actual (Black-Scholes), griegas, máx. ganancia/pérdida, breakevens, prob. de ganancia, explicación de los agentes y semáforo.
8. Sliders "what-if": precio, IV, días transcurridos.
9. **Modo educativo**: botón "¿Qué significa esto?" junto a griegas, semáforo y estrategia; Gemini explica el caso concreto en lenguaje simple.
10. Botón **Confirmar** → crea `option_positions` con precios reales de ejecución (ask al comprar, bid al vender), no el teórico, y envía la orden al venue:
   - Cripto: órdenes reales en el **testnet de Deribit** (`https://test.deribit.com`, OAuth client_credentials; BTC y ETH). Cada pata de la cadena de Binance se mapea al instrumento de Deribit con el mismo vencimiento, strike y tipo; la prima en USD se convierte a la moneda base con el precio índice. Alternativa configurable: testnet de opciones de Binance (`demo-fapi`, llave del testnet de futuros).
   - Acciones/ETFs: órdenes en **Alpaca Paper** (`https://paper-api.alpaca.markets`), con soporte multipata (spreads, iron condor).
   - Si el venue rechaza una pata (por ejemplo, Binance restringe la venta de opciones en algunos subyacentes) o la API falla, se usa **paper trading interno** como respaldo y se emite un evento a `sistema` explicando por qué.
   - Se guardan `venue` y `venue_order_ids_json` en la posición.

**Semáforo (`viability.py`)** — reglas deterministas que alimentan al juez:
- Valor esperado (Monte Carlo) > 0
- Prob. de ganancia y relación ganancia/riesgo
- IV vs HV (penalizar comprar opciones caras / vender baratas)
- Liquidez (spread bid-ask relativo, OI)
- Theta vs horizonte
- Alineación con el sentimiento
- Pérdida máx. ≤ presupuesto

Salida: `verde | amarillo | rojo` + razones + alternativa sugerida si no es verde.

### 7.3 Mis posiciones (pestaña 3)
- P&L y griegas en vivo por posición (refrescadas por el proceso del bot cada pocos minutos).
- **Griegas del portafolio**: delta en dólares, theta diaria, vega total, sumando todos los activos.
- Alertas (emitir a `alertas`): delta desviada más allá de un umbral, 50% de ganancia máxima alcanzada, ≤3 días al vencimiento.
- Historial del copiloto: recomendación vs resultado real (tasa de acierto).

### 7.4 Bot de trading (pestaña 4)
- **Buscador por nombre** (`core/tickers.py`): "Solana", "Apple", "oro" → lista de coincidencias con ticker, nombre, mercado y precio.
  - Acciones/ETFs/materias primas: endpoint de búsqueda de Yahoo Finance (`yfinance.Search` o equivalente).
  - Cripto: buscar contra la lista de mercados de Binance vía ccxt (`load_markets`) para mapear a `XXX/USDT`.
  - El usuario selecciona → se crea `bot_assets` y el buscador asigna el venue automáticamente:
    - Cripto listada en el testnet → `binance_spot_testnet` (órdenes reales en `testnet.binance.vision`).
    - Acciones y ETFs de EE. UU. → `alpaca_paper`.
    - Cualquier otro (materias primas, cripto no listada, etc.) → `paper` (simulación interna con precios de yfinance).
- **Datos vs ejecución**: las señales se calculan con datos públicos del Binance real (mainnet, sin llave); solo la ejecución va al testnet, porque el libro de órdenes del testnet tiene poca liquidez y sus precios pueden desviarse.
- `bot/executor.py` enruta cada orden según el venue del activo (`binance_spot_testnet` / `alpaca_paper` / `paper`).
- Cada operación se publica en `bot_operaciones` con una explicación en lenguaje simple generada por Gemini.
- **Entrenamiento automático** al agregar un activo: descarga histórico, genera features, entrena RF, corre simulación de hiperparámetros y validación, guarda en `model_runs`. Mostrar progreso en la UI.
- **Interfaz de estrategia** (`bot/strategies/base.py`):

```python
@dataclass
class Signal:
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    details: dict   # indicadores y razonamiento, se publica en #bot-señales

class Strategy(Protocol):
    name: str
    def generate_signal(self, df: pd.DataFrame) -> Signal: ...
```

- Estrategias: `random_forest`, `ema50_chandelier`, `rsi_reversion`, `macd_cross`, `voting`.
- Cada combinación (activo × estrategia) tiene su propio capital virtual y se activa/pausa desde la app (el runner lee el estado de la DB en cada ciclo; no requiere reinicio).
- Riesgo (`risk.py`), igual para todas: SL y TP basados en ATR, trailing stop por ATR (se arma con la ganancia configurada y desde ahí reemplaza al TP fijo), tamaño de posición por volatilidad, pérdida diaria máxima (kill-switch). Cada vela cerrada se revisa una sola vez y nunca la vela de entrada.
- Features del RF: indicadores técnicos (EMA, RSI, MACD, ATR, Bollinger, retornos rezagados, volumen) + último puntaje de sentimiento si existe.
- Timeframe por defecto: 15 min (configurable).
- Tabla de posiciones: ROI, Sharpe, drawdown máx., win rate por combinación.
- Extra opcional: reparto de capital entre activos con Pearson + frontera eficiente de Markowitz.
- Log en `logs/trading_bot.log`.

### 7.5 Mesa de agentes
- Cliente Together AI; cada rol con un modelo distinto configurable en `config.yaml` (por ejemplo Llama, DeepSeek, Qwen). El **analista de mercado corre con Gemini** (configurable en `config.yaml`), para que la mesa mezcle modelos de distintas empresas. Si Together falla, Gemini actúa como respaldo de la mesa.
- Roles (prompts en `agents/roles.py`):
  1. **Analista de mercado**: interpreta griegas, IV vs HV, estructura de la estrategia.
  2. **Agente de sentimiento**: usa la salida cacheada del proveedor de sentimiento (Gemini o Grok) + noticias; dice si el mercado acompaña la opinión.
  3. **Abogado del diablo**: busca razones para no entrar.
  4. **Gestor de riesgo (juez)**: lee a los tres + reglas de `viability.py` y da el veredicto final.
- **Una sola ronda** (1–3 en paralelo, luego el juez). No debates multirronda.
- Todos responden en **JSON estricto** (validar con pydantic; reintentar una vez si no parsea).
- Los cálculos los hace Python; los agentes solo interpretan.
- La señal actual del RF del bot sobre ese activo (si existe) se pasa como contexto.
- Cada opinión y el veredicto se emiten a `mesa`.
- Timeouts y límite de tokens por llamada; si un agente falla, el juez decide con los que respondieron y lo indica.

### 7.6 Centro de control (pestaña 5)
Feed en vivo de `events` con filtros por canal; vista tipo chat con colores por tipo. Es la pantalla que se deja abierta en la demo.

### 7.7 Gemini (API de AI Studio, NO Vertex AI)
- Módulo `agents/gemini_client.py` con el SDK `google-genai` y `GEMINI_API_KEY`; modelo en `GEMINI_MODEL` (por defecto `gemini-3.1-flash-lite`).
- **Reparto de IA**: Gemini → tareas frecuentes y de apoyo (con créditos de Google Cloud) y el sentimiento por defecto; Together AI → la mesa de agentes; Grok (xAI) → sentimiento de X, opcional.
- Usos de Gemini:
  a) Noticias: etiquetar el sentimiento de cada titular y el resumen del día.
  b) Convertir el texto libre del usuario a la estructura de opinión del copiloto, y el botón "Formular opinión con esto".
  c) Próximos eventos macro y contexto con búsqueda en Google (grounding).
  d) Explicar en lenguaje simple cada operación del bot para #bot-operaciones.
  e) Modo educativo: botón "¿Qué significa esto?" junto a griegas, semáforo y estrategia, explicando el caso concreto.
  f) Resumen diario automático en Discord (#sistema): desempeño del bot, posiciones abiertas y alertas del día.
  g) Respaldo de la mesa de agentes si Together falla.
  h) Un agente de la mesa (el analista de mercado) corre con Gemini. Configurable en `config.yaml`.
- Manejar límites de peticiones con reintento y backoff; cachear respuestas repetidas.

### 7.8 Horarios y reseteos
- Alpaca solo ejecuta en horario de mercado de EE. UU.: si está cerrado, mostrar aviso en la app y dejar la orden en cola o cancelarla (configurable en `config.yaml`).
- El testnet de Binance se reinicia periódicamente: nuestra base de datos es la fuente de verdad de posiciones y operaciones; reconciliar al arrancar.

## 8. Validación cuantitativa (para el informe)
- **Opciones**: griegas por strike, IV vs HV, correlación de Pearson entre sentimiento e IV.
- **Black-Scholes vs Binance**: tabla comparativa de nuestro precio Black-Scholes, IV y griegas contra el mark price, la IV y las griegas que publica Binance Options (su mark price también usa Black-Scholes).
- **Bot, por activo**: tabla de simulación de hiperparámetros del RF (n_estimators, max_depth, min_samples_leaf…) con RMSE, MAE, MAPE, SMAPE, Variance Score; matriz de confusión, accuracy, precision, recall, F1.
- **Diebold-Mariano**: RF vs KNN y vs las demás estrategias (implementación propia con corrección de Harvey-Leybourne-Newbold).
- **Desempeño**: ROI, Sharpe, Sortino, drawdown máx.
- Todo se muestra en la pestaña del bot y se exporta a CSV/PNG para el informe en PDF.

## 9. Despliegue en la VM
- Compute Engine e2-medium, Ubuntu 22.04, 20–30 GB, **fuera de EE. UU.** (Binance devuelve 451 a IPs estadounidenses); se usa `northamerica-south1` (Querétaro).
- Dos servicios systemd: `copiloto-bot.service` (runner) y `copiloto-app.service` (Streamlit en 0.0.0.0:8501), ambos con `Restart=always` y `EnvironmentFile=.env`.
- Regla de firewall para el puerto de la app → URL pública para la clase.
- Alerta de presupuesto en GCP; límites de gasto en Together y xAI.
- `deploy/setup_vm.sh`: instala dependencias, crea venv, copia servicios, los habilita.

## 10. Convenciones
- Interfaz y mensajes al usuario en **español**.
- **Docstrings en español en todas las funciones** (el curso exige documentación a nivel de funciones y bloques lógicos).
- Type hints en todo.
- Ninguna API key en el código; todo por `.env`.
- Cada módulo de datos con caché y manejo de errores: una API caída nunca tumba la app.
- Parámetros (umbrales, intervalos, modelos) en `config.yaml`, no hardcodeados.
- Tests mínimos para `black_scholes.py` (contra valores conocidos), `diebold_mariano.py` y `viability.py`.

## 11. Plan y prioridades

| Día | Tarea |
|---|---|
| Vie 13 | Estructura del repo, DB, bus de eventos + webhooks de Discord, conexión de datos, Black-Scholes y griegas |
| Sáb 14 | Selector de estrategias, Monte Carlo, semáforo |
| Dom 15 | Mesa de agentes (Together) y sentimiento (Grok), publicando en Discord |
| Lun 16 | App Streamlit, Firebase, noticias, posiciones, centro de control |
| Mar 17 | Bot: buscador, entrenamiento automático, estrategias, testnet, validación; despliegue en VM |
| Mié 18 | Informe PDF, documentación, video de respaldo de la demo, ensayo |
| Jue 19 | Presentación |

**Prioridad si falta tiempo:** copiloto → bus de eventos/Discord/centro de control → noticias y posiciones → bot con BTC y 2 estrategias → buscador de activos → resto de estrategias → Markowitz.

## 12. Criterios de "listo para la demo"
- [ ] La clase entra a la URL pública, se registra e inicia sesión.
- [ ] Noticias + sentimiento (con su proveedor: Gemini + Google Search o X vía Grok) visibles para BTC y para una acción (p. ej. NVDA).
- [ ] "Formular opinión" precarga el copiloto.
- [ ] El copiloto devuelve top 3 con payoff, griegas, Monte Carlo y semáforo en < 30 s.
- [ ] El debate de la mesa aparece en #mesa-agentes y en el Centro de control.
- [ ] Confirmar abre la posición y aparece en Mis posiciones con griegas del portafolio.
- [ ] Confirmar una estrategia en BTC crea la orden en el testnet de Deribit (o de Binance, según configuración).
- [ ] Confirmar una estrategia en NVDA crea la orden multipata en Alpaca Paper.
- [ ] Buscar "Solana" en el bot → seleccionar → entrena y empieza a operar en el Binance spot testnet.
- [ ] Agregar AAPL al bot opera en Alpaca Paper.
- [ ] Tabla comparativa: nuestro Black-Scholes vs mark price de Binance.
- [ ] Cada operación del bot llega a Discord con explicación en lenguaje simple (Gemini).
- [ ] El resumen diario aparece en #sistema.
- [ ] Señales y operaciones visibles en Discord y en la app.
- [ ] Tabla de hiperparámetros, métricas y Diebold-Mariano visibles por activo.
- [ ] El bot sobrevive a un reinicio de la VM (systemd).
- [ ] Video de respaldo grabado.

## 13. Estado del proyecto (actualizado 16 de noviembre de 2026)

> Sección para quien retome el trabajo. Léela antes de tocar nada. No contiene claves: todas están en `.env` (local y en la VM).

### Qué está hecho y verificado
- Todo el código de las secciones 1–10 está implementado y probado: `pytest` (23 pruebas) y `scripts/check_apis.py` (16/16 conexiones) pasan desde la Mac y desde la VM.
- **Repositorio**: `https://github.com/ampecomm-tech/Copiloto-opciones` (privado, rama `main`). En la Mac, `gh` y `gcloud` están instalados y autenticados.
- **VM de producción**: `copiloto-mx`, proyecto GCP `copiloto-opciones`, zona `northamerica-south1-a` (Querétaro), e2-medium, IP pública `34.51.52.155`, app en `http://34.51.52.155:8501`. Código en `/opt/copiloto-opciones`, servicios `copiloto-bot` y `copiloto-app` (sobreviven al reinicio, verificado), logs en `/opt/copiloto-opciones/logs/`. Firewall `copiloto-8501`. Alerta de presupuesto de 15 USD.
  - Entrar: `gcloud compute ssh copiloto-mx --zone=northamerica-south1-a`.
  - Actualizar código: hacer push a `main`; en la VM `cd /opt/copiloto-opciones && sudo git pull && sudo systemctl restart copiloto-bot copiloto-app` (el clon inicial se hizo con un git bundle; si `git pull` pide credenciales, usar un token de GitHub o subir un bundle nuevo con `gcloud compute scp`).
  - Subir un `.env` nuevo: `gcloud compute scp .env copiloto-mx:~/ --zone=northamerica-south1-a` y luego `sudo cp ~/.env /opt/copiloto-opciones/.env && sudo systemctl restart copiloto-bot copiloto-app`.
- **Entorno local (Mac)**: `.venv/` creado con el Python 3.12 de conda (`~/anaconda3/envs/bmv/bin/python`); el Python del sistema es 3.9 y no sirve. `requirements.txt` fija `cryptography<45` porque la versión nueva no carga en esta Mac. Arrancar: `.venv/bin/streamlit run app/main.py --server.port 8501`.
- Ejecución real verificada: órdenes de opciones BTC en el testnet de Deribit (long call ejecutada, spread parcial) y orden multipata de NVDA en Alpaca Paper (en cola fuera de horario).

### Decisiones y limitaciones conocidas (no son bugs)
- **Binance bloquea IPs de EE. UU. (error 451)**: por eso la VM está en Querétaro y no en us-central1. No mover la VM a una región estadounidense.
- **Together AI**: solo 22 de 178 modelos están disponibles en serverless para esta cuenta (los Llama clásicos, Qwen y Mistral piden endpoint dedicado). Los asignados en `config.yaml` (`openai/gpt-oss-120b`, `deepseek-ai/DeepSeek-V4.1-Flash`, `meta-models/Muse-Glimmer-30B`) son modelos de razonamiento: necesitan `max_tokens` ≥ 4000 o devuelven contenido vacío. Para cambiar modelos, sondear primero con una llamada mínima; la API no marca cuáles son serverless.
- **Gemini**: funciona en modo normal y JSON; la búsqueda de Google (grounding) devuelve 429 por cuota del plan. El sentimiento usa los titulares descargados y lo muestra como "Gemini + titulares (sin Google Search)". Si se activa facturación en AI Studio, el grounding se reactiva solo (se reintenta cada hora).
- **xAI Grok**: desde el 19 nov 2026 hay `XAI_API_KEY` en la Mac y en la VM, así que el sentimiento usa "X vía Grok" (`x_search`, unos 20 s por activo, caché de 15 min) y Gemini queda de respaldo. El modelo configurado, `grok-4-fast`, responde aunque no aparece en `/v1/models`; si deja de funcionar, usar uno de esa lista. Together AI no ofrece modelos de xAI.
- **Testnet de opciones de Binance**: `demo-fapi.binance.com` responde 403 en `/eapi`; por eso la ejecución cripto va a Deribit testnet (`config.ejecucion.venue_cripto`). Deribit testnet solo tiene BTC y ETH; otros subyacentes caen a paper con aviso en `#sistema`.
- La API key de Deribit tiene scope `trade:read_write`; desde la Mac no lee saldo (falta `account:read`), desde la VM sí.
- **Diseño (16 nov)**: tema oscuro con un solo acento azul en `.streamlit/config.toml`; los mismos colores están en `app/ui.py` (`TOKENS`), junto con el CSS global, los componentes (pastillas, franja de datos, filas del feed, tabla de patas) y `estilo_figura()` para Plotly. Sin emojis: íconos `:material/...:`. Los cambios en `app/pages/*.py` y en `config.toml` no se recargan en caliente (las páginas se importan con `importlib`): reiniciar Streamlit.
- **Rutas de Streamlit**: los `url_path` de `st.navigation` en `app/main.py` no deben coincidir con los nombres de archivo de `app/pages/` (p. ej. `copiloto`), porque una URL directa ejecutaría la página automática sin el resto de la app.
- El bot en la VM opera en Binance spot testnet y Alpaca Paper (`bot.forzar_paper: false`). Para dejarlo solo en paper interno, poner `forzar_paper: true` en `bot` y en `ejecucion` de `config.yaml`.

### Entregables (19 de noviembre de 2026)
- Informe técnico: `docs/Informe_Tecnico_Copiloto_Opciones.pdf`, generado con `scripts/generar_informe.py` a partir de `docs/resultados/` (exportado con `scripts/exportar_resultados.py` sobre una copia de la base de la VM). Los integrantes de la portada se editan en `docs/portada.yaml`.
- Notebook ejecutado: `notebooks/validacion_cuantitativa.ipynb`, generado con `scripts/crear_notebook.py`.
- Las dependencias de estos scripts están en `requirements-docs.txt`; la app y el bot no las necesitan.
- La presentación ejecutiva es una presentación de claude.ai del usuario, no está en el repositorio.

### Pendiente antes de la demo (jueves 19)
1. Grabar el video de respaldo y ensayar con el Centro de control abierto.
2. La red del campus (FortiGuard) bloquea la IP de la app por no tener categoría: presentar con hotspot del celular y pedir a la clase que entre con datos móviles.
