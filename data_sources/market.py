"""Velas OHLCV y datos de mercado unificados (Binance mainnet, Alpaca, yfinance).

`velas(symbol, venue, timeframe, limite)` es la función que usan el bot y el
copiloto. Las señales del bot se calculan **siempre con datos públicos del
Binance real** (mainnet, sin llave) para cripto; el testnet solo ejecuta.
`cadena_para(activo)` elige Binance Options (cripto), Alpaca (acciones) o
yfinance (respaldo) y completa la IV con Black-Scholes donde falte.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from core.config import cfg
from core.events import log_app
from quant.black_scholes import anios, volatilidad_implicita

CRIPTO_BINANCE_OPCIONES = {"BTC", "ETH", "BNB", "SOL", "XRP", "DOGE"}
"""Subyacentes con opciones europeas en Binance."""
_CACHE: dict[str, tuple[float, Any]] = {}

MAPA_TF_YF = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "4h": "60m", "1d": "1d"}
ALIAS = {"BITCOIN": "BTC", "ETHEREUM": "ETH", "ETHER": "ETH", "SOLANA": "SOL", "RIPPLE": "XRP", "DOGECOIN": "DOGE", "BTC-USD": "BTC", "ETH-USD": "ETH", "SOL-USD": "SOL"}


def es_cripto(activo: str) -> bool:
    """True si el activo es cripto (con opciones en Binance, o `XXX-USD` / `XXX/USDT`)."""
    a = activo.upper()
    return a in CRIPTO_BINANCE_OPCIONES or a.endswith("-USD") or "/USDT" in a


def normalizar_activo(texto: str) -> str:
    """Normaliza entradas del usuario: 'btc', 'bitcoin', 'nvda' → 'BTC', 'NVDA'."""
    t = texto.strip().upper()
    return ALIAS.get(t, t)


def velas(symbol: str, venue: str = "paper", timeframe: str | None = None, limite: int | None = None) -> pd.DataFrame:
    """Velas OHLCV con índice temporal UTC y columnas open/high/low/close/volume.

    Args:
        symbol: `SOL/USDT` (cripto), `NVDA` o `GC=F` (yfinance).
        venue: `binance_spot_testnet`, `alpaca_paper` o `paper`. Para cripto los datos
            salen del Binance mainnet aunque la ejecución sea en testnet.
        timeframe: `15m`, `1h`, `1d`… (por defecto `config.bot.timeframe`).
        limite: Número de velas (por defecto `config.bot.velas_historico`).
    """
    timeframe = timeframe or cfg("bot", "timeframe", default="15m")
    limite = int(limite or cfg("bot", "velas_historico", default=1500))
    clave = f"{symbol}|{timeframe}|{limite}"
    ahora = time.time()
    ttl = 30 if timeframe.endswith("m") else 300   # velas de horas o días cambian poco: 5 min de caché
    if clave in _CACHE and ahora - _CACHE[clave][0] < ttl:
        return _CACHE[clave][1].copy()
    try:
        df = _velas_binance(symbol, timeframe, limite) if "/USDT" in symbol else _velas_yf(symbol, timeframe, limite)
    except Exception as e:
        log_app.error("No se pudieron bajar velas de %s (%s): %s", symbol, venue, e)
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    _CACHE[clave] = (ahora, df)
    return df.copy()


def _velas_binance(symbol: str, timeframe: str, limite: int) -> pd.DataFrame:
    """Velas del Binance mainnet (público, sin llave) vía ccxt, paginando hacia atrás."""
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    ms_tf = ex.parse_timeframe(timeframe) * 1000
    desde = ex.milliseconds() - limite * ms_tf
    todas: list[list[float]] = []
    while len(todas) < limite:
        lote = ex.fetch_ohlcv(symbol, timeframe, since=desde, limit=min(1000, limite - len(todas)))
        if not lote:
            break
        todas += lote
        desde = lote[-1][0] + ms_tf
        if len(lote) < 2:
            break
    df = pd.DataFrame(todas, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").astype(float)


def _velas_yf(symbol: str, timeframe: str, limite: int) -> pd.DataFrame:
    """Velas desde yfinance; intradía limitado a 60 días por Yahoo."""
    import yfinance as yf

    intervalo = MAPA_TF_YF.get(timeframe, "15m")
    periodo = "60d" if intervalo.endswith("m") else "2y"
    h = yf.Ticker(symbol).history(period=periodo, interval=intervalo, auto_adjust=True)
    if h.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    h.index = pd.to_datetime(h.index, utc=True)
    if timeframe == "4h":
        h = h.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return h.tail(limite)


def completar_iv(cadena: pd.DataFrame, S: float, r: float) -> pd.DataFrame:
    """Rellena la columna `iv` con Black-Scholes (Brent) donde falte, usando el precio mid."""
    if cadena.empty:
        return cadena
    c = cadena.copy()
    faltan = c["iv"].isna() if "iv" in c else pd.Series(True, index=c.index)
    for i in c[faltan].index:
        f = c.loc[i]
        c.at[i, "iv"] = volatilidad_implicita(float(f["mid"]), S, float(f["strike"]), anios(float(f["dte"])), r, str(f["type"]))
    return c


def cadena_para(activo: str, r: float | None = None, horizonte_dias: int = 30) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Elige la fuente, descarga la cadena y completa la IV.

    - BTC/ETH/BNB/SOL/XRP/DOGE → Binance European Options.
    - Acciones/ETFs → Alpaca (si hay credenciales); si falla o no hay, yfinance.

    Returns:
        (cadena, spot, meta) con `meta = {fuente, multiplicador, paso_qty, periodos_por_anio, venue_ejecucion}`.
    """
    from data_sources import alpaca, binance_options, yf_options

    r = cfg("app", "tasa_libre_riesgo", default=0.045) if r is None else r
    a = normalizar_activo(activo)
    hasta = max(int(horizonte_dias * 2), 21)
    if a in CRIPTO_BINANCE_OPCIONES:
        cadena, S, m = binance_options.cadena_opciones(a)
        meta = {"fuente": "binance_options", "multiplicador": float(m.get("unit", 1.0)), "paso_qty": float(m.get("paso_qty", 0.01)),
                "periodos_por_anio": 365, "simbolo_velas": f"{a}/USDT", "venue_ejecucion": str(cfg("ejecucion", "venue_cripto", default="deribit_testnet"))}
    else:
        cadena, S = alpaca.cadena_opciones(a, hasta) if alpaca.disponible() else (pd.DataFrame(), float("nan"))
        fuente = "alpaca"
        if cadena.empty:
            cadena, S = yf_options.cadena_opciones(a, hasta_dias=hasta)
            fuente = "yfinance"
        meta = {"fuente": fuente, "multiplicador": 100.0, "paso_qty": 1.0, "periodos_por_anio": 252, "simbolo_velas": a, "venue_ejecucion": "alpaca_paper"}
    if not cadena.empty and S == S:
        cadena = completar_iv(cadena, S, r)
    return cadena, S, meta


def cierres_diarios(activo: str, dias: int = 120) -> pd.Series:
    """Cierres diarios del subyacente para la volatilidad histórica."""
    a = normalizar_activo(activo)
    if a in CRIPTO_BINANCE_OPCIONES:
        from data_sources.binance_options import velas as velas_bin

        df = velas_bin(a, dias)
        if not df.empty:
            return df["close"]
    simbolo = f"{a}-USD" if a in CRIPTO_BINANCE_OPCIONES else a
    df = velas(simbolo, "paper", "1d", dias)
    return df["close"] if not df.empty else pd.Series(dtype=float)
