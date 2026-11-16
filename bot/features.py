"""Indicadores técnicos y matriz de features para el bot.

Todas las funciones reciben un DataFrame OHLCV (columnas open/high/low/close/volume)
y devuelven Series o DataFrames alineados con el índice original.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(serie: pd.Series, periodo: int) -> pd.Series:
    """Media móvil exponencial."""
    return serie.ewm(span=periodo, adjust=False).mean()


def rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    """Índice de fuerza relativa (Wilder)."""
    delta = serie.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / periodo, adjust=False).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / periodo, adjust=False).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(serie: pd.Series, rapida: int = 12, lenta: int = 26, senal: int = 9) -> pd.DataFrame:
    """MACD, línea de señal e histograma."""
    linea = ema(serie, rapida) - ema(serie, lenta)
    sig = ema(linea, senal)
    return pd.DataFrame({"macd": linea, "macd_signal": sig, "macd_hist": linea - sig})


def atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / periodo, adjust=False).mean()


def bollinger(serie: pd.Series, periodo: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger y posición relativa del precio dentro de ellas (%B)."""
    media = serie.rolling(periodo).mean()
    std = serie.rolling(periodo).std()
    sup, inf = media + k * std, media - k * std
    pct_b = (serie - inf) / (sup - inf).replace(0, np.nan)
    return pd.DataFrame({"bb_media": media, "bb_sup": sup, "bb_inf": inf, "bb_pct": pct_b, "bb_ancho": (sup - inf) / media})


def chandelier_exit(df: pd.DataFrame, periodo: int = 22, mult: float = 3.0) -> pd.DataFrame:
    """Chandelier Exit: stop dinámico largo = máx(periodo) − mult·ATR; corto = mín + mult·ATR."""
    a = atr(df, periodo)
    return pd.DataFrame({"chandelier_long": df["high"].rolling(periodo).max() - mult * a, "chandelier_short": df["low"].rolling(periodo).min() + mult * a})


def construir_features(df: pd.DataFrame, sentimiento: float | None = None) -> pd.DataFrame:
    """Matriz de features del RandomForest.

    Incluye EMA (relativas al precio), RSI, MACD, ATR relativo, Bollinger, retornos
    rezagados, volumen relativo y el último puntaje de sentimiento (constante) si existe.
    """
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    for p in (9, 21, 50, 200):
        f[f"ema{p}_rel"] = c / ema(c, p) - 1
    f["rsi14"] = rsi(c, 14)
    f["rsi7"] = rsi(c, 7)
    m = macd(c)
    f["macd_rel"] = m["macd"] / c
    f["macd_hist_rel"] = m["macd_hist"] / c
    f["atr_rel"] = atr(df, 14) / c
    b = bollinger(c)
    f["bb_pct"] = b["bb_pct"]
    f["bb_ancho"] = b["bb_ancho"]
    for k in (1, 2, 3, 5, 10):
        f[f"ret_{k}"] = c.pct_change(k)
    f["vol_rel"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    f["rango_rel"] = (df["high"] - df["low"]) / c
    f["volatilidad_20"] = c.pct_change().rolling(20).std()
    f["sentimiento"] = float(sentimiento) if sentimiento is not None else 0.0
    return f.replace([np.inf, -np.inf], np.nan)


def etiqueta_objetivo(df: pd.DataFrame, horizonte: int = 4) -> pd.Series:
    """Etiqueta binaria: 1 si el cierre sube en `horizonte` velas, 0 si no."""
    futuro = df["close"].shift(-horizonte)
    return (futuro > df["close"]).astype(int).where(futuro.notna())


def retorno_futuro(df: pd.DataFrame, horizonte: int = 4) -> pd.Series:
    """Retorno porcentual a `horizonte` velas (objetivo de regresión para métricas de error)."""
    return df["close"].shift(-horizonte) / df["close"] - 1
