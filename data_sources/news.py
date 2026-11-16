"""Noticias: yfinance `Ticker.news` para acciones y RSS (CoinDesk, Cointelegraph) para cripto.

Cada titular recibe una etiqueta de sentimiento (positivo/negativo/neutral) generada
por **Gemini** (con un léxico simple como respaldo); el sentimiento de X lo da Grok
(ver `agents/grok_sentiment.py`). También expone el resumen del día (Gemini),
Fear & Greed (alternative.me) para cripto y los próximos eventos (earnings +
eventos macro con búsqueda en Google vía Gemini + manuales de `config.yaml`).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests
from pydantic import BaseModel

from core.config import cfg
from core.events import log_app
from data_sources.market import es_cripto, normalizar_activo

RSS_CRIPTO = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
}
_CACHE: dict[str, tuple[float, Any]] = {}

POSITIVAS = {"sube", "rally", "récord", "record", "alza", "gana", "bull", "surge", "soar", "jump", "gain", "high", "beat", "upgrade", "growth", "approve", "adopt", "etf inflow", "breakout", "rebound", "optimis"}
NEGATIVAS = {"cae", "baja", "pierde", "crash", "bear", "drop", "fall", "plunge", "slump", "loss", "low", "miss", "downgrade", "lawsuit", "hack", "ban", "fear", "selloff", "sell-off", "liquidat", "decline", "warn"}


def etiqueta_sentimiento(texto: str) -> str:
    """Etiqueta rápida por léxico: `positivo`, `negativo` o `neutral`."""
    t = texto.lower()
    pos = sum(1 for w in POSITIVAS if w in t)
    neg = sum(1 for w in NEGATIVAS if w in t)
    if pos > neg:
        return "positivo"
    if neg > pos:
        return "negativo"
    return "neutral"


def _cache(clave: str, fn, ttl: float = 600):
    """Caché simple en memoria."""
    ahora = time.time()
    if clave in _CACHE and ahora - _CACHE[clave][0] < ttl:
        return _CACHE[clave][1]
    val = fn()
    _CACHE[clave] = (ahora, val)
    return val


def noticias_yf(ticker: str, maximo: int = 12) -> list[dict[str, Any]]:
    """Titulares de yfinance para un ticker."""
    import yfinance as yf

    def _f() -> list[dict[str, Any]]:
        out = []
        try:
            for n in (yf.Ticker(ticker).news or [])[:maximo]:
                c = n.get("content", n)
                titulo = c.get("title") or n.get("title") or ""
                url = (c.get("canonicalUrl") or {}).get("url") or n.get("link") or ""
                fuente = (c.get("provider") or {}).get("displayName") or n.get("publisher") or "Yahoo"
                fecha = c.get("pubDate") or n.get("providerPublishTime") or ""
                if isinstance(fecha, (int, float)):
                    fecha = datetime.fromtimestamp(fecha, tz=timezone.utc).isoformat()
                out.append({"titulo": titulo, "url": url, "fuente": fuente, "fecha": str(fecha)[:16], "sentimiento": etiqueta_sentimiento(titulo)})
        except Exception as e:
            log_app.warning("Noticias yfinance fallaron para %s: %s", ticker, e)
        return out

    return _cache(f"news:{ticker}", _f)


def noticias_rss(activo: str, maximo: int = 12) -> list[dict[str, Any]]:
    """Titulares de RSS cripto filtrados por el activo (BTC → bitcoin/btc)."""
    import feedparser

    claves = {"BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "eth", "ether"], "SOL": ["solana", "sol"]}
    palabras = claves.get(activo.upper(), [activo.lower()])

    def _f() -> list[dict[str, Any]]:
        out = []
        for fuente, url in RSS_CRIPTO.items():
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:40]:
                    texto = (e.get("title", "") + " " + e.get("summary", "")).lower()
                    if any(p in texto for p in palabras):
                        out.append({"titulo": e.get("title", ""), "url": e.get("link", ""), "fuente": fuente, "fecha": str(e.get("published", ""))[:16], "sentimiento": etiqueta_sentimiento(e.get("title", ""))})
            except Exception as ex:
                log_app.warning("RSS %s falló: %s", fuente, ex)
        return out[:maximo]

    return _cache(f"rss:{activo}", _f)


def noticias(activo: str, maximo: int = 12) -> list[dict[str, Any]]:
    """Titulares del activo según su tipo (RSS para cripto, yfinance para el resto), etiquetados con Gemini."""
    a = normalizar_activo(activo)
    if es_cripto(a):
        lista = noticias_rss(a.replace("-USD", ""), maximo)
        if not lista:
            lista = noticias_yf(f"{a}-USD" if "-" not in a else a, maximo)
    else:
        lista = noticias_yf(a, maximo)
    return etiquetar_con_gemini(a, lista)


class _Etiquetas(BaseModel):
    """Salida de Gemini para el etiquetado de titulares."""

    etiquetas: list[str]


def etiquetar_con_gemini(activo: str, titulares: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sustituye la etiqueta léxica por la de Gemini (una sola llamada por lote, cacheada 10 min)."""
    from agents import gemini_client as gem

    if not titulares or not gem.disponible():
        return titulares
    try:
        prompt = (f"Clasifica el sentimiento de cada titular para el precio de {activo} como inversionista. "
                  f"Devuelve JSON {{\"etiquetas\": [...]}} con exactamente {len(titulares)} valores, cada uno 'positivo', 'negativo' o 'neutral', en el mismo orden.\n\n"
                  + "\n".join(f"{i + 1}. {t['titulo']}" for i, t in enumerate(titulares)))
        r = gem.generar_json(prompt, _Etiquetas, ttl_cache_s=600)
        if len(r.etiquetas) == len(titulares):
            for t, e in zip(titulares, r.etiquetas):
                if e in ("positivo", "negativo", "neutral"):
                    t["sentimiento"] = e
                    t["fuente_etiqueta"] = "gemini"
    except Exception as e:
        log_app.warning("Etiquetado con Gemini falló: %s", e)
    return titulares


def fear_and_greed() -> dict[str, Any] | None:
    """Índice Fear & Greed de cripto (alternative.me)."""
    def _f() -> dict[str, Any] | None:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
            d = r.json()["data"][0]
            return {"valor": int(d["value"]), "clasificacion": d["value_classification"]}
        except Exception as e:
            log_app.warning("Fear & Greed no disponible: %s", e)
            return None

    return _cache("fng", _f, 1800)


class _Eventos(BaseModel):
    """Salida de Gemini para eventos macro."""

    eventos: list[dict[str, str]]


def eventos_macro_gemini(activo: str) -> list[dict[str, str]]:
    """Próximos eventos macro relevantes para el activo, con búsqueda en Google (grounding). Cache 6 h."""
    from agents import gemini_client as gem

    if not gem.grounding_disponible():
        return []
    try:
        hoy = datetime.now(timezone.utc).date().isoformat()
        r = gem.generar_json(
            f"Hoy es {hoy}. Busca los próximos eventos (próximas 4 semanas) que puedan mover el precio de {activo}: decisiones de la Fed, CPI, empleo, "
            f"resultados, vencimientos relevantes, eventos regulatorios o del propio activo. Devuelve JSON "
            f'{{"eventos": [{{"fecha": "YYYY-MM-DD", "descripcion": "texto corto en español"}}]}} con máximo 6 eventos, solo fechas futuras.',
            _Eventos, ttl_cache_s=6 * 3600, grounding=True)
        return [e for e in r.eventos if str(e.get("fecha", "")) >= hoy][:6]
    except Exception as e:
        log_app.warning("Eventos macro con Gemini fallaron: %s", e)
        return []


def eventos_proximos(activo: str) -> list[dict[str, str]]:
    """Próximos eventos: earnings (yfinance), eventos macro con Gemini + Google y los manuales de `config.yaml`."""
    eventos: list[dict[str, str]] = eventos_macro_gemini(normalizar_activo(activo))
    a = normalizar_activo(activo)
    if not es_cripto(a):
        from data_sources.yf_options import fecha_earnings

        f = fecha_earnings(a)
        if f:
            eventos.append({"fecha": f[:10], "descripcion": f"Resultados trimestrales de {a}"})
    hoy = datetime.now(timezone.utc).date().isoformat()
    for ev in cfg("eventos_macro", default=[]) or []:
        if str(ev.get("fecha", "")) >= hoy:
            eventos.append({"fecha": str(ev["fecha"]), "descripcion": str(ev["descripcion"])})
    return sorted(eventos, key=lambda e: e["fecha"])


def resumen_dia(activo: str, titulares: list[dict[str, Any]]) -> str:
    """Resumen del día con Gemini (2-3 frases); si no está disponible, conteo por sentimiento."""
    if not titulares:
        return f"Sin titulares recientes para {activo}."
    pos = sum(1 for t in titulares if t["sentimiento"] == "positivo")
    neg = sum(1 for t in titulares if t["sentimiento"] == "negativo")
    tono = "positivo" if pos > neg else "negativo" if neg > pos else "mixto"
    base = f"{len(titulares)} titulares para {activo}: {pos} positivos, {neg} negativos, tono general {tono}."
    from agents import gemini_client as gem

    if not gem.disponible():
        return base
    try:
        texto = gem.generar(
            f"Resume en 2-3 frases, en español y sin inventar datos, qué está pasando hoy con {activo} según estos titulares:\n"
            + "\n".join(f"- {t['titulo']} ({t['sentimiento']})" for t in titulares[:12]),
            system="Eres un analista financiero conciso.", ttl_cache_s=600, max_tokens=300)
        return texto.strip() or base
    except Exception as e:
        log_app.warning("Resumen del día con Gemini falló: %s", e)
        return base
