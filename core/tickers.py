"""Búsqueda de activos por nombre: "Solana", "Apple", "oro" → ticker, mercado, venue y precio.

Combina el buscador de Yahoo Finance (acciones, ETFs, materias primas) con la
lista de mercados de Binance vía ccxt. El buscador asigna el venue del bot
automáticamente:

- cripto listada en el Binance spot testnet → `binance_spot_testnet`;
- acciones y ETFs de EE. UU. → `alpaca_paper` (si hay credenciales de Alpaca);
- todo lo demás (materias primas, cripto no listada, mercados fuera de EE. UU.) → `paper`.

Cada fuente está protegida con try/except: si una falla, se devuelven las coincidencias de la otra.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from core.config import cfg
from core.events import log_app

ALIAS_CRIPTO: dict[str, str] = {
    "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL", "cardano": "ADA", "ripple": "XRP", "xrp": "XRP",
    "dogecoin": "DOGE", "doge": "DOGE", "bnb": "BNB", "binance coin": "BNB",
    "polkadot": "DOT", "avalanche": "AVAX", "chainlink": "LINK", "litecoin": "LTC",
    "polygon": "MATIC", "toncoin": "TON", "tron": "TRX", "shiba": "SHIB",
}
"""Nombres comunes de criptomonedas y su símbolo base."""

ALIAS_COMMODITIES: dict[str, list[str]] = {
    "oro": ["GC=F", "GLD"], "gold": ["GC=F", "GLD"], "plata": ["SI=F", "SLV"], "silver": ["SI=F", "SLV"],
    "petroleo": ["CL=F", "USO"], "petróleo": ["CL=F", "USO"], "oil": ["CL=F", "USO"], "crudo": ["CL=F"],
    "gas natural": ["NG=F"], "cobre": ["HG=F"], "copper": ["HG=F"], "maíz": ["ZC=F"], "maiz": ["ZC=F"],
    "trigo": ["ZW=F"], "sp500": ["SPY", "^GSPC"], "s&p": ["SPY", "^GSPC"], "nasdaq": ["QQQ", "^IXIC"],
}
"""Nombres en español/inglés de materias primas e índices y sus tickers de Yahoo."""

BOLSAS_EEUU = {"NMS", "NYQ", "NGM", "PCX", "BTS", "ASE", "NCM", "NYS", "NAS", "PNK"}
"""Códigos de bolsa de Yahoo que corresponden a mercados de EE. UU. (operables en Alpaca)."""


@dataclass
class Coincidencia:
    """Resultado de búsqueda de un activo."""

    symbol: str
    display_name: str
    market_type: str   # "cripto" | "accion" | "etf" | "futuro" | "indice" | "otro"
    venue: str         # "binance_spot_testnet" | "alpaca_paper" | "paper"
    price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario plano."""
        return asdict(self)


def forzar_paper() -> bool:
    """True si `config.bot.forzar_paper` está activo (todo el bot en paper interno)."""
    return str(cfg("bot", "forzar_paper", default=False)).lower() == "true"


@lru_cache(maxsize=1)
def mercados_testnet() -> dict[str, dict[str, Any]]:
    """Carga (una vez) la lista de mercados spot del Binance testnet vía ccxt.

    Returns:
        Diccionario `{"SOL/USDT": {...}}`; vacío si el exchange no responde.
    """
    try:
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True})
        ex.set_sandbox_mode(True)
        mercados = ex.load_markets()
        return {s: m for s, m in mercados.items() if m.get("spot") and s.endswith("/USDT")}
    except Exception as e:
        log_app.warning("No se pudo cargar la lista de mercados del Binance testnet: %s", e)
        return {}


def buscar_cripto(consulta: str) -> list[Coincidencia]:
    """Busca criptomonedas por nombre o símbolo contra la lista de mercados del Binance testnet.

    Si el par existe en el testnet, el venue es `binance_spot_testnet` (los datos de señales
    salen del mainnet); si no, `paper` con precios de yfinance (`XXX-USD`).
    """
    q = consulta.strip().lower()
    base = ALIAS_CRIPTO.get(q, q.upper())
    resultados: list[Coincidencia] = []
    mercados = {} if forzar_paper() else mercados_testnet()
    candidatos = [s for s in mercados if s.split("/")[0] == base]
    if not candidatos:
        candidatos = [s for s in mercados if q.upper() in s.split("/")[0]][:5]
    for s in candidatos:
        resultados.append(Coincidencia(symbol=s, display_name=f"{s.split('/')[0]} (cripto, Binance spot testnet)", market_type="cripto", venue="binance_spot_testnet"))
    if q in ALIAS_CRIPTO or len(base) <= 5:
        resultados.append(Coincidencia(symbol=f"{base}-USD", display_name=f"{base} (cripto, paper)", market_type="cripto", venue="paper"))
    return resultados


def buscar_yahoo(consulta: str, maximo: int = 6) -> list[Coincidencia]:
    """Busca acciones, ETFs, futuros e índices con el buscador de Yahoo Finance y asigna venue."""
    from data_sources import alpaca

    alpaca_ok = alpaca.disponible() and not forzar_paper()
    resultados: list[Coincidencia] = []
    q = consulta.strip().lower()
    for alias, tickers in ALIAS_COMMODITIES.items():
        if q == alias:
            for t in tickers:
                es_etf = "=F" not in t and not t.startswith("^")
                resultados.append(Coincidencia(symbol=t, display_name=f"{alias.title()} ({t})", market_type="futuro" if "=F" in t else "indice" if t.startswith("^") else "etf",
                                               venue="alpaca_paper" if es_etf and alpaca_ok else "paper"))
            return resultados
    try:
        import yfinance as yf

        s = yf.Search(consulta, max_results=maximo)
        for q_ in s.quotes:
            tipo = str(q_.get("quoteType", "")).lower()
            mapa = {"equity": "accion", "etf": "etf", "future": "futuro", "index": "indice", "cryptocurrency": "cripto"}
            mt = mapa.get(tipo, "otro")
            eeuu = str(q_.get("exchange", "")) in BOLSAS_EEUU
            venue = "alpaca_paper" if mt in ("accion", "etf") and eeuu and alpaca_ok else "paper"
            resultados.append(Coincidencia(symbol=q_.get("symbol", ""), display_name=q_.get("shortname") or q_.get("longname") or q_.get("symbol", ""), market_type=mt, venue=venue))
    except Exception as e:
        log_app.warning("Búsqueda en Yahoo falló para '%s': %s", consulta, e)
    return resultados


def precio_actual(symbol: str, venue: str) -> float | None:
    """Último precio del activo (Binance mainnet para pares /USDT, yfinance para el resto)."""
    try:
        if "/USDT" in symbol:
            import ccxt

            return float(ccxt.binance({"enableRateLimit": True}).fetch_ticker(symbol)["last"])
        import yfinance as yf

        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        return float(hist["Close"].dropna().iloc[-1]) if len(hist) else None
    except Exception as e:
        log_app.warning("No se pudo obtener precio de %s: %s", symbol, e)
        return None


def buscar(consulta: str, con_precio: bool = True) -> list[dict[str, Any]]:
    """Búsqueda combinada por nombre: cripto (testnet/paper) + Yahoo Finance.

    Args:
        consulta: Texto libre ("Solana", "Apple", "oro").
        con_precio: Si True, consulta el precio de cada coincidencia (más lento).

    Returns:
        Lista de diccionarios con symbol, display_name, market_type, venue y price.
    """
    if not consulta.strip():
        return []
    vistos: set[str] = set()
    todos: list[Coincidencia] = []
    for c in buscar_cripto(consulta) + buscar_yahoo(consulta):
        if c.symbol and c.symbol not in vistos:
            vistos.add(c.symbol)
            todos.append(c)
    if con_precio:
        for c in todos[:8]:
            c.price = precio_actual(c.symbol, c.venue)
    return [c.to_dict() for c in todos[:8]]
