"""Conexión a SQLite, creación de tablas y helpers de consulta.

La base de datos es compartida por la app (Streamlit) y el bot (proceso
systemd), por lo que se usa modo WAL y `busy_timeout` para evitar bloqueos.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from core.config import ruta_db

ESQUEMA: str = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    channel TEXT NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    user_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id TEXT,
    asset TEXT NOT NULL,
    view_json TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    agents_json TEXT NOT NULL,
    verdict TEXT NOT NULL,
    chosen_strategy_json TEXT
);

CREATE TABLE IF NOT EXISTS option_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    recommendation_id INTEGER,
    asset TEXT NOT NULL,
    legs_json TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    entry_cost REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'abierta',
    exit_ts TEXT,
    exit_value REAL,
    pnl REAL,
    last_value REAL,
    greeks_json TEXT,
    last_update_ts TEXT,
    venue TEXT NOT NULL DEFAULT 'paper',
    venue_order_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS option_hedges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER NOT NULL,
    user_id TEXT,
    ts TEXT NOT NULL,
    venue TEXT NOT NULL,
    instrumento TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    order_ids_json TEXT,
    status TEXT NOT NULL DEFAULT 'abierta',
    exit_ts TEXT,
    exit_price REAL,
    pnl REAL,
    detalle TEXT
);
CREATE INDEX IF NOT EXISTS idx_hedges_pos ON option_hedges(position_id, status);

CREATE TABLE IF NOT EXISTS sentiment_cache (
    asset TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    score REAL NOT NULL,
    summary TEXT NOT NULL,
    posts_json TEXT NOT NULL,
    provider TEXT
);

CREATE TABLE IF NOT EXISTS bot_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    market_type TEXT NOT NULL,
    venue TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    added_ts TEXT NOT NULL,
    training_status TEXT NOT NULL DEFAULT 'pendiente'
);

CREATE TABLE IF NOT EXISTS bot_strategy_state (
    asset_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    capital REAL NOT NULL,
    position_json TEXT,
    last_candle_ts TEXT,
    daily_pnl REAL NOT NULL DEFAULT 0,
    daily_date TEXT,
    PRIMARY KEY (asset_id, strategy)
);

CREATE TABLE IF NOT EXISTS bot_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    asset_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    details_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_asset ON bot_signals(asset_id, ts DESC);

CREATE TABLE IF NOT EXISTS bot_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    sl REAL,
    tp REAL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT,
    pnl REAL,
    venue_order_id TEXT,
    exit_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON bot_trades(asset_id, strategy);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    hyperparams_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    dm_results_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_cache (
    clave TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    respuesta TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bot_models (
    asset_id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    model_blob BLOB NOT NULL,
    features_json TEXT NOT NULL
);
"""


def ahora_iso() -> str:
    """Marca de tiempo actual en UTC, formato ISO 8601 con segundos."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def conectar() -> sqlite3.Connection:
    """Abre una conexión SQLite en modo WAL con filas tipo diccionario.

    Returns:
        Conexión lista para usar; el llamador debe cerrarla (ver `conexion()`).
    """
    con = sqlite3.connect(str(ruta_db()), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


@contextmanager
def conexion() -> Iterator[sqlite3.Connection]:
    """Context manager: abre conexión, hace commit al salir y rollback si hay error."""
    con = conectar()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


MIGRACIONES: tuple[tuple[str, str, str], ...] = (
    ("option_positions", "venue", "TEXT NOT NULL DEFAULT 'paper'"),
    ("option_positions", "venue_order_ids_json", "TEXT"),
    ("sentiment_cache", "provider", "TEXT"),
)
"""Columnas añadidas después del esquema inicial: (tabla, columna, definición)."""


def inicializar() -> None:
    """Crea todas las tablas si no existen y aplica migraciones de columnas. Idempotente."""
    with conexion() as con:
        con.executescript(ESQUEMA)
        for tabla, col, definicion in MIGRACIONES:
            existentes = {f["name"] for f in con.execute(f"PRAGMA table_info({tabla})").fetchall()}
            if col not in existentes:
                con.execute(f"ALTER TABLE {tabla} ADD COLUMN {col} {definicion}")


def ejecutar(sql: str, params: tuple | dict = ()) -> int:
    """Ejecuta una sentencia de escritura y devuelve el `lastrowid`.

    Args:
        sql: Sentencia SQL con marcadores `?` o `:nombre`.
        params: Parámetros de la sentencia.
    """
    with conexion() as con:
        cur = con.execute(sql, params)
        return int(cur.lastrowid or 0)


def consultar(sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
    """Ejecuta una consulta de lectura y devuelve una lista de diccionarios."""
    with conexion() as con:
        filas = con.execute(sql, params).fetchall()
        return [dict(f) for f in filas]


def consultar_uno(sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
    """Como `consultar`, pero devuelve solo la primera fila o `None`."""
    filas = consultar(sql, params)
    return filas[0] if filas else None


def a_json(obj: Any) -> str:
    """Serializa a JSON tolerando numpy, fechas y objetos no estándar."""
    return json.dumps(obj, ensure_ascii=False, default=_default_json)


def de_json(texto: str | None, default: Any = None) -> Any:
    """Deserializa JSON devolviendo `default` si el texto es vacío o inválido."""
    if not texto:
        return default
    try:
        return json.loads(texto)
    except (TypeError, ValueError):
        return default


def _default_json(obj: Any) -> Any:
    """Conversión de respaldo para `json.dumps` (numpy, datetime, pandas)."""
    try:
        import numpy as np  # importación local para no obligar numpy en módulos ligeros

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
