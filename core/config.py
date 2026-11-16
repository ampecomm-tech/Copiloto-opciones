"""Carga de configuración: variables de entorno (.env) y parámetros (config.yaml).

Toda la configuración del sistema pasa por este módulo. Ningún otro módulo
lee `os.environ` ni abre `config.yaml` directamente.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

RAIZ: Path = Path(__file__).resolve().parent.parent
"""Carpeta raíz del repositorio."""

load_dotenv(RAIZ / ".env")


@lru_cache(maxsize=1)
def cargar_config() -> dict[str, Any]:
    """Lee `config.yaml` una sola vez y devuelve el diccionario de parámetros.

    Returns:
        Diccionario con todas las secciones del archivo de configuración.
    """
    ruta = RAIZ / "config.yaml"
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg(*claves: str, default: Any = None) -> Any:
    """Accede a un valor anidado de `config.yaml` por ruta de claves.

    Ejemplo: `cfg("bot", "riesgo", "sl_atr_mult")`.

    Args:
        *claves: Secuencia de claves anidadas.
        default: Valor devuelto si la ruta no existe.

    Returns:
        El valor configurado o `default`.
    """
    nodo: Any = cargar_config()
    for k in claves:
        if not isinstance(nodo, dict) or k not in nodo:
            return default
        nodo = nodo[k]
    return nodo


def env(nombre: str, default: str = "") -> str:
    """Devuelve una variable de entorno (cargada desde `.env`) o un valor por defecto.

    Args:
        nombre: Nombre de la variable.
        default: Valor si no existe o está vacía.
    """
    valor = os.environ.get(nombre, "")
    return valor if valor else default


def ruta_db() -> Path:
    """Ruta absoluta del archivo SQLite compartido por app y bot."""
    ruta = Path(env("DB_PATH", "data/app.db"))
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return ruta


def ruta_logs() -> Path:
    """Carpeta de logs; se crea si no existe."""
    ruta = RAIZ / "logs"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def webhook_discord(canal: str) -> str:
    """Devuelve la URL del webhook de Discord asociada a un canal del bus de eventos.

    Args:
        canal: Uno de `mesa`, `sentimiento`, `bot_senales`, `bot_operaciones`, `alertas`, `sistema`.

    Returns:
        URL del webhook o cadena vacía si no está configurado.
    """
    return env(f"DISCORD_WEBHOOK_{canal.upper()}")
