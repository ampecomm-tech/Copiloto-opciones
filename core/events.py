"""Bus de eventos: la única vía para registrar lo que pasa en el sistema.

`emit()` guarda el evento en la tabla `events`, lo envía al webhook de Discord
del canal y lo escribe en el log. Discord nunca puede tumbar la app ni el bot:
cualquier fallo de red se traga y el evento igual queda en la base de datos.
"""
from __future__ import annotations

import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any

import requests

from core import db
from core.config import ruta_logs, webhook_discord

CANALES: tuple[str, ...] = ("mesa", "sentimiento", "bot_senales", "bot_operaciones", "alertas", "sistema")
"""Canales válidos del bus de eventos."""

TIPOS_COLOR: dict[str, int] = {
    "ok": 0x2ECC71,        # verde
    "verde": 0x2ECC71,
    "entrada": 0x2ECC71,
    "warning": 0xF1C40F,   # amarillo
    "amarillo": 0xF1C40F,
    "alerta": 0xF1C40F,
    "error": 0xE74C3C,     # rojo
    "rojo": 0xE74C3C,
    "salida": 0xE74C3C,
    "info": 0x95A5A6,      # gris
    "senal": 0x3498DB,     # azul
    "agente": 0x9B59B6,    # morado
}
"""Color del embed de Discord según el tipo de evento."""

_LIMITE_CAMPO = 1024
_LIMITE_TITULO = 256
_LIMITE_CAMPOS = 25


def _logger(nombre: str) -> logging.Logger:
    """Devuelve un logger con archivo rotativo en `logs/<nombre>.log`."""
    log = logging.getLogger(nombre)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = RotatingFileHandler(ruta_logs() / f"{nombre}.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


log_app = _logger("app")
log_bot = _logger("trading_bot")


def _truncar(texto: Any, limite: int) -> str:
    """Convierte a texto y recorta a `limite` caracteres añadiendo '…' si hace falta."""
    s = str(texto)
    return s if len(s) <= limite else s[: limite - 1] + "…"


def _formatear_valor(valor: Any) -> str:
    """Representación legible de un valor del payload para un campo de Discord."""
    if isinstance(valor, float):
        return f"{valor:,.4f}".rstrip("0").rstrip(".")
    if isinstance(valor, (dict, list)):
        return "```json\n" + _truncar(db.a_json(valor), _LIMITE_CAMPO - 12) + "\n```"
    return _truncar(valor, _LIMITE_CAMPO)


def construir_embed(event_type: str, title: str, payload: dict[str, Any], channel: str) -> dict[str, Any]:
    """Construye el embed de Discord respetando los límites de la API.

    Args:
        event_type: Tipo del evento (define el color).
        title: Título del embed.
        payload: Campos a mostrar.
        channel: Canal (se muestra en el pie).
    """
    campos = []
    for k, v in list(payload.items())[:_LIMITE_CAMPOS]:
        campos.append({"name": _truncar(k, 256), "value": _formatear_valor(v) or "—", "inline": not isinstance(v, (dict, list))})
    return {
        "title": _truncar(title, _LIMITE_TITULO),
        "color": TIPOS_COLOR.get(event_type, TIPOS_COLOR["info"]),
        "fields": campos,
        "footer": {"text": f"canal: {channel} · tipo: {event_type}"},
        "timestamp": db.ahora_iso(),
    }


def enviar_discord(url: str, embed: dict[str, Any], reintentos: int = 3) -> bool:
    """Envía un embed al webhook con reintento y backoff ante 429 o errores de red.

    Returns:
        True si Discord aceptó el mensaje, False en cualquier otro caso.
    """
    if not url:
        return False
    espera = 1.0
    for _ in range(reintentos):
        try:
            r = requests.post(url, json={"embeds": [embed]}, timeout=8)
            if r.status_code in (200, 204):
                return True
            if r.status_code == 429:
                try:
                    espera = float(r.json().get("retry_after", espera))
                except Exception:
                    pass
                time.sleep(min(espera, 10))
                espera *= 2
                continue
            log_app.warning("Discord respondió %s: %s", r.status_code, r.text[:200])
            return False
        except requests.RequestException as e:
            log_app.warning("Error de red con Discord: %s", e)
            time.sleep(espera)
            espera *= 2
    return False


def emit(channel: str, event_type: str, title: str, payload: dict[str, Any], user_id: str | None = None) -> int:
    """Guarda el evento en la tabla `events`, lo manda al webhook de Discord del canal y lo escribe en el log.

    Args:
        channel: Uno de `CANALES`.
        event_type: Tipo (`ok`, `warning`, `error`, `info`, `senal`, `agente`, …).
        title: Título corto.
        payload: Diccionario serializable con los detalles.
        user_id: Usuario asociado, si aplica.

    Returns:
        Id del evento insertado (0 si falló la inserción, lo cual se registra en log).
    """
    if channel not in CANALES:
        raise ValueError(f"Canal inválido: {channel}. Válidos: {CANALES}")
    payload = payload or {}
    event_id = 0
    try:
        event_id = db.ejecutar(
            "INSERT INTO events(ts, channel, event_type, title, payload_json, user_id) VALUES (?,?,?,?,?,?)",
            (db.ahora_iso(), channel, event_type, title, db.a_json(payload), user_id),
        )
    except Exception as e:  # la DB puede fallar por bloqueo; se registra pero no se propaga
        log_app.error("No se pudo guardar el evento '%s' en DB: %s", title, e)

    log = log_bot if channel.startswith("bot") else log_app
    log.info("[%s/%s] %s | %s", channel, event_type, title, db.a_json(payload)[:500])

    url = webhook_discord(channel)
    if url:
        embed = construir_embed(event_type, title, payload, channel)
        # Envío en hilo aparte para no bloquear la UI ni el loop del bot.
        threading.Thread(target=enviar_discord, args=(url, embed), daemon=True).start()
    return event_id


def leer_eventos(
    canal: str | None = None,
    activo: str | None = None,
    desde: str | None = None,
    limite: int = 200,
) -> list[dict[str, Any]]:
    """Lee eventos de la DB con filtros opcionales, del más reciente al más antiguo.

    Args:
        canal: Filtra por canal exacto.
        activo: Filtra por texto (en título o payload).
        desde: ISO timestamp mínimo.
        limite: Máximo de filas.
    """
    sql = "SELECT * FROM events WHERE 1=1"
    params: list[Any] = []
    if canal:
        sql += " AND channel = ?"
        params.append(canal)
    if activo:
        sql += " AND (title LIKE ? OR payload_json LIKE ?)"
        params += [f"%{activo}%", f"%{activo}%"]
    if desde:
        sql += " AND ts >= ?"
        params.append(desde)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limite))
    filas = db.consultar(sql, tuple(params))
    for f in filas:
        f["payload"] = db.de_json(f.pop("payload_json"), {})
    return filas
