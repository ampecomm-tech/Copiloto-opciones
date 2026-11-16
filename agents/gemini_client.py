"""Cliente de Gemini (API de AI Studio, SDK `google-genai`; **no** Vertex AI).

Tareas frecuentes y de apoyo: etiquetas de noticias, opinión estructurada,
eventos con búsqueda en Google (grounding), explicaciones en lenguaje simple,
resumen diario y respaldo de la mesa. Incluye reintento con backoff ante
límites de peticiones y caché de respuestas repetidas en la tabla `llm_cache`.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agents.together_client import extraer_json
from core import db
from core.config import cfg, env
from core.events import log_app

T = TypeVar("T", bound=BaseModel)
_cliente: Any = None
_grounding_bloqueado_hasta: float = 0.0
"""Si la búsqueda de Google devuelve 'cuota excedida', se deja de intentar durante una hora."""


class GroundingNoDisponible(RuntimeError):
    """La búsqueda en Google (grounding) no está en la cuota de la cuenta."""


def grounding_disponible() -> bool:
    """True si no se ha detectado recientemente que el grounding está fuera de cuota."""
    return disponible() and time.time() >= _grounding_bloqueado_hasta


def disponible() -> bool:
    """True si hay `GEMINI_API_KEY`."""
    return bool(env("GEMINI_API_KEY"))


def modelo() -> str:
    """Modelo configurado (`GEMINI_MODEL`, por defecto gemini-3.1-flash-lite)."""
    return env("GEMINI_MODEL", "gemini-3.1-flash-lite")


def _get_cliente() -> Any:
    """Instancia perezosa del cliente `google.genai`."""
    global _cliente
    if _cliente is None:
        from google import genai

        _cliente = genai.Client(api_key=env("GEMINI_API_KEY"))
    return _cliente


def _clave_cache(*partes: Any) -> str:
    """Hash estable del prompt y los parámetros."""
    return hashlib.sha256(json.dumps(partes, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def _leer_cache(clave: str, ttl_s: float) -> str | None:
    """Respuesta cacheada si no ha vencido."""
    try:
        fila = db.consultar_uno("SELECT ts, respuesta FROM llm_cache WHERE clave = ?", (clave,))
    except Exception:
        return None
    if fila and time.time() - float(fila["ts"]) < ttl_s:
        return str(fila["respuesta"])
    return None


def _guardar_cache(clave: str, respuesta: str) -> None:
    """Guarda la respuesta (ignora errores de DB)."""
    try:
        db.ejecutar("INSERT INTO llm_cache(clave, ts, respuesta) VALUES (?,?,?) ON CONFLICT(clave) DO UPDATE SET ts=excluded.ts, respuesta=excluded.respuesta", (clave, time.time(), respuesta))
    except Exception as e:
        log_app.warning("No se pudo guardar en llm_cache: %s", e)


def generar(prompt: str, system: str | None = None, json_mode: bool = False, grounding: bool = False,
            temperatura: float | None = None, max_tokens: int | None = None, ttl_cache_s: float = 0, reintentos: int = 4) -> str:
    """Una llamada a Gemini con backoff exponencial ante 429/503 y caché opcional.

    Args:
        prompt: Mensaje del usuario.
        system: Instrucción de sistema.
        json_mode: Pide salida JSON (`response_mime_type=application/json`; incompatible con grounding).
        grounding: Activa la herramienta de búsqueda en Google.
        ttl_cache_s: Si > 0, reutiliza la respuesta cacheada durante esos segundos.

    Raises:
        RuntimeError si no hay API key o se agotan los reintentos.
    """
    global _grounding_bloqueado_hasta
    if not disponible():
        raise RuntimeError("GEMINI_API_KEY no configurada")
    if grounding and not grounding_disponible():
        raise GroundingNoDisponible("búsqueda de Google fuera de cuota (se reintenta en una hora)")
    clave = _clave_cache(modelo(), system, prompt, json_mode, grounding)
    if ttl_cache_s > 0:
        c = _leer_cache(clave, ttl_cache_s)
        if c is not None:
            return c
    from google.genai import types

    config: dict[str, Any] = {
        "temperature": float(cfg("gemini", "temperatura", default=0.3) if temperatura is None else temperatura),
        "max_output_tokens": int(max_tokens or cfg("gemini", "max_tokens", default=1200)),
    }
    if system:
        config["system_instruction"] = system
    if grounding:
        config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    elif json_mode:
        config["response_mime_type"] = "application/json"
    espera = 2.0
    ultimo: Exception | None = None
    for intento in range(reintentos):
        try:
            resp = _get_cliente().models.generate_content(model=modelo(), contents=prompt, config=types.GenerateContentConfig(**config))
            texto = resp.text or ""
            if ttl_cache_s > 0 and texto:
                _guardar_cache(clave, texto)
            return texto
        except Exception as e:
            ultimo = e
            msg = str(e)
            if grounding and ("RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()):
                # La cuota de grounding es estructural (plan sin búsqueda): no vale la pena reintentar.
                _grounding_bloqueado_hasta = time.time() + 3600
                log_app.warning("Grounding de Gemini fuera de cuota; se desactiva una hora")
                raise GroundingNoDisponible(msg[:200]) from e
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500")):
                log_app.warning("Gemini limitado (intento %d): %s", intento + 1, msg[:120])
                time.sleep(espera)
                espera *= 2
                continue
            raise
    raise RuntimeError(f"Gemini no respondió tras {reintentos} intentos: {ultimo}")


def generar_json(prompt: str, esquema: type[T], system: str | None = None, ttl_cache_s: float = 0, grounding: bool = False) -> T:
    """Llama a Gemini y valida la respuesta con pydantic; reintenta una vez si no parsea."""
    ultimo: Exception | None = None
    for intento in range(2):
        extra = "" if intento == 0 else "\n\nRESPONDE SOLO CON EL OBJETO JSON, sin texto adicional."
        texto = generar(prompt + extra, system=system, json_mode=not grounding, grounding=grounding, ttl_cache_s=ttl_cache_s if intento == 0 else 0)
        try:
            return esquema.model_validate(extraer_json(texto))
        except (ValueError, ValidationError) as e:
            ultimo = e
            log_app.warning("Gemini devolvió JSON inválido (intento %d): %s", intento + 1, str(e)[:150])
    raise ValueError(f"Gemini no devolvió JSON válido: {ultimo}")


def explicar(que: str, contexto: dict[str, Any], ttl_cache_s: float = 3600) -> str:
    """Explicación en lenguaje simple (2-4 frases, español) de un caso concreto; vacío si Gemini no está."""
    if not disponible():
        return ""
    try:
        return generar(
            f"Explica en lenguaje simple, para alguien que empieza en opciones, {que}. Usa SOLO estos datos y no inventes números. Máximo 4 frases, en español.\n\nDATOS:\n{json.dumps(contexto, ensure_ascii=False, default=str)}",
            system="Eres un profesor paciente de finanzas. Nunca das consejo de inversión personalizado; explicas qué significan los números.",
            ttl_cache_s=ttl_cache_s, max_tokens=400,
        ).strip()
    except Exception as e:
        log_app.warning("Gemini no pudo explicar '%s': %s", que[:40], e)
        return ""
