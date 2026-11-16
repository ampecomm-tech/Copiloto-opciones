"""Cliente de Together AI: una llamada de chat que devuelve JSON validado con pydantic.

Si no hay `TOGETHER_API_KEY`, `disponible()` es False y la mesa funciona en
modo degradado (el juez decide solo con las reglas del semáforo).
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from core.config import cfg, env
from core.events import log_app

T = TypeVar("T", bound=BaseModel)
_cliente: Any = None


def disponible() -> bool:
    """True si hay API key configurada."""
    return bool(env("TOGETHER_API_KEY"))


def _get_cliente() -> Any:
    """Instancia perezosa del SDK `together`."""
    global _cliente
    if _cliente is None:
        from together import Together

        _cliente = Together(api_key=env("TOGETHER_API_KEY"), timeout=float(cfg("agentes", "timeout_s", default=45)))
    return _cliente


def extraer_json(texto: str) -> dict[str, Any]:
    """Extrae el primer objeto JSON de una respuesta (tolera ```json ... ``` y texto alrededor)."""
    texto = texto.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.S)
    if m:
        texto = m.group(1)
    else:
        i, j = texto.find("{"), texto.rfind("}")
        if i >= 0 and j > i:
            texto = texto[i : j + 1]
    return json.loads(texto)


def chat(modelo: str, system: str, user: str, max_tokens: int | None = None, temperatura: float | None = None, json_mode: bool = True) -> str:
    """Una llamada de chat a Together; devuelve el texto de la respuesta."""
    cli = _get_cliente()
    kwargs: dict[str, Any] = {
        "model": modelo,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": int(max_tokens or cfg("agentes", "max_tokens", default=900)),
        "temperature": float(cfg("agentes", "temperatura", default=0.2) if temperatura is None else temperatura),
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = cli.chat.completions.create(**kwargs)
    except Exception as e:
        if json_mode and "response_format" in str(e).lower():
            kwargs.pop("response_format")
            resp = cli.chat.completions.create(**kwargs)
        else:
            raise
    msg = resp.choices[0].message
    contenido = msg.content or ""
    if not contenido.strip():
        # Modelos de razonamiento: si agotaron los tokens pensando, el JSON puede estar en el razonamiento.
        razonamiento = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        if razonamiento and "{" in razonamiento:
            log_app.warning("%s devolvió contenido vacío; se intenta rescatar el JSON del razonamiento", modelo)
            contenido = razonamiento
    return contenido


def chat_json(modelo: str, system: str, user: str, esquema: type[T], max_tokens: int | None = None) -> T:
    """Llama al modelo y valida la respuesta con `esquema`; reintenta una vez si no parsea.

    Raises:
        ValueError si tras dos intentos la respuesta no es JSON válido según el esquema.
    """
    ultimo_error: Exception | None = None
    for intento in range(2):
        texto = chat(modelo, system, user if intento == 0 else user + "\n\nRESPONDE SOLO CON EL OBJETO JSON, sin texto adicional.", max_tokens)
        try:
            return esquema.model_validate(extraer_json(texto))
        except (ValueError, ValidationError) as e:
            ultimo_error = e
            log_app.warning("Respuesta no válida de %s (intento %d): %s", modelo, intento + 1, str(e)[:200])
    raise ValueError(f"El modelo {modelo} no devolvió JSON válido: {ultimo_error}")
