"""Proveedor de sentimiento con xAI Grok y la herramienta `x_search` (Responses API).

Solo se activa si `XAI_API_KEY` existe y no está vacía. La orquestación, el caché
y el respaldo con Gemini viven en `agents/sentiment.py`.
"""
from __future__ import annotations

import requests

from agents.together_client import extraer_json
from core.config import cfg, env

URL = "https://api.x.ai/v1/responses"


class GrokSentimentProvider:
    """Sentimiento de X (posts de las últimas 24 h) con Grok + `x_search`."""

    nombre = "X vía Grok"
    clave = "grok"

    def disponible(self) -> bool:
        """True si hay `XAI_API_KEY`."""
        return bool(env("XAI_API_KEY"))

    def analizar(self, activo: str, nombre_largo: str) -> dict:
        """Devuelve `{score, resumen, fuentes[]}`; lanza si la API falla o el JSON no es válido."""
        n = int(cfg("sentimiento", "max_posts", default=8))
        prompt = (
            f"Busca en X los posts más relevantes de las últimas 24 horas sobre {nombre_largo} ({activo}) "
            f"como activo financiero. Evalúa el sentimiento agregado de los inversionistas. "
            f'Responde ÚNICAMENTE con un objeto JSON con estas claves: "score" (número de -1 muy bajista a 1 muy alcista), '
            f'"resumen" (2-3 frases en español), "fuentes" (lista de hasta {n} strings: autor y resumen del post).'
        )
        cuerpo = {"model": cfg("sentimiento", "modelo_grok", default="grok-4-fast"), "input": [{"role": "user", "content": prompt}], "tools": [{"type": "x_search"}]}
        r = requests.post(URL, json=cuerpo, headers={"Authorization": f"Bearer {env('XAI_API_KEY')}", "Content-Type": "application/json"}, timeout=90)
        r.raise_for_status()
        data = r.json()
        texto = data.get("output_text") or ""
        if not texto:
            for item in data.get("output", []):
                for c in item.get("content", []) or []:
                    if c.get("type") in ("output_text", "text") and c.get("text"):
                        texto += c["text"]
        d = extraer_json(texto)
        return {"score": float(d.get("score", 0)), "resumen": str(d.get("resumen", "")), "fuentes": [str(x) for x in (d.get("fuentes") or d.get("posts_clave") or [])][:n]}
