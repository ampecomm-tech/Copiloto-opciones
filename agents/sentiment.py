"""Sentimiento del activo con proveedores intercambiables.

- `GrokSentimentProvider` (X vía Grok, `x_search`): solo si `XAI_API_KEY` existe.
- `GeminiSentimentProvider` (Gemini + búsqueda en Google): noticias y lo que se comenta
  del activo en las últimas 24-48 h; para cripto se combina con el índice Fear & Greed.

Selección automática: Grok si hay llave; si no, Gemini. Si Grok falla en tiempo de
ejecución se cae a Gemini y se emite un evento a `sistema`. Ambos devuelven el mismo
JSON `{score: -1..1, resumen, fuentes[], proveedor}`. Caché de 15 min por activo en
`sentiment_cache`; cada resultado se emite a `sentimiento` indicando la fuente.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agents import gemini_client as gem
from agents.grok_sentiment import GrokSentimentProvider
from core import db
from core.config import cfg
from core.events import emit, log_app


class ResultadoSentimiento(BaseModel):
    """Salida común de todos los proveedores."""

    score: float = Field(ge=-1, le=1)
    resumen: str
    fuentes: list[str] = Field(default_factory=list)
    proveedor: str = ""


class Proveedor(Protocol):
    """Interfaz de un proveedor de sentimiento."""

    nombre: str
    clave: str

    def disponible(self) -> bool: ...
    def analizar(self, activo: str, nombre_largo: str) -> dict: ...


class _SalidaGemini(BaseModel):
    """JSON que pedimos a Gemini."""

    score: float = Field(ge=-1, le=1)
    resumen: str
    fuentes: list[str] = Field(default_factory=list)


class GeminiSentimentProvider:
    """Gemini con búsqueda en Google (grounding) sobre noticias y comentarios recientes; Fear & Greed para cripto.

    Si la cuenta no tiene cuota de grounding, usa los titulares ya descargados
    (`data_sources.news`) como contexto y lo indica en el nombre del proveedor.
    """

    nombre = "Gemini + Google Search"
    clave = "gemini"

    def disponible(self) -> bool:
        """True si hay `GEMINI_API_KEY`."""
        return gem.disponible()

    def analizar(self, activo: str, nombre_largo: str) -> dict:
        """Devuelve `{score, resumen, fuentes[], proveedor}`; lanza si Gemini falla."""
        from data_sources.market import es_cripto
        from data_sources.news import fear_and_greed, noticias

        hoy = datetime.now(timezone.utc).date().isoformat()
        n = int(cfg("sentimiento", "max_posts", default=8))
        system = "Eres un analista de sentimiento de mercado. No inventes fuentes."
        salida = (f'Responde ÚNICAMENTE con un objeto JSON: {{"score": número de -1 (muy bajista) a 1 (muy alcista), '
                  f'"resumen": "2-3 frases en español", "fuentes": [hasta {n} strings con medio/autor y la idea clave]}}.')
        proveedor = self.nombre
        try:
            prompt = (f"Hoy es {hoy}. Busca en Google noticias y comentarios (analistas, redes, foros) de las últimas 24-48 horas sobre "
                      f"{nombre_largo} ({activo}) como activo financiero y evalúa el sentimiento agregado de los inversionistas. " + salida)
            r = gem.generar_json(prompt, _SalidaGemini, system=system, grounding=True)
        except gem.GroundingNoDisponible:
            titulares = noticias(activo)
            if not titulares:
                raise RuntimeError("sin búsqueda de Google ni titulares para analizar")
            proveedor = "Gemini + titulares (sin Google Search)"
            prompt = (f"Hoy es {hoy}. Evalúa el sentimiento agregado de los inversionistas sobre {nombre_largo} ({activo}) a partir de estos titulares "
                      f"de las últimas 24-48 horas. Usa solo esta información. " + salida + "\n\nTITULARES:\n"
                      + "\n".join(f"- [{t['fuente']}] {t['titulo']}" for t in titulares[:15]))
            r = gem.generar_json(prompt, _SalidaGemini, system=system, ttl_cache_s=600)
        score, resumen = float(r.score), r.resumen
        fuentes = list(r.fuentes)[:n]
        if es_cripto(activo):
            fng = fear_and_greed()
            if fng:
                fng_score = (float(fng["valor"]) - 50.0) / 50.0
                score = max(-1.0, min(1.0, 0.7 * score + 0.3 * fng_score))
                fuentes.append(f"Fear & Greed (alternative.me): {fng['valor']} · {fng['clasificacion']}")
                resumen += f" Índice Fear & Greed en {fng['valor']} ({fng['clasificacion']})."
        return {"score": score, "resumen": resumen, "fuentes": fuentes, "proveedor": proveedor}


PROVEEDORES: tuple[Any, ...] = (GrokSentimentProvider(), GeminiSentimentProvider())
"""Orden de preferencia: Grok (si hay llave) y luego Gemini."""


def proveedor_activo() -> Any | None:
    """Primer proveedor disponible (Grok si hay llave; si no, Gemini); None si ninguno."""
    for p in PROVEEDORES:
        if p.disponible():
            return p
    return None


def leer_cache(activo: str) -> dict[str, Any] | None:
    """Sentimiento cacheado si tiene menos de `config.sentimiento.ttl_min` minutos."""
    fila = db.consultar_uno("SELECT * FROM sentiment_cache WHERE asset = ?", (activo,))
    if not fila:
        return None
    try:
        ts = datetime.fromisoformat(fila["ts"])
    except ValueError:
        return None
    if datetime.now(timezone.utc) - ts > timedelta(minutes=float(cfg("sentimiento", "ttl_min", default=15))):
        return None
    return {"activo": activo, "ts": fila["ts"], "score": fila["score"], "resumen": fila["summary"], "fuentes": db.de_json(fila["posts_json"], []),
            "proveedor": fila.get("provider") or "", "cache": True, "disponible": True}


def _guardar(activo: str, r: ResultadoSentimiento) -> str:
    """Guarda en `sentiment_cache` y devuelve el timestamp."""
    ts = db.ahora_iso()
    db.ejecutar(
        "INSERT INTO sentiment_cache(asset, ts, score, summary, posts_json, provider) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(asset) DO UPDATE SET ts=excluded.ts, score=excluded.score, summary=excluded.summary, posts_json=excluded.posts_json, provider=excluded.provider",
        (activo, ts, r.score, r.resumen, db.a_json(r.fuentes), r.proveedor),
    )
    return ts


def sentimiento(activo: str, nombre_largo: str | None = None, forzar: bool = False, user_id: str | None = None) -> dict[str, Any]:
    """Sentimiento del activo con caché de 15 min, selección automática de proveedor y respaldo.

    Returns:
        `{activo, ts, score, resumen, fuentes, proveedor, cache, disponible}`. Si ningún
        proveedor está disponible o todos fallan: `score=0`, `disponible=False`.
    """
    activo = activo.upper()
    if not forzar:
        c = leer_cache(activo)
        if c:
            return c
    nombre = nombre_largo or activo
    candidatos = [p for p in PROVEEDORES if p.disponible()]
    if not candidatos:
        return {"activo": activo, "score": 0.0, "resumen": "Sentimiento no disponible (faltan XAI_API_KEY y GEMINI_API_KEY).", "fuentes": [], "proveedor": "", "disponible": False}
    resultado: ResultadoSentimiento | None = None
    for i, p in enumerate(candidatos):
        try:
            d = p.analizar(activo, nombre)
            resultado = ResultadoSentimiento(score=max(-1.0, min(1.0, float(d["score"]))), resumen=str(d["resumen"]), fuentes=[str(f) for f in d.get("fuentes", [])], proveedor=str(d.get("proveedor") or p.nombre))
            break
        except Exception as e:
            log_app.error("Proveedor de sentimiento %s falló para %s: %s", p.nombre, activo, e)
            siguiente = candidatos[i + 1].nombre if i + 1 < len(candidatos) else None
            emit("sistema", "warning", f"Sentimiento: {p.nombre} falló para {activo}" + (f", se usa {siguiente}" if siguiente else ""), {"error": str(e)[:300]}, user_id)
    if resultado is None:
        return {"activo": activo, "score": 0.0, "resumen": "No se pudo obtener el sentimiento con ningún proveedor.", "fuentes": [], "proveedor": "", "disponible": False}
    ts = _guardar(activo, resultado)
    tipo = "verde" if resultado.score > 0.2 else "rojo" if resultado.score < -0.2 else "amarillo"
    emit("sentimiento", tipo, f"Sentimiento de {activo}: {resultado.score:+.2f} · Fuente: {resultado.proveedor}",
         {"score": resultado.score, "proveedor": resultado.proveedor, "resumen": resultado.resumen, "fuentes": resultado.fuentes[:5]}, user_id)
    return {"activo": activo, "ts": ts, **resultado.model_dump(), "cache": False, "disponible": True}


def ultimo_score(activo: str) -> float | None:
    """Último puntaje guardado (aunque esté vencido), para features del bot."""
    fila = db.consultar_uno("SELECT score FROM sentiment_cache WHERE asset = ?", (activo.upper(),))
    return float(fila["score"]) if fila else None


# Compatibilidad con el nombre anterior.
sentimiento_x = sentimiento
