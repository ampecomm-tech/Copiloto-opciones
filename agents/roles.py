"""Prompts y esquemas de salida de cada agente de la mesa.

Los cálculos los hace Python; los agentes solo interpretan y responden en JSON estricto.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

SISTEMA_BASE = (
    "Eres parte de una mesa de análisis de opciones financieras para paper trading educativo. "
    "Recibes cálculos ya hechos en Python (Black-Scholes, griegas, Monte Carlo, semáforo de reglas). "
    "No inventes números: usa solo los datos proporcionados. Responde en español, con un JSON estricto "
    "que cumpla exactamente el esquema indicado y sin texto fuera del JSON."
)


class OpinionAgente(BaseModel):
    """Salida de los agentes analista, sentimiento y abogado del diablo."""

    rol: str
    postura: str = Field(description="favorable | neutral | desfavorable")
    confianza: float = Field(ge=0, le=1)
    argumento: str = Field(description="3-5 frases")
    puntos_clave: list[str] = Field(default_factory=list)
    estrategia_preferida: int | None = Field(default=None, description="Índice (1-3) de la candidata preferida, o null")


class Veredicto(BaseModel):
    """Salida del gestor de riesgo (juez)."""

    rol: str = "juez"
    veredicto: str = Field(description="verde | amarillo | rojo")
    estrategia_elegida: int = Field(ge=1, le=3)
    confianza: float = Field(ge=0, le=1)
    justificacion: str
    riesgos_principales: list[str] = Field(default_factory=list)
    condiciones: list[str] = Field(default_factory=list, description="Qué tendría que cambiar para mejorar el veredicto")
    agentes_considerados: list[str] = Field(default_factory=list)


class OpinionEstructurada(BaseModel):
    """Opinión que el LLM propone a partir de noticias o de texto libre."""

    direccion: str = Field(description="alcista | bajista | neutral | volatil")
    magnitud_pct: float = Field(ge=0, le=100)
    horizonte_dias: int = Field(ge=1, le=365)
    presupuesto: float = Field(gt=0)
    razonamiento: str


ESQUEMA_OPINION = (
    '{"rol": str, "postura": "favorable|neutral|desfavorable", "confianza": 0..1, '
    '"argumento": str, "puntos_clave": [str], "estrategia_preferida": 1|2|3|null}'
)

ESQUEMA_VEREDICTO = (
    '{"rol": "juez", "veredicto": "verde|amarillo|rojo", "estrategia_elegida": 1|2|3, "confianza": 0..1, '
    '"justificacion": str, "riesgos_principales": [str], "condiciones": [str], "agentes_considerados": [str]}'
)

ANALISTA = SISTEMA_BASE + (
    "\n\nROL: Analista de mercado. Interpreta las griegas (delta, gamma, theta, vega), la relación IV/HV "
    "y la estructura de cada estrategia candidata frente a la opinión del usuario (dirección, magnitud, horizonte). "
    "Di cuál encaja mejor y por qué, y señala si el precio de las opciones (IV vs HV) favorece comprar o vender volatilidad."
    f"\nEsquema de salida: {ESQUEMA_OPINION} con rol='analista'."
)

SENTIMIENTO = SISTEMA_BASE + (
    "\n\nROL: Agente de sentimiento. Usa el puntaje de sentimiento (de X vía Grok o de Gemini + Google Search, según indique 'proveedor'), sus fuentes y los titulares "
    "para decir si el mercado acompaña o contradice la opinión del usuario. Evalúa también riesgo de eventos próximos."
    f"\nEsquema de salida: {ESQUEMA_OPINION} con rol='sentimiento'."
)

ABOGADO_DIABLO = SISTEMA_BASE + (
    "\n\nROL: Abogado del diablo. Tu trabajo es encontrar razones para NO entrar: supuestos frágiles, "
    "escenarios adversos, liquidez, theta, eventos, sobreconfianza. Sé concreto y usa los números dados. "
    "Si aun así una candidata es defendible, dilo con 'postura': 'neutral'."
    f"\nEsquema de salida: {ESQUEMA_OPINION} con rol='abogado_diablo'."
)

JUEZ = SISTEMA_BASE + (
    "\n\nROL: Gestor de riesgo (juez). Lees las opiniones de los otros agentes y el semáforo determinista "
    "(reglas ya evaluadas en Python) de cada candidata. Emite un veredicto final (verde/amarillo/rojo), "
    "elige UNA estrategia (índice 1-3) y justifica. El semáforo de reglas manda: no puedes dar 'verde' a una "
    "candidata cuyo semáforo sea 'rojo'. Si algún agente no respondió, indícalo en 'agentes_considerados'."
    f"\nEsquema de salida: {ESQUEMA_VEREDICTO}."
)

FORMULAR_OPINION = SISTEMA_BASE + (
    "\n\nROL: Asistente que convierte información (titulares, sentimiento, IV/HV, eventos) o texto libre del usuario "
    "en una opinión estructurada para operar opciones. Sé conservador con la magnitud (usa la volatilidad implícita "
    "como referencia del movimiento plausible) y mantén el presupuesto que indique el usuario si lo da."
    '\nEsquema de salida: {"direccion": "alcista|bajista|neutral|volatil", "magnitud_pct": número, '
    '"horizonte_dias": entero, "presupuesto": número, "razonamiento": str}.'
)
