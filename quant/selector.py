"""Generación y ranking de estrategias candidatas según la opinión del usuario.

Entrada: cadena de opciones (DataFrame), spot, opinión (dirección, magnitud,
horizonte, presupuesto). Salida: top N estrategias con Monte Carlo incluido.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.config import cfg
from quant import montecarlo
from quant import strategies_options as so


class Opinion(BaseModel):
    """Opinión estructurada del usuario sobre un activo."""

    activo: str
    direccion: str = Field(description="alcista | bajista | neutral | volatil")
    magnitud_pct: float = Field(default=5.0, ge=0, description="Movimiento esperado en %")
    horizonte_dias: int = Field(default=30, ge=1)
    presupuesto: float = Field(default=500.0, gt=0, description="Pérdida máxima aceptada en USD")
    comentario: str = ""

    def normalizar(self) -> "Opinion":
        """Normaliza la dirección a los cuatro valores válidos."""
        d = self.direccion.strip().lower().replace("á", "a")
        mapa = {"alcista": "alcista", "bullish": "alcista", "sube": "alcista", "bajista": "bajista", "bearish": "bajista",
                "baja": "bajista", "neutral": "neutral", "lateral": "neutral", "volatil": "volatil", "volatile": "volatil"}
        self.direccion = mapa.get(d, "alcista")
        return self


def _vencimientos_cercanos(cadena: pd.DataFrame, horizonte: int, maximo: int) -> list[str]:
    """Elige los vencimientos más cercanos al horizonte (preferentemente ≥ horizonte)."""
    vtos = cadena.groupby("expiry")["dte"].first().reset_index()
    vtos = vtos[vtos["dte"] >= 1]
    if vtos.empty:
        return []
    # Preferimos vencimientos que cubran el horizonte (≥ 70 % de él); si no hay, los más largos disponibles.
    cubren = vtos[(vtos["dte"] >= horizonte * 0.7) & (vtos["dte"] <= horizonte * 2.5)]
    if cubren.empty:
        cubren = vtos[vtos["dte"] >= horizonte * 0.7]
    vtos = cubren if not cubren.empty else vtos.sort_values("dte", ascending=False).head(maximo)
    vtos = vtos.assign(dist=(vtos["dte"] - horizonte).abs())
    return vtos.sort_values("dist")["expiry"].head(maximo).tolist()


def _liquido(fila: pd.Series, min_oi: float, max_spread: float, S: float, prima_min_rel: float = 0.0015) -> bool:
    """Filtro de liquidez: bid/ask válidos, open interest, spread relativo y prima mínima.

    La prima mínima (0.15% del spot) descarta opciones de centavos que generan
    spreads "gratis" con cantidades absurdas.
    """
    if fila["bid"] <= 0 or fila["ask"] <= 0 or fila["ask"] < fila["bid"]:
        return False
    mid = (fila["bid"] + fila["ask"]) / 2
    if mid < prima_min_rel * S:
        return False
    spread_rel = (fila["ask"] - fila["bid"]) / mid if mid > 0 else 1.0
    return bool((fila.get("oi", 0) or 0) >= min_oi and spread_rel <= max_spread)


def _strikes_cerca(sub: pd.DataFrame, objetivo: float, n: int) -> list[dict[str, Any]]:
    """Los `n` contratos con strike más cercano a `objetivo`."""
    if sub.empty:
        return []
    idx = (sub["strike"] - objetivo).abs().argsort()[:n]
    return [sub.iloc[i].to_dict() for i in idx]


def _qty(costo_unitario: float, presupuesto: float, paso: float) -> float:
    """Cantidad de contratos que cabe en el presupuesto, redondeada al paso."""
    if costo_unitario <= 0:
        return paso
    q = math.floor(presupuesto / costo_unitario / paso) * paso
    return max(round(q, 4), 0.0)


def generar_candidatas(cadena: pd.DataFrame, S: float, opinion: Opinion, r: float, multiplicador: float, paso_qty: float) -> list[so.Estrategia]:
    """Genera todas las estrategias coherentes con la opinión probando strikes y vencimientos.

    Reglas: solo patas líquidas, pérdida máxima ≤ presupuesto y nunca pérdida ilimitada.
    """
    op = opinion.normalizar()
    max_v = int(cfg("opciones", "max_vencimientos", default=4))
    n_k = int(cfg("opciones", "max_strikes_por_lado", default=6))
    min_oi = float(cfg("opciones", "min_open_interest", default=10))
    max_spread = float(cfg("opciones", "max_spread_relativo", default=0.25))
    mag = op.magnitud_pct / 100.0
    objetivo_up, objetivo_dn = S * (1 + mag), S * (1 - mag)

    liq = cadena[cadena.apply(lambda f: _liquido(f, min_oi, max_spread, S), axis=1)].copy()
    candidatas: list[so.Estrategia] = []

    def agregar(constructor, *contratos) -> None:
        """Construye con qty=1 unidad, re-escala al presupuesto y añade si es válida."""
        try:
            base = constructor(*contratos, 1.0, S, r, multiplicador)
        except Exception:
            return
        if base["perdida_ilimitada"] or base["max_perdida"] <= 0:
            return
        q = _qty(base["max_perdida"], op.presupuesto, paso_qty)
        if q <= 0:
            return
        est = constructor(*contratos, q, S, r, multiplicador)
        if est["max_perdida"] > op.presupuesto * 1.0001:
            return
        candidatas.append(est)

    for exp in _vencimientos_cercanos(liq, op.horizonte_dias, max_v):
        sub = liq[liq["expiry"] == exp]
        calls = sub[sub["type"] == "call"].sort_values("strike")
        puts = sub[sub["type"] == "put"].sort_values("strike")
        if calls.empty or puts.empty:
            continue

        if op.direccion == "alcista":
            for c in _strikes_cerca(calls, S, n_k):
                agregar(so.long_call, c)
                for c2 in _strikes_cerca(calls[calls["strike"] > c["strike"]], objetivo_up, 3):
                    agregar(so.bull_call_spread, c, c2)
            for p2 in _strikes_cerca(puts[puts["strike"] <= S], S * (1 - mag / 2), 3):
                for p1 in _strikes_cerca(puts[puts["strike"] < p2["strike"]], objetivo_dn, 2):
                    agregar(so.bull_put_spread, p1, p2)

        elif op.direccion == "bajista":
            for p in _strikes_cerca(puts, S, n_k):
                agregar(so.long_put, p)
                for p1 in _strikes_cerca(puts[puts["strike"] < p["strike"]], objetivo_dn, 3):
                    agregar(so.bear_put_spread, p1, p)
            for c1 in _strikes_cerca(calls[calls["strike"] >= S], S * (1 + mag / 2), 3):
                for c2 in _strikes_cerca(calls[calls["strike"] > c1["strike"]], objetivo_up, 2):
                    agregar(so.bear_call_spread, c1, c2)

        elif op.direccion == "neutral":
            for p2 in _strikes_cerca(puts[puts["strike"] < S], objetivo_dn, 3):
                for c1 in _strikes_cerca(calls[calls["strike"] > S], objetivo_up, 3):
                    for p1 in _strikes_cerca(puts[puts["strike"] < p2["strike"]], p2["strike"] * (1 - mag), 2):
                        for c2 in _strikes_cerca(calls[calls["strike"] > c1["strike"]], c1["strike"] * (1 + mag), 2):
                            agregar(so.iron_condor, p1, p2, c1, c2)
            for c2 in _strikes_cerca(calls, S, 3):
                k = c2["strike"]
                for ancho in (mag * S, 2 * mag * S):
                    # Alas simétricas: el strike superior debe estar exactamente a la misma distancia que el inferior.
                    c1s = _strikes_cerca(calls[calls["strike"] < k], k - ancho, 1)
                    if not c1s:
                        continue
                    c3s = calls[np.isclose(calls["strike"], 2 * k - c1s[0]["strike"])]
                    if len(c3s):
                        agregar(so.long_butterfly, c1s[0], c2, c3s.iloc[0].to_dict())
                p0s = _strikes_cerca(puts[puts["strike"] == k], k, 1)
                if p0s:
                    p1s = _strikes_cerca(puts[puts["strike"] < k], k * (1 - mag), 1)
                    if p1s:
                        c2s = calls[np.isclose(calls["strike"], 2 * k - p1s[0]["strike"])]
                        if len(c2s):
                            agregar(so.iron_butterfly, p1s[0], p0s[0], c2, c2s.iloc[0].to_dict())

        elif op.direccion == "volatil":
            for c in _strikes_cerca(calls, S, 3):
                ps = _strikes_cerca(puts[puts["strike"] == c["strike"]], c["strike"], 1)
                if ps:
                    agregar(so.long_straddle, c, ps[0])
            for p in _strikes_cerca(puts[puts["strike"] < S], objetivo_dn, 3):
                for c in _strikes_cerca(calls[calls["strike"] > S], objetivo_up, 3):
                    agregar(so.long_strangle, p, c)

    # Quitar duplicados exactos (mismas patas)
    vistos: set[str] = set()
    unicas: list[so.Estrategia] = []
    for e in candidatas:
        clave = e["tipo"] + "|" + "|".join(f"{l['type']}{l['strike']}{l['side']}{l['expiry']}" for l in e["legs"])
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(e)
    return unicas


def puntuar(estrategia: so.Estrategia, mc: dict[str, Any]) -> float:
    """Puntaje para el ranking: EV/riesgo + probabilidad de ganancia, con castigo a spreads anchos."""
    spread_rel = np.mean([(l["ask"] - l["bid"]) / max((l["ask"] + l["bid"]) / 2, 1e-9) for l in estrategia["legs"]])
    return float(0.6 * mc["ev_sobre_riesgo"] + 0.4 * mc["prob_ganancia"] - 0.5 * spread_rel)


def seleccionar(cadena: pd.DataFrame, S: float, opinion: Opinion, r: float, multiplicador: float = 1.0, paso_qty: float = 1.0, top_n: int | None = None) -> list[so.Estrategia]:
    """Pipeline completo: genera candidatas, corre Monte Carlo, rankea y devuelve el top N.

    Cada estrategia devuelta incluye la clave `montecarlo` y `score`.
    """
    top_n = int(top_n or cfg("opciones", "top_n", default=3))
    candidatas = generar_candidatas(cadena, S, opinion, r, multiplicador, paso_qty)
    # Monte Carlo rápido (2000 trayectorias) para preseleccionar, luego completo en el top.
    for e in candidatas:
        e["montecarlo"] = montecarlo.simular(e, S, r, opinion.horizonte_dias, n=2000, opinion=opinion.model_dump())
        e["score"] = puntuar(e, e["montecarlo"])
    candidatas.sort(key=lambda e: e["score"], reverse=True)
    # Diversidad: no más de 2 del mismo tipo en el top
    top: list[so.Estrategia] = []
    conteo: dict[str, int] = {}
    for e in candidatas:
        if conteo.get(e["tipo"], 0) >= 2:
            continue
        conteo[e["tipo"]] = conteo.get(e["tipo"], 0) + 1
        top.append(e)
        if len(top) == top_n:
            break
    for e in top:
        e["montecarlo"] = montecarlo.simular(e, S, r, opinion.horizonte_dias, opinion=opinion.model_dump())
        e["score"] = puntuar(e, e["montecarlo"])
    return top
