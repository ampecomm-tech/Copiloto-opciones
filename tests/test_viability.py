"""Tests del semáforo de viabilidad."""
from quant.viability import evaluar


def _estrategia(**kw):
    """Estrategia sintética con valores saludables por defecto."""
    base = {
        "tipo": "bull_call_spread", "direccion": "alcista",
        "legs": [{"bid": 5.0, "ask": 5.2, "oi": 500, "side": 1, "qty": 1}, {"bid": 2.0, "ask": 2.1, "oi": 400, "side": -1, "qty": 1}],
        "max_ganancia": 700.0, "max_perdida": 320.0, "ganancia_ilimitada": False, "perdida_ilimitada": False,
        "griegas": {"delta": 0.3, "gamma": 0.01, "theta": -2.0, "vega": 5.0, "rho": 0.1},
        "montecarlo": {"prob_ganancia": 0.55, "valor_esperado": 80.0},
    }
    base.update(kw)
    return base


OPINION = {"direccion": "alcista", "horizonte_dias": 30, "presupuesto": 500.0}


def test_verde_cuando_todo_bien():
    s = evaluar(_estrategia(), iv_hv=1.0, sentimiento=0.3, opinion=OPINION)
    assert s.color == "verde"
    assert s.alternativa is None
    assert all(s.reglas.values())


def test_rojo_si_excede_presupuesto():
    s = evaluar(_estrategia(max_perdida=900.0), iv_hv=1.0, sentimiento=0.0, opinion=OPINION)
    assert s.color == "rojo"
    assert s.reglas["dentro_presupuesto"] is False


def test_rojo_si_probabilidad_muy_baja():
    s = evaluar(_estrategia(montecarlo={"prob_ganancia": 0.1, "valor_esperado": 10.0}), iv_hv=1.0, sentimiento=0.0, opinion=OPINION)
    assert s.color == "rojo"


def test_amarillo_por_iv_cara_al_comprar():
    s = evaluar(_estrategia(tipo="long_call", ganancia_ilimitada=True), iv_hv=2.0, sentimiento=0.0, opinion=OPINION)
    assert s.reglas["iv_hv_ok"] is False
    assert s.color in ("amarillo", "rojo")
    assert s.alternativa == "bull_call_spread"


def test_sentimiento_en_contra_penaliza():
    s = evaluar(_estrategia(), iv_hv=1.0, sentimiento=-0.8, opinion=OPINION)
    assert s.reglas["sentimiento_alineado"] is False
    assert any("sentimiento" in a.lower() for a in s.advertencias)


def test_theta_excesiva_penaliza():
    s = evaluar(_estrategia(griegas={"delta": 0.3, "gamma": 0.01, "theta": -40.0, "vega": 5.0, "rho": 0.1}), iv_hv=1.0, sentimiento=0.0, opinion=OPINION)
    assert s.reglas["theta_ok"] is False
