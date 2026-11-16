"""Tests de Black-Scholes contra valores conocidos (Hull, ejemplo clásico S=K=100 o similares)."""
import math

import pytest

from quant.black_scholes import griegas, precio, volatilidad_implicita


def test_precio_call_put_referencia():
    """S=100, K=100, T=1, r=5%, σ=20%: call ≈ 10.4506, put ≈ 5.5735."""
    assert precio(100, 100, 1, 0.05, 0.2, "call") == pytest.approx(10.4506, abs=1e-3)
    assert precio(100, 100, 1, 0.05, 0.2, "put") == pytest.approx(5.5735, abs=1e-3)


def test_paridad_put_call():
    """C − P = S − K·e^{−rT}."""
    S, K, T, r, s = 95, 105, 0.5, 0.03, 0.35
    assert precio(S, K, T, r, s, "call") - precio(S, K, T, r, s, "put") == pytest.approx(S - K * math.exp(-r * T), abs=1e-9)


def test_griegas_referencia():
    """Delta de la call ATM ≈ 0.6368, gamma ≈ 0.01876, vega ≈ 0.3752/1%, theta ≈ −0.01757/día."""
    g = griegas(100, 100, 1, 0.05, 0.2, "call")
    assert g.delta == pytest.approx(0.6368, abs=1e-3)
    assert g.gamma == pytest.approx(0.01876, abs=1e-4)
    assert g.vega == pytest.approx(0.3752, abs=1e-3)
    assert g.theta == pytest.approx(-6.414 / 365, abs=1e-3)
    assert g.rho == pytest.approx(0.5323, abs=1e-3)
    gp = griegas(100, 100, 1, 0.05, 0.2, "put")
    assert gp.delta == pytest.approx(0.6368 - 1, abs=1e-3)


def test_griegas_por_diferencias_finitas():
    """Delta y vega coinciden con derivadas numéricas del precio."""
    S, K, T, r, s = 120, 110, 0.3, 0.02, 0.45
    h = 1e-4
    delta_num = (precio(S + h, K, T, r, s, "put") - precio(S - h, K, T, r, s, "put")) / (2 * h)
    vega_num = (precio(S, K, T, r, s + h, "put") - precio(S, K, T, r, s - h, "put")) / (2 * h) / 100
    g = griegas(S, K, T, r, s, "put")
    assert g.delta == pytest.approx(delta_num, abs=1e-5)
    assert g.vega == pytest.approx(vega_num, abs=1e-5)


def test_volatilidad_implicita_recupera_sigma():
    """La IV del precio teórico devuelve la sigma original."""
    for sigma in (0.1, 0.35, 0.9):
        p = precio(100, 110, 0.25, 0.04, sigma, "call")
        assert volatilidad_implicita(p, 100, 110, 0.25, 0.04, "call") == pytest.approx(sigma, abs=1e-5)


def test_volatilidad_implicita_fuera_de_rango():
    """Precios por debajo del intrínseco descontado no tienen IV."""
    assert volatilidad_implicita(0.5, 100, 80, 0.5, 0.05, "call") is None


def test_vencimiento_devuelve_intrinseco():
    assert precio(110, 100, 0, 0.05, 0.2, "call") == 10
    assert precio(90, 100, 0, 0.05, 0.2, "put") == 10
