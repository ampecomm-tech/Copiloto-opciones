"""Tests de VaR y CVaR (histórico y normal) y de su integración en Monte Carlo y en el desempeño del bot."""
import numpy as np
import pandas as pd
import pytest

from quant.portfolio import metricas_desempeno
from quant.riesgo import resumen_riesgo, var_cvar_historico, var_cvar_normal


def test_normal_estandar_valores_conocidos():
    """Para N(0, 1): VaR 95 % = 1.6449 y CVaR 95 % = φ(1.6449)/0.05 = 2.0627."""
    r = var_cvar_normal(0.0, 1.0, 0.95)
    assert r["var"] == pytest.approx(1.6449, abs=1e-4)
    assert r["cvar"] == pytest.approx(2.0627, abs=1e-4)


def test_historico_converge_a_la_normal():
    """Con una muestra normal grande, el histórico se acerca al paramétrico."""
    x = np.random.default_rng(0).normal(0, 1, 400_000)
    h = var_cvar_historico(x, 0.95)
    assert h["var"] == pytest.approx(1.6449, abs=0.02)
    assert h["cvar"] == pytest.approx(2.0627, abs=0.02)


def test_historico_muestra_simple_y_cvar_mayor_que_var():
    """Pérdidas positivas, CVaR ≥ VaR, y NaN con menos de dos datos."""
    h = var_cvar_historico(np.arange(-50, 50, dtype=float), 0.95)
    assert h["var"] > 0 and h["cvar"] >= h["var"]
    assert np.isnan(var_cvar_historico([1.0])["var"])


def test_resumen_trae_niveles_95_y_99():
    """El resumen trae las dos versiones y los dos niveles."""
    r = resumen_riesgo(np.random.default_rng(1).normal(0, 2, 5000))
    for k in ("var_95", "cvar_95", "var_99", "cvar_99", "var_95_normal", "cvar_99_normal"):
        assert k in r
    assert r["cvar_99"] >= r["var_99"] >= r["var_95"]


def test_desempeno_incluye_var_y_cvar():
    """La curva de capital del bot reporta VaR y CVaR de sus retornos por operación."""
    m = metricas_desempeno(pd.Series([1000, 1010, 990, 1005, 980, 1000]), periodos_por_anio=1)
    assert m["var_95"] > 0 and m["cvar_95"] >= m["var_95"]


def test_montecarlo_devuelve_cola_y_distribuciones():
    """Monte Carlo de un long call: VaR y CVaR acotados por la pérdida máxima, histograma de precios y cuantiles."""
    from quant.montecarlo import simular

    legs = [{"type": "call", "strike": 100.0, "side": 1, "qty": 1, "price": 5.0, "iv": 0.4, "dte": 30, "expiry": "2026-12-19"}]
    e = {"legs": legs, "multiplicador": 1.0, "dte": 30, "costo_neto": 5.0, "max_perdida": 5.0, "max_ganancia": 1e9}
    mc = simular(e, 100.0, 0.045, n=20_000)
    assert 0 < mc["var_95"] <= mc["cvar_95"] <= 5.0 + 1e-9
    assert len(mc["cuantiles_pnl"]) == 101 and mc["cuantiles_pnl"][0] == pytest.approx(-5.0)
    assert sum(mc["histograma_precios"]["y"]) == 20_000
