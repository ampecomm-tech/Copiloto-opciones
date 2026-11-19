"""Tests de CAViaR, de las pruebas de Kupiec y Christoffersen y de Diebold-Mariano con pérdida directa."""
import numpy as np
import pytest

from quant.caviar import analizar, christoffersen, estimar, kupiec, perdida_cuantil, pronosticar, recursion
from validation.diebold_mariano import diebold_mariano


def _garch(n=2500, semilla=3):
    """Retornos GARCH(1,1) con volatilidad agrupada, donde un VaR dinámico tiene sentido."""
    rng = np.random.default_rng(semilla)
    w, a, b = 0.02, 0.1, 0.85
    h, x = w / (1 - a - b), np.zeros(n)
    for t in range(n):
        x[t] = np.sqrt(h) * rng.standard_normal()
        h = w + a * x[t] ** 2 + b * h
    return x


def test_perdida_cuantil_es_asimetrica():
    """Con θ = 0.05, caer por debajo del cuantil cuesta 19 veces más que quedar por encima a la misma distancia."""
    assert perdida_cuantil(np.array([-1.0]), np.array([0.0]), 0.05)[0] == pytest.approx(0.95)
    assert perdida_cuantil(np.array([1.0]), np.array([0.0]), 0.05)[0] == pytest.approx(0.05)


def test_recursion_coincide_con_el_bucle():
    """El filtro lineal reproduce q_t = β0 + β1·q_{t−1} + β2·|r_{t−1}| calculado a mano."""
    r = np.random.default_rng(0).normal(0, 1, 50)
    beta = np.array([-0.1, 0.8, -0.3])
    q = recursion(r, beta, "sav", -1.5)
    manual = [-1.5]
    for t in range(1, len(r)):
        manual.append(beta[0] + beta[1] * manual[-1] + beta[2] * abs(r[t - 1]))
    np.testing.assert_allclose(q, manual)


@pytest.mark.parametrize("modelo", ["sav", "as"])
def test_estimacion_cubre_el_cinco_por_ciento(modelo):
    """En la muestra de estimación la tasa de violaciones queda cerca de θ y la recursión es estable."""
    x = _garch()
    m = estimar(x, 0.05, modelo)
    tasa = float((x < pronosticar(x, m)).mean())
    assert 0.04 <= tasa <= 0.06
    assert 0 < m.beta[1] < 1


def test_kupiec_acepta_la_tasa_correcta_y_rechaza_la_incorrecta():
    """5 violaciones en 100 al 5 % no se rechazan; 20 en 100 sí."""
    assert kupiec(np.r_[np.ones(5), np.zeros(95)].astype(bool), 0.05)["p_valor"] > 0.9
    assert kupiec(np.r_[np.ones(20), np.zeros(80)].astype(bool), 0.05)["p_valor"] < 0.001


def test_christoffersen_detecta_rachas():
    """Las mismas violaciones juntas en racha se rechazan; repartidas, no."""
    racha = np.zeros(400, int)
    racha[100:120] = 1
    repartidas = np.zeros(400, int)
    repartidas[::20] = 1
    assert christoffersen(racha, 0.05)["p_ind"] < 0.001
    assert christoffersen(repartidas, 0.05)["p_ind"] > 0.05


def test_diebold_mariano_con_perdida_directa():
    """Con pérdidas ya calculadas, el diferencial es su resta: la serie con menor pérdida gana."""
    rng = np.random.default_rng(1)
    l1, l2 = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500) + 0.3
    res = diebold_mariano(l1, l2, perdida="directa")
    assert res.mejor == "modelo_1" and res.diferencia_media == pytest.approx(float((l1 - l2).mean()))


def test_analizar_devuelve_backtest_y_comparacion():
    """El análisis completo trae series alineadas, backtests dentro y fuera de muestra y Diebold-Mariano."""
    x = _garch(1500)
    a = analizar(x)
    assert len(a["q_caviar"]) == len(x) == len(a["q_historico"])
    assert a["kupiec_fuera"]["n"] == len(x) - a["inicio_prueba"]
    assert {"caviar", "historico"} <= set(a["perdida_cuantil_fuera"])
    assert a["dm_vs_historico"]["n"] > 100


def test_reestimacion_usa_solo_informacion_pasada():
    """Cada bloque fuera de muestra se pronostica sin ver sus propios retornos: cambiar el futuro no cambia el pasado."""
    x = _garch(1500)
    a = analizar(x, reestimar_cada=100)
    y = x.copy()
    y[1300:] *= 5  # el futuro cambia mucho a partir de la vela 1300
    b = analizar(y, reestimar_cada=100)
    np.testing.assert_allclose(a["q_caviar"][:1301], b["q_caviar"][:1301])
    assert a["reestimaciones"] >= 3
