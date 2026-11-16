"""Tests de la prueba de Diebold-Mariano con corrección HLN."""
import numpy as np

from validation.diebold_mariano import diebold_mariano


def test_modelos_iguales_no_rechaza():
    """Errores de la misma distribución → p-valor alto y empate."""
    rng = np.random.default_rng(0)
    e1, e2 = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    r = diebold_mariano(e1, e2, h=1)
    assert r.n == 500
    assert r.p_valor > 0.05
    assert r.mejor == "empate"


def test_modelo_claramente_mejor():
    """Si el modelo 1 tiene errores mucho menores, DM < 0 y gana modelo_1."""
    rng = np.random.default_rng(1)
    e1, e2 = rng.normal(0, 0.5, 300), rng.normal(0, 2.0, 300)
    r = diebold_mariano(e1, e2, h=1)
    assert r.dm_hln < 0
    assert r.p_valor < 0.01
    assert r.mejor == "modelo_1"
    r2 = diebold_mariano(e2, e1, h=1)
    assert r2.mejor == "modelo_2"


def test_correccion_hln_reduce_estadistico():
    """El factor HLN es < 1 para h > 1, así que |DM_hln| < |DM|."""
    rng = np.random.default_rng(2)
    e1, e2 = rng.normal(0, 1, 200), rng.normal(0.3, 1, 200)
    r = diebold_mariano(e1, e2, h=4)
    assert abs(r.dm_hln) < abs(r.dm)
    r1 = diebold_mariano(e1, e2, h=1)
    assert abs(r1.dm_hln) < abs(r1.dm)


def test_pocos_datos():
    r = diebold_mariano([1, 2], [1, 2])
    assert r.mejor == "empate" and np.isnan(r.p_valor)
