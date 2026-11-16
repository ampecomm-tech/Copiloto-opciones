"""Tests del enrutador de ejecución y de la validación BS vs Binance (sin red)."""
import os

import pandas as pd

os.environ.setdefault("DB_PATH", "data/test_exec.db")

from core import db  # noqa: E402
from core import execution  # noqa: E402
from validation.bs_vs_binance import comparar, resumen  # noqa: E402
from quant.black_scholes import griegas  # noqa: E402

db.inicializar()

ESTRATEGIA = {"nombre": "Long call prueba", "activo": "BTC", "legs": [{"symbol": "BTC-TEST-C", "side": 1, "qty": 0.1, "price": 100.0, "tick_size": 0.1}], "multiplicador": 1}


def test_sin_credenciales_cae_a_paper_y_emite_evento():
    """Sin llaves de Binance la orden cae a paper interno y queda un evento en `sistema`."""
    antes = len(db.consultar("SELECT id FROM events WHERE channel='sistema'"))
    r = execution.ejecutar(ESTRATEGIA, "binance_options_testnet")
    assert r["venue"] == "paper" and r["order_ids"] == []
    assert "respaldo" in r["detalle"].lower()
    assert len(db.consultar("SELECT id FROM events WHERE channel='sistema'")) == antes + 1


def test_venue_desconocido_cae_a_paper():
    r = execution.ejecutar(ESTRATEGIA, "otro_venue")
    assert r["venue"] == "paper"


def test_redondeo_tick():
    assert execution._redondear_tick(101.23, 0.5) == 101.0
    assert execution._redondear_tick(101.26, 0.5) == 101.5


def test_bs_vs_binance_sintetico():
    """Si el mark price sale de nuestro propio BS, el error debe ser ~0 y la IV recuperada igual a la de marca."""
    S, r = 100.0, 0.05
    filas = []
    for K in (90, 100, 110):
        for t in ("call", "put"):
            g = griegas(S, K, 30 / 365, r, 0.5, t)
            filas.append({"symbol": f"X-{K}-{t}", "type": t, "strike": K, "dte": 30, "mark_iv": 0.5, "mark_price": g.precio, "bin_r": r,
                          "bin_delta": g.delta, "bin_gamma": g.gamma, "bin_theta": g.theta, "bin_vega": g.vega})
    tabla = comparar(pd.DataFrame(filas), S)
    res = resumen(tabla)
    assert res["n"] == 6
    assert res["mae_precio"] < 1e-6 and res["mae_iv"] < 1e-5 and res["mae_delta"] < 1e-9
