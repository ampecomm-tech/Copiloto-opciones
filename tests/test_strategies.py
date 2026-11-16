"""Tests de payoff y resumen de estrategias multi-pata."""
import numpy as np

from quant.strategies_options import bull_call_spread, long_call, payoff_vencimiento, resumen_estrategia


def _c(strike, bid, ask, dte=30):
    return {"type": "call", "strike": strike, "expiry": "2026-12-12", "dte": dte, "bid": bid, "ask": ask, "iv": 0.3, "oi": 100, "volume": 10, "symbol": f"C{strike}"}


def test_long_call_payoff_y_resumen():
    e = long_call(_c(100, 4.9, 5.0), 1, 100, 0.05, 100)
    assert e["costo_neto"] == 500.0          # paga el ask
    assert e["max_perdida"] == 500.0
    assert e["ganancia_ilimitada"] is True
    assert e["breakevens"][0] == np.testing.assert_allclose(e["breakevens"][0], 105.0, atol=0.1) or e["breakevens"][0]
    assert payoff_vencimiento(e["legs"], 120.0, 100)[0] == 1500.0


def test_bull_call_spread_limites():
    e = bull_call_spread(_c(100, 4.9, 5.0), _c(110, 2.0, 2.1), 2, 100, 0.05, 100)
    assert e["costo_neto"] == 2 * 100 * (5.0 - 2.0)      # compra al ask, vende al bid
    assert e["max_perdida"] == 600.0
    assert e["max_ganancia"] == 2 * 100 * 10 - 600.0
    assert e["ganancia_ilimitada"] is False and e["perdida_ilimitada"] is False
