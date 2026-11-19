"""Tests de las órdenes netas en Alpaca, del mercado cerrado y del filtro de contratos de Deribit (sin red)."""
import pandas as pd
import pytest

from bot import executor, runner
from bot.strategies.base import Signal
from core import db, execution


@pytest.mark.parametrize("actual, objetivo, esperado", [
    (0, 6.4, [("buy", 6.4, False)]),                                   # abrir largo, fracciones permitidas
    (6.4, 0, [("sell", 6.4, False)]),                                  # cerrar largo
    (0, -5, [("sell", 5, True)]),                                      # abrir corto entero
    (0.3369, -5, [("sell", 0.3369, False), ("sell", 5, True)]),        # cruzar de largo a corto: cerrar y luego abrir
    (-5, 6.4, [("buy", 5, False), ("buy", 6.4, False)]),               # cruzar de corto a largo
    (-5, -11, [("sell", 6, True)]),                                    # agrandar un corto
    (6.4, 6.4, []),                                                    # sin cambio neto: no se envía nada
    (0, -4.6631, [("sell", 4, True)]),                                 # corto fraccionario: se redondea hacia cero
])
def test_plan_alpaca(actual, objetivo, esperado):
    """El plan nunca pide vender más de lo que hay ni cortos con fracciones."""
    plan = executor.plan_alpaca(actual, objetivo)
    assert [(l, pytest.approx(q), c) for l, q, c in plan] == [(l, pytest.approx(q), c) for l, q, c in esperado]


@pytest.fixture
def bot_alpaca(tmp_path, monkeypatch):
    """Activo de Alpaca con dos estrategias, broker simulado y reloj de mercado controlable."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "neto.db"))
    db.inicializar()
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts, training_status) VALUES ('NVDA','NVIDIA','accion','alpaca_paper',1,?,'listo')", (db.ahora_iso(),))
    for e in ("ema50_chandelier", "macd_cross"):
        db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,?,1,1000,NULL,0,'')", (aid, e))
    broker = {"neto": 0.0, "ordenes": [], "abierto": True}

    def enviar(symbol, lado, q, corto):
        if lado == "sell" and not corto and q > broker["neto"] + 1e-9:
            raise RuntimeError("insufficient qty available")
        if corto and broker["neto"] > 1e-9:  # como Alpaca: no se abre un corto con la cuenta en largo
            raise RuntimeError("insufficient qty available for order")
        if corto and q != int(q):
            raise RuntimeError("fractional orders cannot be sold short")
        broker["neto"] += q if lado == "buy" else -q
        broker["ordenes"].append((lado, round(q, 4), corto))
        return {"id": str(len(broker["ordenes"])), "status": "filled", "qty": str(q), "filled_qty": str(q), "filled_avg_price": "150"}

    monkeypatch.setattr(executor.alpaca, "disponible", lambda: True)
    monkeypatch.setattr(executor.alpaca, "mercado_abierto", lambda: broker["abierto"])
    monkeypatch.setattr(executor.alpaca, "ordenes_abiertas", lambda s: [])
    monkeypatch.setattr(executor.alpaca, "posicion_neta", lambda s: broker["neto"])
    monkeypatch.setattr(executor.alpaca, "enviar_orden_accion", enviar)
    monkeypatch.setattr(executor, "emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_explicar_async", lambda *a, **k: None)
    asset = db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,))
    estado = lambda e: db.consultar_uno("SELECT * FROM bot_strategy_state WHERE asset_id=? AND strategy=?", (aid, e))  # noqa: E731
    return asset, estado, broker


def test_corto_con_otra_estrategia_en_largo_no_cae_a_paper(bot_alpaca):
    """Con una estrategia en largo, otra puede abrir un corto: el broker cruza en dos órdenes y ambas quedan en Alpaca."""
    asset, estado, broker = bot_alpaca
    runner._abrir(asset, estado("ema50_chandelier"), "long", 150.0, 3.0, {}, "v1")
    runner._abrir(asset, estado("macd_cross"), "short", 150.0, 3.0, {}, "v1")
    largo = db.de_json(estado("ema50_chandelier")["position_json"])
    corto = db.de_json(estado("macd_cross")["position_json"])
    assert largo["modo"] == corto["modo"] == "alpaca_paper"
    assert corto["qty"] == int(corto["qty"])
    assert broker["neto"] == pytest.approx(largo["qty"] - corto["qty"])


def test_cierre_devuelve_el_broker_a_la_suma_de_las_demas(bot_alpaca):
    """Al cerrar una estrategia, la cuenta queda con la posición neta de las que siguen abiertas."""
    asset, estado, broker = bot_alpaca
    runner._abrir(asset, estado("ema50_chandelier"), "long", 150.0, 3.0, {}, "v1")
    runner._abrir(asset, estado("macd_cross"), "short", 150.0, 3.0, {}, "v1")
    corto = db.de_json(estado("macd_cross")["position_json"])
    runner._cerrar(asset, estado("ema50_chandelier"), db.de_json(estado("ema50_chandelier")["position_json"]), 151.0, "señal")
    assert broker["neto"] == pytest.approx(-corto["qty"])
    assert db.consultar_uno("SELECT exit_reason FROM bot_trades WHERE strategy='ema50_chandelier'")["exit_reason"] == "señal"


def test_mercado_cerrado_no_abre_ni_cae_a_paper(bot_alpaca, monkeypatch):
    """Con el mercado cerrado se registra la señal, pero no se abre ninguna posición."""
    asset, estado, broker = bot_alpaca
    broker["abierto"] = False
    base = [(100.0, 100.0, 100.0)] * 80
    idx = pd.date_range("2026-11-18", periods=len(base), freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
    monkeypatch.setattr(runner, "velas", lambda s, v: df)
    monkeypatch.setattr(runner, "_sentimiento", lambda s: None)

    class Compra:
        def generate_signal(self, d):
            return Signal("BUY", 0.9, {"razon": "prueba"})

    monkeypatch.setattr(runner, "crear", lambda *a: Compra())
    runner.procesar_activo(asset)
    assert broker["ordenes"] == []
    assert db.consultar_uno("SELECT COUNT(*) n FROM bot_trades")["n"] == 0
    assert db.consultar_uno("SELECT COUNT(*) n FROM bot_signals")["n"] == 2


def test_filtro_deribit_deja_solo_contratos_existentes(monkeypatch):
    """La cadena se reduce a los contratos con el mismo vencimiento, strike y tipo que hay en Deribit."""
    monkeypatch.setattr(execution, "venue_cripto", lambda: "deribit_testnet")
    monkeypatch.setattr(execution.deribit, "instrumentos", lambda m: {"2026-12-04|58000|call": {}, "2026-12-04|56000|put": {}})
    cadena = pd.DataFrame({"expiry": ["2026-12-04"] * 3, "strike": [58000.0, 56000.0, 57000.0], "type": ["call", "put", "call"]})
    filtrada, n = execution.filtrar_cadena_deribit(cadena, "BTC")
    assert n == 2 and list(filtrada["strike"]) == [58000.0, 56000.0]
    assert execution.filtrar_cadena_deribit(cadena, "SOL")[1] == -1
