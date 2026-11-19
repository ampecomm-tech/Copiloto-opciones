"""Tests del trailing stop, de la revisión de velas del runner y de la confirmación de órdenes en Alpaca (sin red)."""
import pandas as pd
import pytest

from bot import executor, risk, runner
from bot.strategies.base import Signal
from core import db

P = risk.ParametrosRiesgo(trailing_habilitado=True, trailing_activacion_atr=1.0, trailing_distancia_atr=2.0)


def _largo(**extra):
    """Posición larga con entrada en 100, SL en 85 y TP en 130 (ATR 10)."""
    return {"side": "long", "qty": 1.0, "entry_price": 100.0, "sl": 85.0, "tp": 130.0, "trade_id": 1, **extra}


def _corto(**extra):
    """Posición corta con entrada en 100, SL en 115 y TP en 70 (ATR 10)."""
    return {"side": "short", "qty": 1.0, "entry_price": 100.0, "sl": 115.0, "tp": 70.0, "trade_id": 1, **extra}


def test_no_se_arma_sin_ganancia_suficiente():
    """Por debajo de la activación el stop no cambia."""
    n = risk.actualizar_trailing(_largo(), 109.0, 99.0, 10.0, P)
    assert n["sl"] == 85.0 and n["trailing_activo"] is False and n["extremo"] == 109.0


def test_largo_sube_el_stop_y_nunca_lo_baja():
    """El stop sigue al máximo y no retrocede aunque el precio caiga o el ATR crezca."""
    n = risk.actualizar_trailing(_largo(), 130.0, 120.0, 10.0, P)
    assert n["trailing_activo"] is True and n["sl"] == 110.0
    n2 = risk.actualizar_trailing(n, 125.0, 115.0, 20.0, P)
    assert n2["sl"] == 110.0 and n2["extremo"] == 130.0


def test_corto_baja_el_stop_y_nunca_lo_sube():
    """Caso simétrico para cortos."""
    n = risk.actualizar_trailing(_corto(), 80.0, 70.0, 10.0, P)
    assert n["trailing_activo"] is True and n["sl"] == 90.0
    assert risk.actualizar_trailing(n, 85.0, 75.0, 20.0, P)["sl"] == 90.0


def test_deshabilitado_conserva_stop_fijo():
    """Con el trailing apagado la posición queda igual."""
    p = risk.ParametrosRiesgo(trailing_habilitado=False)
    assert risk.actualizar_trailing(_largo(), 200.0, 150.0, 10.0, p) == _largo()


def test_salidas_con_trailing_armado():
    """Armado, el stop se reporta como TRAILING y el TP fijo deja de aplicar."""
    armada = _largo(trailing_activo=True, sl=110.0)
    assert risk.evaluar_salida(armada, 140.0, 115.0, 135.0) is None
    assert risk.evaluar_salida(armada, 120.0, 109.0, 112.0) == "TRAILING"
    assert risk.evaluar_salida(_largo(), 131.0, 95.0, 120.0) == "TP"
    assert risk.evaluar_salida(_largo(), 105.0, 84.0, 90.0) == "SL"


@pytest.fixture
def bot_prueba(tmp_path, monkeypatch):
    """Base temporal con un activo paper y el runner aislado de red, Discord y Gemini."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "bot.db"))
    db.inicializar()
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts, training_status) VALUES ('TEST','Test','accion','paper',1,?,'listo')", (db.ahora_iso(),))
    db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,'macd_cross',1,1000,NULL,0,'')", (aid,))
    monkeypatch.setattr(runner, "emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_explicar_async", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_sentimiento", lambda s: None)
    monkeypatch.setattr(runner, "atr", lambda df, n: pd.Series([10.0] * len(df), index=df.index))
    monkeypatch.setattr(runner.risk.ParametrosRiesgo, "desde_config", classmethod(lambda cls: P))
    senal = {"accion": "BUY"}

    class Estrategia:
        def generate_signal(self, df):
            return Signal(senal["accion"], 0.9, {"razon": "prueba"})

    monkeypatch.setattr(runner, "crear", lambda *a: Estrategia())
    velas = {"df": None}
    monkeypatch.setattr(runner, "velas", lambda s, v: velas["df"])

    def ciclo(filas):
        """Corre un ciclo con velas (alto, bajo, cierre); agrega al final la vela en formación que el bot descarta."""
        base = [(100.0, 100.0, 100.0)] * 60 + filas + [(0.0, 0.0, 0.0)]
        idx = pd.date_range("2026-11-18", periods=len(base), freq="15min", tz="UTC")
        velas["df"] = pd.DataFrame({"open": [b[2] for b in base], "high": [b[0] for b in base], "low": [b[1] for b in base], "close": [b[2] for b in base], "volume": 1.0}, index=idx)
        runner.procesar_activo(db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,)))
        estado = db.consultar_uno("SELECT * FROM bot_strategy_state WHERE asset_id=?", (aid,))
        return db.de_json(estado["position_json"])

    return ciclo, senal


def test_vela_de_entrada_no_dispara_salida(bot_prueba):
    """Una mecha previa a la entrada no cierra la posición aunque el ciclo revise otra vez esa vela."""
    ciclo, senal = bot_prueba
    entrada = [(101.0, 70.0, 100.0)]
    pos = ciclo(entrada)
    assert pos and pos["sl"] == 85.0
    senal["accion"] = "HOLD"
    assert ciclo(entrada) is not None
    assert db.consultar_uno("SELECT exit_ts FROM bot_trades")["exit_ts"] is None


def test_ciclo_mueve_stop_y_cierra_por_trailing(bot_prueba):
    """El stop sube con velas nuevas, la misma vela no lo dispara y la caída posterior cierra por TRAILING."""
    ciclo, senal = bot_prueba
    velas = [(101.0, 99.0, 100.0)]
    ciclo(velas)
    senal["accion"] = "HOLD"
    velas = velas + [(128.0, 105.0, 125.0)]
    pos = ciclo(velas)
    assert pos["trailing_activo"] is True and pos["sl"] == 108.0
    assert ciclo(velas)["sl"] == 108.0
    assert db.consultar_uno("SELECT sl FROM bot_trades")["sl"] == 108.0
    assert ciclo(velas + [(120.0, 107.0, 109.0)]) is None
    t = db.consultar_uno("SELECT exit_price, exit_reason, pnl FROM bot_trades")
    assert t["exit_reason"] == "TRAILING" and t["exit_price"] == 108.0 and t["pnl"] > 0


def test_alpaca_guarda_la_cantidad_llenada(monkeypatch):
    """La respuesta inicial con `filled_qty` "0" no deja la posición en cantidad cero."""
    monkeypatch.setattr(executor.time, "sleep", lambda s: None)
    monkeypatch.setattr(executor.alpaca, "orden", lambda i, ttl=5: {"id": i, "status": "filled", "qty": "6", "filled_qty": "6", "filled_avg_price": "154.5"})
    r = executor._resultado_alpaca({"id": "a1", "status": "accepted", "qty": "6", "filled_qty": "0", "filled_avg_price": None}, 6.49, 150.0)
    assert r["qty"] == 6.0 and r["precio"] == 154.5


def test_alpaca_sin_confirmar_usa_cantidad_solicitada(monkeypatch):
    """Si el llenado no se confirma a tiempo se conserva la cantidad enviada."""
    monkeypatch.setattr(executor.time, "sleep", lambda s: None)
    monkeypatch.setattr(executor.alpaca, "orden", lambda i, ttl=5: None)
    r = executor._resultado_alpaca({"id": "a1", "status": "accepted", "qty": "6", "filled_qty": "0"}, 6.49, 150.0)
    assert r["qty"] == 6.0 and r["precio"] == 150.0


@pytest.mark.parametrize("qty, corto, esperado", [(1.3679, False, "1.3679"), (6.0, False, "6"), (6.49, True, "6")])
def test_alpaca_formato_de_cantidad(monkeypatch, qty, corto, esperado):
    """Los cierres de largos conservan fracciones y los cortos se abren con acciones enteras."""
    enviado = {}

    class Resp:
        status_code = 200

        def json(self):
            return {}

    monkeypatch.setattr(executor.alpaca.requests, "post", lambda url, headers, json, timeout: enviado.update(json) or Resp())
    executor.alpaca.enviar_orden_accion("NVDA", "sell", qty, corto)
    assert enviado["qty"] == esperado


@pytest.fixture
def cierre_prueba(tmp_path, monkeypatch):
    """Base temporal con un activo de Alpaca y el envío de órdenes interceptado."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cierre.db"))
    db.inicializar()
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts, training_status) VALUES ('INTC','Intel','accion','alpaca_paper',1,?,'listo')", (db.ahora_iso(),))
    db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,'voting',1,1000,NULL,0,'')", (aid,))
    enviadas = []
    monkeypatch.setattr(runner, "emit", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_explicar_async", lambda *a, **k: None)
    monkeypatch.setattr(runner, "ejecutar_orden", lambda s, v, l, q, p, **k: enviadas.append(v) or {"precio": p, "qty": q, "venue_order_id": None, "modo": v})
    asset = db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,))
    estado = db.consultar_uno("SELECT * FROM bot_strategy_state WHERE asset_id=?", (aid,))

    def trade(venue_order_id):
        """Registra una operación abierta con o sin id de orden del broker."""
        return db.ejecutar("INSERT INTO bot_trades(asset_id, strategy, side, qty, entry_price, sl, tp, entry_ts, venue_order_id) VALUES (?,'voting','short',14,55.5,57,52.6,?,?)",
                           (aid, db.ahora_iso(), venue_order_id))

    return asset, estado, enviadas, trade


def test_apertura_en_paper_se_cierra_en_paper(cierre_prueba):
    """Un corto que Alpaca rechazó y quedó en paper no manda una compra real al cerrarse."""
    asset, estado, enviadas, trade = cierre_prueba
    pos = {"side": "short", "qty": 14.0, "entry_price": 55.5, "sl": 57.0, "tp": 52.6, "trade_id": trade(None), "modo": "paper"}
    runner._cerrar(asset, estado, pos, 57.0, "SL")
    assert enviadas == ["paper"]


def test_apertura_en_alpaca_se_cierra_en_alpaca(cierre_prueba):
    """Una posición abierta en el broker se cierra en el broker."""
    asset, estado, enviadas, trade = cierre_prueba
    pos = {"side": "short", "qty": 14.0, "entry_price": 55.5, "sl": 57.0, "tp": 52.6, "trade_id": trade("orden-1"), "modo": "alpaca_paper"}
    runner._cerrar(asset, estado, pos, 57.0, "SL")
    assert enviadas == ["alpaca_paper"]


@pytest.mark.parametrize("orden_id, esperado", [(None, "paper"), ("orden-1", "alpaca_paper")])
def test_posiciones_anteriores_usan_el_id_de_orden(cierre_prueba, orden_id, esperado):
    """Sin el campo `modo`, el venue de cierre se deduce de si la apertura tuvo id de orden."""
    asset, _, _, trade = cierre_prueba
    pos = {"side": "short", "qty": 14.0, "trade_id": trade(orden_id)}
    assert runner.venue_de_cierre(asset, pos) == esperado
