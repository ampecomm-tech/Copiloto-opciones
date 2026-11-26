"""Tests de las estrategias de swing (velas diarias) y del runner con varias temporalidades (sin red)."""
import numpy as np
import pandas as pd
import pytest

from bot import risk, runner
from bot.strategies import SWING, crear, timeframe_de
from bot.strategies.base import Signal
from bot.strategies.swing import DonchianSwing, EMAPullbackSwing, RSI2Swing
from core import db


def _df(cierres, alto=None, bajo=None, freq="1D"):
    """DataFrame OHLCV a partir de cierres (alto/bajo = cierre ± 0.5 si no se dan)."""
    c = np.asarray(cierres, dtype=float)
    h = np.asarray(alto, dtype=float) if alto is not None else c + 0.5
    l = np.asarray(bajo, dtype=float) if bajo is not None else c - 0.5
    idx = pd.date_range("2025-01-01", periods=len(c), freq=freq, tz="UTC")
    return pd.DataFrame({"open": c, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def test_registro_y_temporalidad():
    """Las tres estrategias de swing existen y operan en diario; las demás en la base."""
    for n in SWING:
        assert crear(n, 1).name == n and timeframe_de(n) == "1d"
    assert timeframe_de("macd_cross") == "15m"


def test_donchian_ruptura_y_salida():
    """Ruptura del máximo de 20 → BUY; del mínimo → SELL; perder el mínimo de 10 → SELL solo cierre."""
    lateral = [100 + (i % 5) for i in range(30)]
    assert DonchianSwing().generate_signal(_df(lateral + [120])).action == "BUY"
    assert DonchianSwing().generate_signal(_df(lateral + [80])).action == "SELL"
    s = DonchianSwing().generate_signal(_df([90] * 15 + [100] * 15 + [99]))   # bajo el mínimo de 10 (99.5), dentro del de 20
    assert s.action == "SELL" and s.details["solo_cierre"] is True
    assert DonchianSwing().generate_signal(_df(lateral + [102])).action == "HOLD"


def test_ema_pullback_compra_el_rebote():
    """Tendencia alcista y una vela que toca la EMA20 y cierra sobre ella → BUY."""
    c = list(np.linspace(100, 200, 80))
    ema20 = pd.Series(c + [200.0]).ewm(span=20, adjust=False).mean().iloc[-1]
    alto, bajo = [x + 0.5 for x in c] + [201.0], [x - 0.5 for x in c] + [ema20 - 1]
    s = EMAPullbackSwing().generate_signal(_df(c + [200.0], alto, bajo))
    assert s.action == "BUY" and s.details["timeframe"] == "1d"
    s = EMAPullbackSwing().generate_signal(_df(list(np.linspace(200, 100, 80))))
    assert s.action == "SELL"


def test_rsi2_compra_caidas_en_tendencia_alcista():
    """Sobre la SMA200 y RSI2 < 10 → BUY; con el cierre ya sobre la SMA5 → SELL solo cierre."""
    c = list(np.linspace(100, 300, 220)) + [290, 280]
    s = RSI2Swing().generate_signal(_df(c))
    assert s.action == "BUY" and s.details["rsi2"] < 10
    s = RSI2Swing().generate_signal(_df(c + [300]))
    assert s.action == "SELL" and s.details.get("solo_cierre") is True
    assert RSI2Swing().generate_signal(_df(c[:100])).details["razon"].startswith("datos insuficientes")


@pytest.fixture
def bot_swing(tmp_path, monkeypatch):
    """Activo paper con una estrategia de swing; velas de 15 min y diarias controladas."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "swing.db"))
    db.inicializar()
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts, training_status) VALUES ('TEST','Test','accion','paper',1,?,'listo')", (db.ahora_iso(),))
    db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,'donchian_swing',1,1000,NULL,0,'')", (aid,))
    for nombre in ("emit", "_explicar_async"):
        monkeypatch.setattr(runner, nombre, lambda *a, **k: None)
    monkeypatch.setattr(runner, "_sentimiento", lambda s: None)
    monkeypatch.setattr(runner.risk.ParametrosRiesgo, "desde_config", classmethod(lambda cls: risk.ParametrosRiesgo(trailing_habilitado=False)))
    llamadas = {"senales": 0}
    senal = {"s": Signal("BUY", 0.9, {"razon": "prueba"})}

    class Falsa:
        timeframe = "1d"

        def generate_signal(self, df):
            llamadas["senales"] += 1
            llamadas["ultimo_df"] = df
            return senal["s"]

    monkeypatch.setattr(runner, "crear", lambda *a: Falsa())
    datos = {"15m": _df([100.0] * 80 + [0.0], freq="15min"), "1d": _df([100.0 + (i % 2) * 20 for i in range(40)] + [0.0])}
    monkeypatch.setattr(runner, "velas", lambda s, v, tf=None, lim=None: datos[tf or "15m"])

    def correr():
        runner.procesar_activo(db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,)))
        return db.de_json(db.consultar_uno("SELECT position_json FROM bot_strategy_state WHERE asset_id=?", (aid,))["position_json"])

    return correr, datos, llamadas, senal


def test_runner_usa_velas_diarias_y_atr_diario(bot_swing):
    """La señal se calcula con las velas diarias, una sola vez por día; el stop usa el ATR diario."""
    correr, datos, llamadas, _ = bot_swing
    pos = correr()
    assert llamadas["senales"] == 1 and len(llamadas["ultimo_df"]) == 40
    assert pos["timeframe"] == "1d" and pos["entry_price"] == 100.0
    assert 100.0 - pos["sl"] > 20      # ATR diario ≈ 20 → stop a ~30, no a 1.5 del ATR de 15 min
    nueva = _df([100.0] * 81 + [0.0], freq="15min")
    datos["15m"] = nueva
    correr()
    assert llamadas["senales"] == 1    # nueva vela de 15 min, mismo día: no hay otra señal


def test_solo_cierre_no_abre_cortos(bot_swing):
    """Una señal SELL de solo cierre sin posición no abre un corto, aunque el venue lo permita."""
    correr, _, _, senal = bot_swing
    senal["s"] = Signal("SELL", 0.55, {"razon": "salir del largo", "solo_cierre": True})
    assert correr() is None


def test_asegurar_estrategias_agrega_las_nuevas(tmp_path, monkeypatch):
    """Los activos existentes reciben las estrategias por defecto que les falten, sin duplicar."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "asegurar.db"))
    db.inicializar()
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts) VALUES ('X','X','cripto','paper',1,?)", (db.ahora_iso(),))
    db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital) VALUES (?,'macd_cross',1,1000)", (aid,))
    n = runner.asegurar_estrategias()
    assert n >= 3 and runner.asegurar_estrategias() == 0
    nombres = {e["strategy"] for e in db.consultar("SELECT strategy FROM bot_strategy_state WHERE asset_id=?", (aid,))}
    assert set(SWING) <= nombres
