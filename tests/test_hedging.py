"""Tests de la cobertura de delta desde la app: sugerencia, venues, valuación y cierre (sin red)."""
import pytest

from bot import executor, runner
from core import db, hedging, positions


@pytest.fixture
def base(tmp_path, monkeypatch):
    """Base temporal con una posición de BTC y otra de NVDA; sin Discord ni red."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cob.db"))
    db.inicializar()
    for m in (hedging, positions, executor, runner):
        monkeypatch.setattr(m, "emit", lambda *a, **k: None)
    precios = {"BTC": 90000.0, "NVDA": 150.0}
    monkeypatch.setattr(hedging, "_precio_actual", lambda a: precios[a])
    ids = {}
    for asset, mult in (("BTC", 1.0), ("NVDA", 100.0)):
        legs = {"legs": [{"type": "call", "strike": 1.0, "expiry": "2026-12-31", "dte": 30, "side": 1, "qty": 1, "iv": 0.5, "price": 1.0}],
                "multiplicador": mult, "nombre": "Long call", "tipo": "long_call", "max_ganancia": 0, "max_perdida": 100, "breakevens": [], "spot_entrada": precios[asset], "expiry": "2026-12-31"}
        ids[asset] = db.ejecutar("INSERT INTO option_positions(user_id, asset, legs_json, entry_ts, entry_cost, status, venue) VALUES ('u',?,?,?,100,'abierta','paper')",
                                 (asset, db.a_json(legs), db.ahora_iso()))
    return ids, precios


def _paper(monkeypatch):
    """Fuerza el destino de las coberturas a paper interno."""
    monkeypatch.setattr(hedging, "destino", lambda a: {"venue": "paper", "instrumento": a, "enteras": a == "NVDA"})


def test_sugerencia_tiene_signo_y_descuenta_lo_cubierto(base, monkeypatch):
    """Delta +0.361 → vender 0.361; con 0.2 ya vendidos, falta vender 0.161."""
    ids, _ = base
    _paper(monkeypatch)
    pos = {"id": ids["BTC"], "asset": "BTC"}
    s = hedging.sugerencia(pos, 0.361, 90000.0)
    assert s["qty"] == pytest.approx(-0.361) and s["lado"] == "vender" and "Vender" in s["texto"]
    hedging.abrir(ids["BTC"], -0.2, "u")
    s = hedging.sugerencia(pos, 0.361, 90000.0)
    assert s["delta_cobertura"] == pytest.approx(-0.2) and s["qty"] == pytest.approx(-0.161)
    assert hedging.sugerencia(pos, -0.5, 90000.0)["lado"] == "comprar"


def test_redondeo_por_venue():
    """Acciones enteras; Deribit en contratos de 10 USD (BTC) y 1 USD (ETH)."""
    assert hedging.redondear(-36.4, {"venue": "alpaca_paper", "enteras": True, "instrumento": "NVDA"}, 150) == -36
    q = hedging.redondear(0.36123, {"venue": "deribit_testnet", "enteras": False, "instrumento": "BTC-PERPETUAL"}, 90000)
    assert q * 90000 == pytest.approx(32510)
    q = hedging.redondear(0.5, {"venue": "deribit_testnet", "enteras": False, "instrumento": "ETH-PERPETUAL"}, 3000.4)
    assert round(q * 3000.4, 6) == 1500


def test_destino_por_activo(monkeypatch):
    """BTC/ETH a Deribit, acciones a Alpaca, otra cripto a paper."""
    from data_sources import alpaca, deribit

    monkeypatch.setattr(deribit, "disponible", lambda: True)
    monkeypatch.setattr(alpaca, "disponible", lambda: True)
    monkeypatch.setattr(hedging, "_forzar_paper", lambda: False)
    assert hedging.destino("BTC")["instrumento"] == "BTC-PERPETUAL"
    assert hedging.destino("NVDA")["venue"] == "alpaca_paper"
    assert hedging.destino("SOL")["venue"] == "paper"
    monkeypatch.setattr(hedging, "_forzar_paper", lambda: True)
    assert hedging.destino("BTC")["venue"] == "paper"


def test_deribit_abre_y_cierra_el_mismo_monto(base, monkeypatch):
    """La cobertura en Deribit guarda el monto en USD y el cierre lo deshace con reduce_only."""
    from data_sources import deribit

    ids, precios = base
    llamadas = []

    def ordenar(moneda, lado, qty_base=None, monto_usd=None, reduce_only=False):
        llamadas.append((lado, qty_base, monto_usd, reduce_only))
        monto = monto_usd if monto_usd is not None else round(abs(qty_base) * 90000 / 10) * 10
        px = 90000.0 if not reduce_only else 91000.0
        return {"order_id": f"o{len(llamadas)}", "instrumento": "BTC-PERPETUAL", "precio": px, "monto_usd": monto, "qty_base": (1 if lado == "buy" else -1) * monto / px}

    monkeypatch.setattr(deribit, "disponible", lambda: True)
    monkeypatch.setattr(deribit, "ordenar_perpetuo", ordenar)
    monkeypatch.setattr(hedging, "_forzar_paper", lambda: False)
    h = hedging.abrir(ids["BTC"], -0.361, "u")
    assert h["venue"] == "deribit_testnet" and h["qty"] == pytest.approx(-0.361)
    assert llamadas[0][0] == "sell"
    precios["BTC"] = 91000.0
    r = hedging.cerrar(h["id"], "u")
    assert llamadas[1] == ("buy", None, 32490.0, True)
    assert r["pnl"] == pytest.approx(-0.361 * 1000)   # vendido a 90k, recomprado a 91k


def test_rechazo_del_venue_no_registra_nada(base, monkeypatch):
    """Si Deribit rechaza, se lanza el error y no queda cobertura en la base."""
    from data_sources import deribit

    ids, _ = base
    monkeypatch.setattr(deribit, "disponible", lambda: True)
    monkeypatch.setattr(hedging, "_forzar_paper", lambda: False)
    monkeypatch.setattr(deribit, "ordenar_perpetuo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not_enough_funds")))
    with pytest.raises(RuntimeError, match="not_enough_funds"):
        hedging.abrir(ids["BTC"], -0.3, "u")
    assert hedging.abiertas(ids["BTC"]) == []


def test_valuar_incluye_la_cobertura(base, monkeypatch):
    """La delta neta y su exposición en dólares suman la cobertura; el P&L de opciones no cambia."""
    ids, _ = base
    _paper(monkeypatch)
    pos = next(p for p in positions.listar(status="abierta") if p["id"] == ids["BTC"])
    v0 = positions.valuar(pos, None, 90000.0, 0.045)
    hedging.abrir(ids["BTC"], -v0["griegas"]["delta"], "u")
    v1 = positions.valuar(pos, None, 91000.0, 0.045)
    assert v1["delta_cobertura"] == pytest.approx(-v0["griegas"]["delta"], abs=1e-4)
    assert abs(v1["delta_neta"]) < abs(v1["delta_opciones"])
    assert v1["pnl_cobertura"] == pytest.approx(v1["delta_cobertura"] * 1000)
    assert v1["pnl_total"] == pytest.approx(v1["pnl"] + v1["pnl_cobertura"])


def test_cerrar_posicion_cierra_sus_coberturas(base, monkeypatch):
    """Al cerrar la posición sus coberturas quedan cerradas con P&L; el historial lo suma."""
    ids, precios = base
    _paper(monkeypatch)
    hedging.abrir(ids["BTC"], -0.1, "u")
    precios["BTC"] = 80000.0
    positions.cerrar_posicion(ids["BTC"], 100.0, "u")
    h = db.consultar_uno("SELECT * FROM option_hedges WHERE position_id=?", (ids["BTC"],))
    assert h["status"] == "cerrada" and h["pnl"] == pytest.approx(1000.0)
    assert positions.historial_copiloto("u")["pnl_coberturas_realizado"] == pytest.approx(1000.0)


@pytest.fixture
def broker(base, monkeypatch):
    """Alpaca simulado con posición neta por símbolo y el bot de NVDA con una estrategia."""
    neto = {"v": 0.0}

    def enviar(symbol, lado, q, corto):
        neto["v"] += q if lado == "buy" else -q
        return {"id": "x", "status": "filled", "qty": str(q), "filled_qty": str(q), "filled_avg_price": "150"}

    monkeypatch.setattr(executor.alpaca, "disponible", lambda: True)
    monkeypatch.setattr(executor.alpaca, "mercado_abierto", lambda: True)
    monkeypatch.setattr(executor.alpaca, "ordenes_abiertas", lambda s: [])
    monkeypatch.setattr(executor.alpaca, "posicion_neta", lambda s: neto["v"])
    monkeypatch.setattr(executor.alpaca, "enviar_orden_accion", enviar)
    monkeypatch.setattr(hedging, "destino", lambda a: {"venue": "alpaca_paper", "instrumento": a, "enteras": True})
    aid = db.ejecutar("INSERT INTO bot_assets(symbol, display_name, market_type, venue, active, added_ts, training_status) VALUES ('NVDA','NVIDIA','accion','alpaca_paper',1,?,'listo')", (db.ahora_iso(),))
    pos = {"side": "long", "qty": 5.0, "entry_price": 150.0, "sl": 140.0, "tp": 170.0, "trade_id": 1, "modo": "alpaca_paper"}
    db.ejecutar("INSERT INTO bot_strategy_state(asset_id, strategy, active, capital, position_json, daily_pnl, daily_date) VALUES (?,'macd_cross',1,1000,?,0,'')", (aid, db.a_json(pos)))
    neto["v"] = 5.0
    return base[0], aid, neto


def test_alpaca_cobertura_se_netea_con_el_bot(broker):
    """Vender 3 de cobertura con el bot largo 5 deja la cuenta en 2; si el bot cierra, la cuenta queda en −3 (la cobertura)."""
    ids, aid, neto = broker
    hedging.abrir(ids["NVDA"], -3, "u")
    assert neto["v"] == pytest.approx(2.0)
    assert executor.neta_de_estrategias(aid, "alpaca_paper") == pytest.approx(2.0)
    objetivo = executor.neta_de_estrategias(aid, "alpaca_paper", excluir="macd_cross")
    assert objetivo == pytest.approx(-3.0)
    asset = db.consultar_uno("SELECT * FROM bot_assets WHERE id=?", (aid,))
    executor.ejecutar_neta_alpaca(asset, "macd_cross", objetivo, 5.0, "sell", 150.0)
    assert neto["v"] == pytest.approx(-3.0)


def test_alpaca_cerrar_cobertura_respeta_al_bot(broker):
    """Cerrar la cobertura devuelve la cuenta a la posición del bot."""
    ids, _, neto = broker
    h = hedging.abrir(ids["NVDA"], -3, "u")
    hedging.cerrar(h["id"], "u")
    assert neto["v"] == pytest.approx(5.0)
