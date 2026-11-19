"""Tests del respaldo en Gate.io testnet: mapeo de patas, cadena Deribit → Gate.io → paper y cancelación por tiempo (sin red)."""
from datetime import datetime, timedelta, timezone

import pytest

from core import db, execution, positions
from data_sources import gate_options

VENCE = int(datetime(2026, 12, 4, 8, tzinfo=timezone.utc).timestamp())
CONTRATOS = [
    {"name": "BTC_USDT-20261204-58000-C", "expiration_time": VENCE, "is_call": True, "strike_price": "58000", "multiplier": "0.01", "order_price_round": "1", "order_size_min": 1},
    {"name": "BTC_USDT-20261204-58000-P", "expiration_time": VENCE, "is_call": False, "strike_price": "58000", "multiplier": "0.01", "order_price_round": "1", "order_size_min": 1},
]


def _pata(**k):
    """Pata de Binance con valores por defecto."""
    return {"type": "call", "strike": 58000.0, "expiry": "2026-12-04", "side": 1, "qty": 0.52, "price": 1295.4, **k}


@pytest.fixture
def contratos(monkeypatch):
    """Catálogo de contratos de BTC sin consultar la red."""
    monkeypatch.setattr(gate_options, "contratos", lambda activo: CONTRATOS)


def test_mapeo_convierte_a_contratos_con_signo_y_tick(contratos):
    """0.52 BTC son 52 contratos de 0.01; vender da tamaño negativo; el precio se redondea a 1 USDT."""
    compra = gate_options.mapear_pata("BTC", _pata())
    assert compra == {"contract": "BTC_USDT-20261204-58000-C", "size": 52, "price": 1295.0, "multiplicador": 0.01}
    assert gate_options.mapear_pata("BTC", _pata(type="put", side=-1))["size"] == -52


def test_mapeo_rechaza_strikes_inexistentes_y_cantidades_pequenas(contratos):
    """Un strike que no existe o menos de un contrato lanzan ValueError antes de enviar nada."""
    with pytest.raises(ValueError, match="no existe"):
        gate_options.mapear_pata("BTC", _pata(strike=56000.0))
    with pytest.raises(ValueError, match="menos de un contrato"):
        gate_options.mapear_pata("BTC", _pata(qty=0.004))


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Base temporal, eventos capturados y Deribit fallando como con SOL."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "gate.db"))
    db.inicializar()
    eventos = []
    monkeypatch.setattr(execution, "emit", lambda *a, **k: eventos.append(a))
    monkeypatch.setattr(positions, "emit", lambda *a, **k: eventos.append(a))
    monkeypatch.setattr(execution, "venue_cripto", lambda: "deribit_testnet")
    monkeypatch.setattr(execution.deribit, "disponible", lambda: True)

    def deribit_falla(e):
        raise RuntimeError("Deribit testnet solo tiene opciones de BTC, ETH (activo SOL)")

    monkeypatch.setattr(execution, "ordenar_deribit", deribit_falla)
    monkeypatch.setattr(execution.gate_options, "disponible", lambda: True)
    return eventos


ESTRATEGIA = {"nombre": "Bull call spread prueba", "activo": "SOL", "legs": [_pata()], "multiplicador": 1}


def test_si_deribit_falla_va_a_gate_en_libro(entorno, monkeypatch):
    """La orden llega a Gate.io con estado en libro y queda un aviso informativo, no un paso a paper."""
    monkeypatch.setattr(execution.gate_options, "ordenar_estrategia", lambda a, l: {"order_ids": ["11"], "contratos": ["SOL_USDT-X"], "detalle": "1 orden"})
    r = execution.ejecutar(ESTRATEGIA, "cripto")
    assert r["venue"] == "gate_options_testnet" and r["estado"] == "en_libro" and r["order_ids"] == ["11"] and r["enviada_ts"]
    assert "respaldo Gate.io" in entorno[-1][2]


def test_si_gate_tambien_falla_cae_a_paper_con_ambos_motivos(entorno, monkeypatch):
    """Sin contrato en Gate.io, la posición queda en paper y el aviso explica los dos rechazos."""
    def gate_falla(a, l):
        raise ValueError("no existe en Gate.io testnet el contrato")

    monkeypatch.setattr(execution.gate_options, "ordenar_estrategia", gate_falla)
    r = execution.ejecutar(ESTRATEGIA, "cripto")
    assert r["venue"] == "paper" and "Deribit" in r["detalle"] and "Gate.io" in r["detalle"]


def _posicion_gate(minutos_atras: float) -> int:
    """Posición abierta en Gate.io con sus órdenes en libro desde hace `minutos_atras` minutos."""
    enviada = (datetime.now(timezone.utc) - timedelta(minutes=minutos_atras)).isoformat(timespec="seconds")
    return db.ejecutar("INSERT INTO option_positions(user_id, asset, legs_json, entry_ts, entry_cost, status, venue, venue_order_ids_json) VALUES (?,?,?,?,?,'abierta','gate_options_testnet',?)",
                       ("u", "SOL", db.a_json({"legs": []}), db.ahora_iso(), 100.0, db.a_json({"order_ids": ["1", "2"], "estado": "en_libro", "enviada_ts": enviada, "contratos": ["A", "B"]})))


def test_sin_contraparte_a_tiempo_se_cancela_y_pasa_a_paper(entorno, monkeypatch):
    """Pasado el tiempo de espera, se cancelan las órdenes abiertas y la posición pasa a paper con aviso."""
    canceladas = []
    monkeypatch.setattr(gate_options, "estado_ordenes", lambda ids: {"llenas": [], "abiertas": ids, "otras": [], "todas_llenas": False})
    monkeypatch.setattr(gate_options, "cancelar", lambda oid: canceladas.append(oid))
    reciente, vieja = _posicion_gate(2), _posicion_gate(30)
    assert positions.revisar_ordenes_gate() == [vieja]
    assert canceladas == ["1", "2"]
    p = db.consultar_uno("SELECT venue, venue_order_ids_json FROM option_positions WHERE id=?", (vieja,))
    assert p["venue"] == "paper" and db.de_json(p["venue_order_ids_json"])["estado"] == "cancelada_sin_contraparte"
    assert db.consultar_uno("SELECT venue FROM option_positions WHERE id=?", (reciente,))["venue"] == "gate_options_testnet"


def test_ordenes_llenadas_quedan_ejecutadas_en_gate(entorno, monkeypatch):
    """Si alguien llena todas las órdenes, la posición se queda en Gate.io como ejecutada."""
    monkeypatch.setattr(gate_options, "estado_ordenes", lambda ids: {"llenas": ids, "abiertas": [], "otras": [], "todas_llenas": True})
    pid = _posicion_gate(30)
    positions.revisar_ordenes_gate()
    p = db.consultar_uno("SELECT venue, venue_order_ids_json FROM option_positions WHERE id=?", (pid,))
    assert p["venue"] == "gate_options_testnet" and db.de_json(p["venue_order_ids_json"])["estado"] == "ejecutada"
