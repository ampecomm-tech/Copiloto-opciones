"""Verifica cada conexión externa con una llamada mínima y muestra una tabla ✅/❌/⏭.

Uso (desde la raíz del repo): `.venv/bin/python scripts/check_apis.py`
Nunca imprime llaves: solo si funcionó y, si falló, el mensaje de error.
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from core.config import cfg, env  # noqa: E402

OK, FAIL, SKIP = "✅", "❌", "⏭"
resultados: list[tuple[str, str, str]] = []


def _err(e: Exception | str) -> str:
    """Mensaje de error corto, sin llaves."""
    txt = str(e).replace("\n", " ")
    for var in ("TOGETHER_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "FIREBASE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                "BINANCE_SPOT_TESTNET_API_KEY", "BINANCE_SPOT_TESTNET_SECRET", "BINANCE_FUTURES_TESTNET_API_KEY", "BINANCE_FUTURES_TESTNET_SECRET"):
        v = env(var)
        if v:
            txt = txt.replace(v, "***")
    return txt[:160]


def prueba(nombre: str, fn, requiere: tuple[str, ...] = ()) -> None:
    """Ejecuta `fn()`; registra ✅ con su detalle, ❌ con el error o ⏭ si faltan variables."""
    faltan = [v for v in requiere if not env(v)]
    if faltan:
        resultados.append((nombre, SKIP, f"omitido: falta {', '.join(faltan)}"))
        return
    t = time.time()
    try:
        detalle = fn() or "ok"
        resultados.append((nombre, OK, f"{detalle} ({time.time() - t:.1f}s)"))
    except Exception as e:
        resultados.append((nombre, FAIL, _err(e)))


# ---------- pruebas ----------

def gemini() -> str:
    from agents import gemini_client as gem

    r = gem.generar("Responde exactamente: OK", max_tokens=5)
    return f"modelo {gem.modelo()} respondió '{r.strip()[:10]}'"


def together() -> str:
    from agents import together_client as tc

    modelo = cfg("agentes", "modelos", "juez")
    r = tc.chat(modelo, "Responde solo con la palabra OK.", "Hola", max_tokens=300, json_mode=False)
    if not r.strip():
        raise RuntimeError(f"{modelo} respondió vacío")
    return f"{modelo} respondió '{r.strip()[:10]}'"


def xai() -> str:
    r = requests.post("https://api.x.ai/v1/responses", json={"model": cfg("sentimiento", "modelo_grok"), "input": "Responde OK"},
                      headers={"Authorization": f"Bearer {env('XAI_API_KEY')}"}, timeout=30)
    r.raise_for_status()
    return "respondió"


def firebase() -> str:
    base = "https://identitytoolkit.googleapis.com/v1/accounts"
    key = env("FIREBASE_API_KEY")
    email = f"check-{uuid.uuid4().hex[:8]}@copiloto.test"
    r = requests.post(f"{base}:signUp", params={"key": key}, json={"email": email, "password": "Prueba123!", "returnSecureToken": True}, timeout=15)
    if r.status_code != 200:
        raise RuntimeError((r.json().get("error") or {}).get("message", r.text[:100]))
    token = r.json()["idToken"]
    d = requests.post(f"{base}:delete", params={"key": key}, json={"idToken": token}, timeout=15)
    if d.status_code != 200:
        raise RuntimeError("usuario creado pero no se pudo borrar: " + (d.json().get("error") or {}).get("message", ""))
    return "usuario de prueba creado y borrado"


def alpaca() -> str:
    from data_sources import alpaca as alp

    r = requests.get(f"{env('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')}/v2/account", headers=alp._headers(), timeout=15)
    r.raise_for_status()
    a = r.json()
    return f"cuenta {a.get('status')} · equity {float(a.get('equity', 0)):,.0f} USD · mercado {'abierto' if alp.mercado_abierto() else 'cerrado'}"


def binance_spot() -> str:
    import ccxt

    ex = ccxt.binance({"apiKey": env("BINANCE_SPOT_TESTNET_API_KEY"), "secret": env("BINANCE_SPOT_TESTNET_SECRET"), "enableRateLimit": True})
    ex.set_sandbox_mode(True)
    bal = ex.fetch_balance().get("total") or {}
    top = {k: v for k, v in bal.items() if v and float(v) > 0}
    return "saldo " + ", ".join(f"{k} {float(v):,.2f}" for k, v in list(top.items())[:4])


def _firmado(base: str, ruta: str) -> requests.Response:
    """GET firmado HMAC con la llave del testnet de futuros."""
    params = {"timestamp": int(time.time() * 1000), "recvWindow": 5000}
    q = urlencode(params)
    firma = hmac.new(env("BINANCE_FUTURES_TESTNET_SECRET").encode(), q.encode(), hashlib.sha256).hexdigest()
    return requests.get(f"{base}{ruta}?{q}&signature={firma}", headers={"X-MBX-APIKEY": env("BINANCE_FUTURES_TESTNET_API_KEY")}, timeout=15)


def binance_futuros_demo() -> str:
    base = str(cfg("ejecucion", "binance_options_testnet_url")).rstrip("/")
    errores = []
    for ruta, etiqueta in (("/eapi/v1/account", "cuenta de opciones"), ("/fapi/v2/balance", "saldo de futuros USD-M")):
        r = _firmado(base, ruta)
        if r.status_code == 200:
            d = r.json()
            if isinstance(d, list):
                activos = [f"{x.get('asset')} {float(x.get('balance', 0)):,.2f}" for x in d if float(x.get("balance", 0) or 0) > 0][:3]
            else:
                activos = [f"{x.get('asset')} {float(x.get('equity', x.get('marginBalance', 0)) or 0):,.2f}" for x in d.get("asset", [])][:3]
            return f"{etiqueta} en {base}: " + (", ".join(activos) or "sin saldo")
        errores.append(f"{ruta} → {r.status_code} {r.text[:80]}")
    raise RuntimeError("; ".join(errores))


def deribit_testnet() -> str:
    from data_sources import deribit as d

    d._auth()
    n = len(d.instrumentos("BTC"))
    try:
        r = d.resumen_cuenta("BTC")
        return f"autenticado · {n} opciones de BTC · equity {r['equity']} BTC"
    except Exception as e:
        if "forbidden" in str(e).lower():
            return f"autenticado · {n} opciones de BTC · sin scope account:read (revisa los permisos de la API key)"
        raise


def binance_opciones_publico() -> str:
    from data_sources.binance_options import cadena_opciones

    cad, S, _ = cadena_opciones("BTC")
    if cad.empty:
        raise RuntimeError("cadena vacía")
    return f"{len(cad)} contratos de BTC · índice {S:,.0f} USDT"


def yfinance_aapl() -> str:
    from data_sources.yf_options import spot

    return f"AAPL {spot('AAPL'):,.2f} USD"


def discord(canal: str):
    def _f() -> str:
        url = env(f"DISCORD_WEBHOOK_{canal}")
        if not url.startswith("https://discord.com/api/webhooks/"):
            raise RuntimeError("no es una URL de webhook (se necesita https://discord.com/api/webhooks/<id>/<token>, no solo el id)")
        r = requests.post(url, json={"embeds": [{"title": f"Prueba de conexión · {canal.lower()}", "description": "check_apis.py del Copiloto de Opciones", "color": 0x95A5A6}]}, timeout=10)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"{r.status_code} {r.text[:80]}")
        return "mensaje de prueba enviado"
    return _f


def main() -> int:
    """Corre todas las pruebas e imprime la tabla."""
    prueba("Gemini (AI Studio)", gemini, ("GEMINI_API_KEY",))
    prueba("Together AI", together, ("TOGETHER_API_KEY",))
    prueba("xAI Grok", xai, ("XAI_API_KEY",))
    prueba("Firebase Auth", firebase, ("FIREBASE_API_KEY",))
    prueba("Alpaca Paper", alpaca, ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    prueba("Binance Spot Testnet", binance_spot, ("BINANCE_SPOT_TESTNET_API_KEY", "BINANCE_SPOT_TESTNET_SECRET"))
    prueba("Binance Futures/Demo", binance_futuros_demo, ("BINANCE_FUTURES_TESTNET_API_KEY", "BINANCE_FUTURES_TESTNET_SECRET"))
    prueba("Deribit Testnet", deribit_testnet, ("DERIBIT_TESTNET_CLIENT_ID", "DERIBIT_TESTNET_CLIENT_SECRET"))
    prueba("Binance Options (público)", binance_opciones_publico)
    prueba("yfinance", yfinance_aapl)
    for canal in ("MESA", "SENTIMIENTO", "BOT_SENALES", "BOT_OPERACIONES", "ALERTAS", "SISTEMA"):
        prueba(f"Discord #{canal.lower()}", discord(canal), (f"DISCORD_WEBHOOK_{canal}",))
    ancho = max(len(n) for n, _, _ in resultados)
    print()
    for nombre, estado, detalle in resultados:
        print(f"{estado}  {nombre.ljust(ancho)}  {detalle}")
    fallos = sum(1 for _, e, _ in resultados if e == FAIL)
    print(f"\n{len(resultados) - fallos}/{len(resultados)} sin fallos")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
