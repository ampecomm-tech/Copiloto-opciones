"""Genera el informe técnico en PDF a partir de las tablas y gráficas de `docs/resultados/`.

Antes hay que exportar los resultados con `scripts/exportar_resultados.py`. Los datos de
portada (integrantes, materia, enlaces) se leen de `docs/portada.yaml`.

Uso:
    python scripts/generar_informe.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "docs" / "resultados"
SALIDA = RAIZ / "docs" / "Informe_Tecnico_Copiloto_Opciones.pdf"
AZUL, TINTA, GRIS_T, BORDE, FONDO = colors.HexColor("#1d4ed8"), colors.HexColor("#0f172a"), colors.HexColor("#475569"), colors.HexColor("#cbd5e1"), colors.HexColor("#eff6ff")

# --- Tipografía con soporte de letras griegas (DejaVu viene incluida con matplotlib) ---------------------
_FUENTES = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_FUENTES / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_FUENTES / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(_FUENTES / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(_FUENTES / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

E = {
    "cuerpo": ParagraphStyle("cuerpo", fontName="DV", fontSize=9.2, leading=13.4, alignment=TA_JUSTIFY, textColor=TINTA, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=AZUL, spaceBefore=4, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11, leading=14, textColor=TINTA, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "formula": ParagraphStyle("formula", fontName="DV", fontSize=9.2, leading=14, alignment=TA_CENTER, textColor=TINTA, spaceBefore=2, spaceAfter=6,
                              backColor=FONDO, borderPadding=5),
    "pie": ParagraphStyle("pie", fontName="DV-I", fontSize=7.8, leading=10, alignment=TA_CENTER, textColor=GRIS_T, spaceAfter=8),
    "celda": ParagraphStyle("celda", fontName="DV", fontSize=7.4, leading=9.2, textColor=TINTA),
    "celda_b": ParagraphStyle("celda_b", fontName="DV-B", fontSize=6.8, leading=8.6, textColor=colors.white),
    "lista": ParagraphStyle("lista", fontName="DV", fontSize=9.2, leading=13, textColor=TINTA),
}


def _amp(texto: str) -> str:
    """Escapa los "&" sueltos (como en P&L) para que ReportLab no los lea como entidades."""
    import re

    return re.sub(r"&(?!#?\w+;)", "&amp;", str(texto))


def P(texto: str, estilo: str = "cuerpo") -> Paragraph:
    """Párrafo con el estilo indicado."""
    return Paragraph(_amp(texto), E[estilo])


def lista(items: list[str]) -> ListFlowable:
    """Lista con viñetas."""
    return ListFlowable([ListItem(P(t, "lista"), leftIndent=12) for t in items], bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8, spaceAfter=6)


def tabla(filas: list[list[Any]], anchos: list[float] | None = None, resaltar: list[int] | None = None) -> Table:
    """Tabla con encabezado azul, filas alternadas y texto ajustable."""
    datos = [[Paragraph(_amp(c), E["celda_b"] if i == 0 else E["celda"]) for c in fila] for i, fila in enumerate(filas)]
    t = Table(datos, colWidths=anchos, repeatRows=1, hAlign="CENTER")
    estilo = [("BACKGROUND", (0, 0), (-1, 0), AZUL), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, BORDE),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(filas)):
        if i % 2 == 0:
            estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    for i in resaltar or []:
        estilo.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#dbeafe")))
    t.setStyle(TableStyle(estilo))
    return t


def figura(nombre: str, ancho_cm: float, pie: str) -> KeepTogether:
    """Imagen escalada al ancho indicado con su pie de figura."""
    ruta = RES / nombre
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(ruta)).getSize()
    ancho = ancho_cm * cm
    return KeepTogether([Image(str(ruta), width=ancho, height=ancho * h / w), P(pie, "pie")])


def f(x: Any, d: int = 3) -> str:
    """Formatea números con `d` decimales y deja el texto como está."""
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def diagrama_arquitectura(destino: Path) -> None:
    """Dibuja el diagrama de arquitectura del sistema (app, bot, datos, IA y ejecución)."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def caja(x: float, y: float, w: float, h: float, titulo: str, texto: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#334155", lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, titulo, ha="center", va="top", fontsize=8.6, fontweight="bold", color="#0f172a")
        ax.text(x + w / 2, y + h - 5.6, texto, ha="center", va="top", fontsize=6.7, color="#1e293b", linespacing=1.35)

    def flecha(x1: float, y1: float, x2: float, y2: float, txt: str = "") -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=0.9))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.8, txt, fontsize=6.2, color="#475569", ha="center")

    ax.add_patch(FancyBboxPatch((22, 14.5), 50, 45, boxstyle="round,pad=0.5,rounding_size=1.5", fc="#f8fafc", ec="#94a3b8", lw=1, ls="--"))
    ax.text(47, 58.3, "VM Google Cloud · e2-medium · Ubuntu 22.04 · northamerica-south1 (systemd 24/7)", ha="center", fontsize=7.5, color="#475569", fontweight="bold")
    caja(1, 40, 18, 15, "Usuario / clase", "Navegador\nURL pública :8501\nFirebase Auth\n(registro y login)", "#e0e7ff")
    caja(24, 36, 22, 19.5, "App Streamlit", "1 Noticias y sentimiento\n2 Copiloto de opciones\n3 Mis posiciones\n4 Bot de trading\n5 Centro de control", "#dbeafe")
    caja(48, 36, 22, 19.5, "Bot runner (24/7)", "Velas 15 min por activo\n5 estrategias + RF\nSL/TP ATR + trailing\nkill-switch diario\nresumen diario", "#dcfce7")
    caja(24, 16, 22, 17, "Núcleo cuantitativo", "Black-Scholes y griegas\nIV (Brent + Newton)\nMonte Carlo GBM\nSemáforo · Markowitz\nRF · KNN · Diebold-Mariano", "#fef9c3")
    caja(48, 16, 22, 17, "SQLite WAL + bus emit()", "events · recommendations\noption_positions\nbot_trades · bot_signals\nmodel_runs\nlogs/trading_bot.log", "#fae8ff")
    caja(76, 42, 32, 15, "IA", "Gemini (AI Studio): analista, noticias,\nexplicaciones, resumen, respaldo\nTogether AI: sentimiento, abogado\ndel diablo y juez · Grok opcional", "#ffedd5")
    caja(76, 23, 32, 15, "Datos de mercado", "Binance European Options (eapi)\nBinance spot mainnet (señales)\nAlpaca options data · yfinance\nRSS + Fear & Greed", "#e2e8f0")
    caja(76, 2, 32, 17, "Ejecución (sin dinero real)", "Deribit testnet: opciones BTC/ETH\nGate.io testnet: respaldo de opciones\nAlpaca Paper (órdenes netas) · Binance\nspot testnet · paper de respaldo", "#fee2e2")
    caja(24, 1, 46, 10, "Discord (6 canales) + Centro de control", "#mesa-agentes · #sentimiento-x · #bot-señales · #bot-operaciones · #alertas · #sistema", "#ccfbf1")
    flecha(19, 47, 24, 47, "HTTPS")
    flecha(46, 26, 48, 26)
    flecha(35, 36, 35, 33)
    flecha(59, 36, 59, 33)
    flecha(70, 49, 76, 49, "prompts JSON")
    flecha(70, 30, 76, 30, "precios")
    flecha(70, 20, 76, 12, "órdenes")
    flecha(59, 16, 59, 11, "emit()")
    fig.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _pagina(canvas: Any, doc: Any, titulo: str) -> None:
    """Encabezado y pie de las páginas interiores."""
    canvas.saveState()
    canvas.setStrokeColor(BORDE)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, letter[1] - 1.5 * cm, letter[0] - 2 * cm, letter[1] - 1.5 * cm)
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(GRIS_T)
    canvas.drawString(2 * cm, letter[1] - 1.3 * cm, titulo)
    canvas.drawRightString(letter[0] - 2 * cm, letter[1] - 1.3 * cm, "Informe técnico")
    canvas.drawCentredString(letter[0] / 2, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


def _portada(canvas: Any, doc: Any, datos: dict[str, Any]) -> None:
    """Portada con bloque de color, título, integrantes y enlaces."""
    canvas.saveState()
    W, H = letter
    canvas.setFillColor(colors.HexColor("#0f172a"))
    canvas.rect(0, H * 0.45, W, H * 0.55, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#2563eb"))
    canvas.rect(0, H * 0.45, W, 0.25 * cm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#93c5fd"))
    canvas.setFont("DV-B", 9)
    canvas.drawString(2.2 * cm, H - 3 * cm, "PROYECTO FINAL INTEGRADOR · INFORME TÉCNICO")
    canvas.setFillColor(colors.white)
    canvas.setFont("DV-B", 30)
    canvas.drawString(2.2 * cm, H - 6 * cm, datos["titulo"])
    canvas.setFont("DV", 12.5)
    y = H - 7.4 * cm
    for linea in _partir(datos["subtitulo"], 70):
        canvas.drawString(2.2 * cm, y, linea)
        y -= 0.6 * cm
    canvas.setFont("DV", 10)
    canvas.setFillColor(colors.HexColor("#cbd5e1"))
    y -= 0.6 * cm
    for linea in _partir(datos["materia"], 85):
        canvas.drawString(2.2 * cm, y, linea)
        y -= 0.5 * cm
    if datos.get("institucion"):
        canvas.drawString(2.2 * cm, y, datos["institucion"])
    canvas.setFillColor(TINTA)
    canvas.setFont("DV-B", 10)
    y = H * 0.45 - 2 * cm
    canvas.drawString(2.2 * cm, y, "Autor" if len(datos.get("integrantes", [])) == 1 else "Integrantes")
    canvas.setFont("DV", 10)
    for nombre in datos.get("integrantes", []):
        y -= 0.55 * cm
        canvas.drawString(2.2 * cm, y, nombre)
    y -= 1.2 * cm
    for etiqueta, valor in (("Modalidad", datos.get("modalidad", "")), ("Fecha de entrega y defensa", datos["fecha"]), ("Repositorio", datos["repositorio"]), ("Aplicación en vivo", datos["app"])):
        if not valor:
            continue
        canvas.setFont("DV-B", 9)
        canvas.drawString(2.2 * cm, y, etiqueta)
        canvas.setFont("DV", 9)
        canvas.drawString(8.4 * cm, y, valor)
        y -= 0.55 * cm
    canvas.setFont("DV-I", 7.5)
    canvas.setFillColor(GRIS_T)
    canvas.drawString(2.2 * cm, 1.6 * cm, "Sistema educativo: todas las operaciones se hicieron en testnet y paper trading, nunca con dinero real.")
    canvas.restoreState()


def _partir(texto: str, n: int) -> list[str]:
    """Parte un texto en líneas de hasta `n` caracteres sin cortar palabras."""
    lineas, actual = [], ""
    for palabra in texto.split():
        if len(actual) + len(palabra) + 1 > n:
            lineas.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    return lineas + ([actual] if actual else [])


def _texto_markowitz(corr: pd.DataFrame, mk: pd.DataFrame) -> str:
    """Comentario de la frontera de Markowitz calculado a partir de las tablas, para que nunca contradiga los números."""
    def pesos(nombre: str) -> str:
        fila = mk[mk["cartera"] == nombre].iloc[0]
        w = sorted(((c.replace("w_", ""), float(fila[c])) for c in mk.columns if c.startswith("w_")), key=lambda t: -t[1])
        return ", ".join(f"{a} {p:.0%}" for a, p in w if p >= 0.05)
    ms, mv = mk[mk["cartera"] == "Máximo Sharpe"].iloc[0], mk[mk["cartera"] == "Mínima volatilidad"].iloc[0]
    btc_sol = float(corr.loc["BTC", "SOL"]) if "BTC" in corr.index and "SOL" in corr.columns else float("nan")
    return (f"BTC y SOL tienen una correlación de {btc_sol:.2f}, así que repartir capital del bot entre ambos diversifica poco. "
            f"La cartera de máximo Sharpe ({float(ms['sharpe']):.2f}) queda en {pesos('Máximo Sharpe')}, con volatilidad anual de {float(ms['vol']):.0%}: "
            "es una cartera concentrada que depende del último año y no conviene usar sin restricciones de peso. "
            f"La de mínima volatilidad ({float(mv['vol']):.0%} anual) queda en {pesos('Mínima volatilidad')}. "
            "La frontera es una referencia para repartir el capital del bot entre activos.")


def contenido(datos: dict[str, Any]) -> list[Any]:
    """Arma todas las secciones del informe con los resultados exportados."""
    hp = pd.read_csv(RES / "hiperparametros_rf.csv")
    met = pd.read_csv(RES / "metricas_modelos.csv")
    dm = pd.read_csv(RES / "diebold_mariano.csv")
    viv = pd.read_csv(RES / "confusion_en_vivo.csv")
    des = pd.read_csv(RES / "desempeno_bot.csv")
    bs = pd.read_csv(RES / "bs_vs_binance_resumen.csv").iloc[0]
    corr = pd.read_csv(RES / "correlacion_pearson.csv", index_col=0)
    mk = pd.read_csv(RES / "markowitz_carteras.csv")
    recs = pd.read_csv(RES / "recomendaciones_copiloto.csv")
    ops = pd.read_csv(RES / "posiciones_opciones.csv")
    sis = json.loads((RES / "resumen_sistema.json").read_text(encoding="utf-8"))
    rb = pd.read_csv(RES / "riesgo_bot.csv")
    cav = pd.read_csv(RES / "caviar.csv")
    rc = pd.read_csv(RES / "riesgo_copiloto.csv")
    activos = list(dict.fromkeys(met["activo"]))
    rf = met[met["modelo"] == "Random Forest"].set_index("activo")
    knn = met[met["modelo"] == "KNN (base)"].set_index("activo")
    testnet = des[des["venue"] == "binance_spot_testnet"]
    pnl_testnet, cap_testnet = testnet["pnl_usd"].sum(), 1000 * len(testnet)
    rf_gana = dm[dm["resultado"] == "RF mejor"]
    s: list[Any] = []

    # 1. Resumen ejecutivo
    s += [P("1. Resumen ejecutivo", "h1"),
          P("El <b>Copiloto de Opciones con IA</b> es una aplicación web con un solo inicio de sesión donde el usuario se informa sobre un activo, "
            "formula una opinión de mercado y recibe la estrategia de opciones que mejor encaja con ella. Una mesa de agentes de IA de distintos "
            "proveedores debate cada recomendación, y un semáforo determinista de viabilidad decide si conviene entrar. Al confirmar, la orden se "
            "envía a un venue de prueba: el testnet de Deribit para BTC y ETH, con respaldo en el testnet de Gate.io para los demás subyacentes, y Alpaca Paper para acciones y ETFs. "
            "El proyecto se desarrolló de forma individual."),
          P("En paralelo, un bot de trading corre las 24 horas en una máquina virtual de Google Cloud. El bot opera cualquier activo que el usuario "
            "busque por nombre, con cinco estrategias por activo: un Random Forest entrenado automáticamente, tres estrategias técnicas y una votación. "
            "Todo el razonamiento interno se publica en seis canales de Discord y en la pestaña Centro de control."),
          P("<b>Resultados principales</b>"),
          lista([f"Nuestro Black-Scholes reproduce el mark price de Binance Options con un error mediano de <b>{bs['err_precio_pct_mediano']:.4f} %</b> en {int(bs['n'])} contratos de BTC, "
                 f"y la delta coincide con correlación {bs['corr_delta']:.4f}.",
                 f"El Random Forest se entrenó con 1,500 velas de 15 minutos por activo y una simulación de 27 combinaciones de hiperparámetros. Su F1 en prueba va de "
                 f"{rf['f1'].min():.3f} a {rf['f1'].max():.3f}, cerca del azar, como se espera en un horizonte de una hora.",
                 f"La prueba de Diebold-Mariano con corrección HLN muestra que el RF supera de forma significativa a EMA50 + Chandelier en los {len(rf_gana)} activos, "
                 "y que no hay diferencia significativa frente al KNN de referencia.",
                 f"En el Binance spot testnet el bot cerró {int(testnet['operaciones_validas'].sum())} operaciones con un P&L total de {pnl_testnet:+.2f} USD "
                 f"sobre {cap_testnet:,.0f} USD de capital virtual, un ROI de {pnl_testnet / cap_testnet * 100:+.2f} %.",
                 "Riesgo en las dos piezas: el copiloto reporta VaR y CVaR al 95 % y al 99 % de cada estrategia a partir del Monte Carlo, y el bot mide VaR y CVaR "
                 f"históricos y normales de cada activo y de sus operaciones, además de un VaR dinámico CAViaR que pasa la prueba de Kupiec en "
                 f"{', '.join(cav[cav['kupiec_p'] >= 0.05]['activo'])}.",
                 f"El sistema registró {sis['senales']:,} señales, {sis['operaciones_bot']} operaciones del bot, {sis['recomendaciones']} recomendaciones del copiloto y "
                 f"{sum(sis['eventos_por_canal'].values()):,} eventos entre el {sis['senales_desde'][:10]} y el {sis['senales_hasta'][:10]}."]),
          P("El valor del proyecto está en la integración: valuación de derivados, simulación, aprendizaje automático validado estadísticamente, agentes de IA "
            "y ejecución en vivo sobre infraestructura en la nube, con trazabilidad completa de cada decisión.")]

    # 2. Objetivo y alcance
    s += [PageBreak(), P("2. Objetivo y alcance", "h1"),
          P("El proyecto combina tres opciones del documento del curso y toma elementos de una cuarta. La tabla resume cómo se cubre cada requisito y dónde está en el código."),
          tabla([["Opción del curso", "Requisito", "Implementación", "Archivo principal"],
                 ["6. App financiera con IA", "Núcleo cuantitativo", "Black-Scholes y griegas, Markowitz y Pearson, RF y KNN con validación en pantalla, panel conectado al bot de la VM", "quant/, validation/, app/pages/4_bot.py"],
                 ["6. App financiera con IA", "Firebase Authentication", "Registro e inicio de sesión con email y contraseña vía Identity Toolkit", "app/auth.py"],
                 ["6. App financiera con IA", "Motor de IA en tiempo real", "Gemini, Together AI y Grok opcional", "agents/"],
                 ["6. App financiera con IA", "URL pública", "Streamlit en la VM, puerto 8501", "deploy/copiloto-app.service"],
                 ["7. Black-Scholes con agentes", "Precio teórico y griegas por strike", "Fórmulas cerradas, IV por Brent y Newton", "quant/black_scholes.py"],
                 ["7. Black-Scholes con agentes", "Agente sobre la cadena de opciones", "Mesa de cuatro agentes: IV vs HV, sentimiento, riesgos y veredicto", "agents/mesa.py, agents/roles.py"],
                 ["7. Black-Scholes con agentes", "Alertas de cobertura", "Delta desviada, 50 % de ganancia máxima, vencimiento próximo", "core/positions.py"],
                 ["3. Bot en VM", "Datos históricos y en vivo", "Binance mainnet, yfinance, Alpaca y Gemini", "data_sources/"],
                 ["3. Bot en VM", "Buy / Hold / Sell con modelo", "Random Forest de 20 variables técnicas y sentimiento", "bot/strategies/random_forest.py"],
                 ["3. Bot en VM", "Simulación de hiperparámetros y DM", "Grid de 27 combinaciones, métricas y Diebold-Mariano HLN", "validation/hyperparams.py, validation/diebold_mariano.py"],
                 ["3. Bot en VM", "Gestión de riesgo", "SL y TP por ATR, trailing stop, tamaño por volatilidad, kill-switch", "bot/risk.py"],
                 ["3 y 7. Métricas de riesgo", "Riesgo cuantitativo", "VaR y CVaR históricos, normales y de Monte Carlo; CAViaR con Kupiec y Christoffersen", "quant/riesgo.py, quant/caviar.py"],
                 ["3. Bot en VM", "Testnet y ROI simulado", "Binance spot testnet, Alpaca Paper, matriz de confusión en vivo", "bot/executor.py"],
                 ["3. Bot en VM", "VM 24/7 con .env y log", "e2-medium, systemd, .env y logs/trading_bot.log", "deploy/"],
                 ["4. Agentes multirol", "Analista, sentimiento y gestor de riesgo", "Roles con JSON estricto y modelos de distintas empresas", "agents/"]],
                [3.2 * cm, 3.4 * cm, 6.2 * cm, 4.6 * cm]),
          P("Restricción de seguridad del proyecto: nunca se opera con dinero real. Toda la ejecución ocurre en testnets y cuentas paper.", "pie")]

    # 3. Arquitectura
    s += [PageBreak(), P("3. Arquitectura del sistema", "h1"),
          figura("arquitectura.png", 16.5, "Figura 1. Componentes del sistema y flujo de datos."),
          P("La aplicación Streamlit y el bot son dos procesos systemd independientes en la misma VM. Comparten una base SQLite en modo WAL, que permite "
            "lecturas y escrituras concurrentes sin bloqueos. La base es la fuente de verdad de posiciones y operaciones: al arrancar, el bot la reconcilia "
            "con los venues, porque el testnet de Binance se reinicia periódicamente."),
          P("La pieza central es el bus de eventos <font face='DVM'>emit(canal, tipo, título, payload)</font>. Cada evento se guarda en la tabla "
            "<i>events</i>, se publica como embed en el canal de Discord correspondiente y se escribe en el log. Si Discord falla, el evento igual queda en la "
            "base, así que una API externa caída nunca detiene la app ni el bot."),
          P("Reparto de la IA", "h2"),
          tabla([["Proveedor", "Modelo", "Uso"],
                 ["Google Gemini (AI Studio)", "gemini-3.1-flash-lite", "Analista de mercado de la mesa, etiquetas y resumen de noticias, opinión estructurada a partir de texto libre, explicación de cada operación del bot, modo educativo, resumen diario y respaldo de la mesa"],
                 ["Together AI", "DeepSeek-V4.1-Flash", "Abogado del diablo"],
                 ["Together AI", "Muse-Glimmer-30B", "Agente de sentimiento"],
                 ["Together AI", "gpt-oss-120b", "Gestor de riesgo y juez"],
                 ["xAI Grok", "grok-4-fast con x_search", "Sentimiento de X, opcional; sin llave se usa Gemini"]],
                [3.6 * cm, 3.6 * cm, 10.2 * cm])]

    # 4. Datos
    s += [PageBreak(), P("4. Datos y preparación", "h1"),
          tabla([["Fuente", "Contenido", "Uso"],
                 ["Binance European Options (eapi)", "Cadena de BTC, ETH, BNB, SOL, XRP y DOGE con bid, ask, mark price, IV y griegas publicadas", "Copiloto cripto y validación de Black-Scholes"],
                 ["Binance spot mainnet", "Velas OHLCV públicas, sin llave", "Señales del bot cripto"],
                 ["Alpaca market data", "Contratos, snapshots de opciones y reloj de mercado", "Copiloto de acciones y horarios"],
                 ["yfinance", "Velas de acciones, materias primas, noticias y fechas de earnings", "Bot de acciones, respaldo de opciones, Markowitz"],
                 ["RSS y Fear & Greed", "CoinDesk, Cointelegraph y alternative.me", "Noticias y sentimiento cripto"],
                 ["Deribit testnet", "Instrumentos, índice y órdenes", "Ejecución de opciones BTC y ETH"]],
                [4.2 * cm, 7.4 * cm, 5.8 * cm]),
          P("Variables del modelo", "h2"),
          P("Para cada vela de 15 minutos se construyen 20 variables: distancia relativa del precio a las EMA de 9, 21, 50 y 200 periodos; RSI de 14 y 7; "
            "MACD y su histograma normalizados por el precio; ATR relativo; posición y ancho de las bandas de Bollinger; retornos rezagados de 1, 2, 3, 5 y 10 velas; "
            "volumen relativo a su media de 20; rango relativo de la vela; volatilidad de 20 periodos; y el último puntaje de sentimiento del activo."),
          P("La etiqueta es binaria: 1 si el cierre sube dentro de 4 velas, es decir una hora. Se eliminan las filas con valores faltantes por el calentamiento "
            "de los indicadores, y se reemplazan los infinitos. La partición es temporal, 70 % para entrenar y 30 % para probar, sin barajar, para no usar "
            "información del futuro."),
          P("Temporalidad", "h2"),
          P("Se eligieron velas de 15 minutos. Dan unas 96 observaciones diarias por activo, suficientes para entrenar con 1,500 velas, que cubren unos 15 días, "
            "y permiten que el bot opere varias veces al día durante la demostración. En 5 minutos el ruido de microestructura domina, y en velas diarias el "
            "bot casi no generaría operaciones en el periodo del proyecto.")]

    # 5. Metodología
    s += [PageBreak(), P("5. Metodología cuantitativa", "h1"),
          P("5.1 Black-Scholes y letras griegas", "h2"),
          P("Para una opción europea sobre un subyacente S, con strike K, tiempo T en años, tasa r y volatilidad σ:"),
          P("d<sub>1</sub> = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)     d<sub>2</sub> = d<sub>1</sub> − σ·√T", "formula"),
          P("C = S·N(d<sub>1</sub>) − K·e<super>−rT</super>·N(d<sub>2</sub>)     P = K·e<super>−rT</super>·N(−d<sub>2</sub>) − S·N(−d<sub>1</sub>)", "formula"),
          P("Las griegas se calculan en forma cerrada: Δ = N(d<sub>1</sub>) para calls y N(d<sub>1</sub>) − 1 para puts; Γ = φ(d<sub>1</sub>) / (S·σ·√T); "
            "vega = S·φ(d<sub>1</sub>)·√T, expresada por cada punto porcentual de volatilidad; y Θ por día calendario. La volatilidad implícita se obtiene "
            "con el método de Brent sobre [0.0001, 5] y se pule con dos pasos de Newton-Raphson usando la vega."),
          P("5.2 Selección de estrategias y Monte Carlo", "h2"),
          P("Según la dirección de la opinión, el selector genera candidatas: long call o put, spreads verticales, straddle, strangle, iron condor y butterfly. "
            "Prueba strikes y vencimientos entre 0.7 y 2.5 veces el horizonte, descarta primas menores a 0.15 % del spot y exige wings simétricas. "
            "Cada candidata usa precios reales de ejecución: el ask al comprar y el bid al vender."),
          P("S<sub>T</sub> = S<sub>0</sub>·exp[ (μ − σ²/2)·T + σ·√T·Z ],   Z ~ N(0, 1),   10,000 trayectorias", "formula"),
          P("La deriva μ se toma de la opinión del usuario, de modo que E[S<sub>T</sub>] = S<sub>0</sub>·(1 ± magnitud). Para la opinión volátil se mezclan "
            "por mitades los escenarios alcista y bajista. Se reporta la probabilidad de ganancia, el valor esperado y los percentiles 5 y 95 del P&L, junto con un "
            "escenario neutral de referencia. Las candidatas se ordenan y pasan las tres mejores a la mesa."),
          P("5.3 Semáforo de viabilidad", "h2"),
          tabla([["Regla", "Umbral", "Peso"],
                 ["Valor esperado Monte Carlo", "> 0", "0.20"],
                 ["Probabilidad de ganancia", "≥ 30 % para amarillo y ≥ 45 % para verde", "0.10 + 0.10"],
                 ["Relación ganancia máxima / pérdida máxima", "≥ 0.5", "0.10"],
                 ["IV / HV", "≤ 1.4 al comprar volatilidad; ≥ 0.8 al venderla", "0.15"],
                 ["Liquidez", "Spread relativo ≤ 15 % y OI mínimo", "0.10"],
                 ["Theta diaria", "≤ 3 % del presupuesto por día", "0.10"],
                 ["Sentimiento", "No contradice la dirección por más de 0.3", "0.05"],
                 ["Pérdida máxima", "≤ presupuesto", "0.10"]],
                [6.5 * cm, 7.4 * cm, 3.5 * cm]),
          P("Verde exige un puntaje de al menos 0.8 y que pasen las cuatro reglas estructurales: valor esperado, IV/HV, liquidez y presupuesto. Rojo aparece si falla "
            "el presupuesto o la probabilidad mínima, o si el puntaje es menor que 0.5. En cualquier otro caso es amarillo, con una alternativa sugerida."),
          P("5.4 Mesa de agentes", "h2"),
          P("El analista de mercado, el agente de sentimiento y el abogado del diablo responden en paralelo, en una sola ronda. Después, el juez lee las tres opiniones "
            "y las reglas del semáforo. Todos responden en JSON estricto, validado con pydantic, con un reintento si no se puede interpretar. Los cálculos los hace Python: "
            "los agentes solo interpretan. El juez nunca puede dar un veredicto mejor que el semáforo. Si Together AI falla, Gemini toma su lugar; si todos fallan, "
            "decide solo el semáforo."),
          P("5.5 Bot de trading y gestión de riesgo", "h2"),
          P("El Random Forest compra si la probabilidad de subida supera 0.58 y vende si es menor que 0.42. Las estrategias técnicas son EMA50 con Chandelier Exit, "
            "reversión por RSI en 30 y 70, y cruce de MACD. La votación suma las señales ponderadas por su confianza. Cada combinación de activo y estrategia tiene "
            "1,000 USD de capital virtual propio."),
          P("q = (riesgo% · capital) / (k<sub>SL</sub> · ATR<sub>14</sub>),   SL = entrada ∓ 1.5·ATR,   TP = entrada ± 3·ATR", "formula"),
          P("El tamaño se fija para que tocar el stop cueste el 2 % del capital, sin apalancamiento. Desde el 18 de noviembre de 2026 el stop es móvil: cuando la "
            "ganancia alcanza 1 ATR, el stop sigue al mejor precio a 2 ATR de distancia y deja de aplicar el take profit fijo. Un kill-switch detiene la "
            "estrategia si la pérdida del día supera el 5 % del capital."),
          P("5.6 Métricas y prueba de Diebold-Mariano", "h2"),
          P("Las métricas de regresión comparan la señal del modelo, P(sube) − 0.5, contra el retorno futuro normalizado y recortado a [−0.5, 0.5]: RMSE, MSE, MAE, MAPE, "
            "SMAPE y Variance Score. Las de clasificación son accuracy, precision, recall, F1 y la matriz de confusión."),
          P("d<sub>t</sub> = e<sub>1,t</sub>² − e<sub>2,t</sub>²     DM = d̄ / √( [γ<sub>0</sub> + 2·Σ<sub>k=1</sub><super>h−1</super> γ<sub>k</sub>] / n )", "formula"),
          P("DM<sub>HLN</sub> = DM · √( [n + 1 − 2h + h(h − 1)/n] / n ),   p-valor con t de Student y n − 1 grados de libertad", "formula"),
          P("La hipótesis nula es igual precisión predictiva. Con un horizonte de h = 4 velas los errores se traslapan, por eso la varianza de largo plazo incluye "
            "autocovarianzas hasta h − 1 y se aplica la corrección de Harvey, Leybourne y Newbold para muestras finitas. Un estadístico negativo con p &lt; 0.05 significa que el "
            "modelo 1, el RF, tiene menor pérdida."),
          P("Configuración de las pruebas de Diebold-Mariano usadas en el proyecto:"),
          tabla([["Parámetro", "Random Forest contra modelos y estrategias", "CAViaR contra VaR histórico"],
                 ["Modelo 1 / modelo 2", "Random Forest / KNN (k = 15), EMA50 + Chandelier, reversión RSI y cruce MACD", "CAViaR SAV / VaR histórico móvil de 300 velas"],
                 ["Qué se pronostica", "Dirección del cierre dentro de 4 velas (una hora)", "Cuantil del 5 % del retorno de la siguiente vela"],
                 ["Error o pérdida", "Error e<sub>t</sub> = retorno normalizado − (P(sube) − 0.5); pérdida cuadrática e²", "Pérdida de cuantil ρ<sub>θ</sub>(r − q), θ = 0.05"],
                 ["Horizonte h", "4 velas de 15 min: autocovarianzas hasta el rezago 3", "1 vela"],
                 ["Muestra", "Tramo de prueba: 30 % final, n = 443, sin barajar", "30 % final, n = 450, reestimado cada 100 velas"],
                 ["Estadístico", "DM con corrección HLN", "DM con corrección HLN"],
                 ["Distribución y nivel", "t de Student con n − 1 g.l., bilateral, α = 0.05", "t de Student con n − 1 g.l., bilateral, α = 0.05"]],
                [3.4 * cm, 7.4 * cm, 6.6 * cm]),
          P("5.7 Métricas de riesgo", "h2"),
          P("VaR<sub>α</sub> es la pérdida que solo se supera en el (1 − α) de los peores casos, y CVaR<sub>α</sub>, también llamado Expected Shortfall, es la pérdida "
            "promedio dentro de esa cola. Ambas se reportan como pérdidas positivas."),
          P("VaR<sub>α</sub> = −q<sub>1−α</sub>(X)     CVaR<sub>α</sub> = −E[ X | X ≤ q<sub>1−α</sub>(X) ]     normal: CVaR<sub>α</sub> = −(μ − σ·φ(z)/(1 − α))", "formula"),
          tabla([["Pieza", "Métrica", "Cómo se calcula", "Para qué sirve"],
                 ["Copiloto", "VaR y CVaR al 95 % y al 99 %", "Sobre las 10,000 trayectorias de P&L del Monte Carlo de cada candidata", "Tamaño del peor escenario frente a la pérdida máxima"],
                 ["Copiloto", "CVaR 99 % / pérdida máxima", "Cociente de las dos cifras", "Qué tanto del máximo se pierde en el peor 1 %"],
                 ["Copiloto", "Prob. de ganancia, valor esperado, P5 y P95", "Monte Carlo con deriva de la opinión y escenario neutral", "Insumos del semáforo y de la mesa"],
                 ["Copiloto", "Pérdida y ganancia máximas, breakevens", "Payoff exacto al vencimiento", "Límite contra el presupuesto"],
                 ["Copiloto", "Griegas de posición y de portafolio", "Black-Scholes con la cadena actual; medidores tipo válvula", "Alertas de cobertura de delta y de theta"],
                 ["Bot", "VaR y CVaR históricos y normales, 95 % y 99 %", "Retornos de 15 min de cada activo y P&L de las operaciones", "Comparar la cola real con la normal (curtosis)"],
                 ["Bot", "CAViaR (Engle y Manganelli)", "Regresión de cuantiles q<sub>t</sub> = β0 + β1·q<sub>t−1</sub> + β2·|r<sub>t−1</sub>|", "VaR que se adapta a la volatilidad"],
                 ["Bot", "Kupiec y Christoffersen", "Razón de verosimilitud χ²(1) y χ²(2) sobre las violaciones", "Validar que el VaR falle cerca del 5 % y sin rachas"],
                 ["Bot", "Sharpe, Sortino, drawdown máximo, win rate", "Curva de capital por estrategia", "Desempeño ajustado por riesgo"],
                 ["Bot", "Stop y take profit por ATR, trailing, tamaño 2 %, kill-switch 5 %", "Reglas en cada vela cerrada", "Limitar la pérdida por operación y por día"]],
                [2 * cm, 4.2 * cm, 6.2 * cm, 5 * cm]),
          P("5.8 Markowitz y correlación de Pearson", "h2"),
          P("Con un año de retornos diarios de los activos del bot se calcula la matriz de Pearson y se simulan 5,000 carteras long-only con pesos de Dirichlet. Para "
            "cada cartera, μ<sub>p</sub> = wᵀμ·252 y σ<sub>p</sub> = √(wᵀΣw·252). Se reportan la de máximo Sharpe y la de mínima volatilidad.")]

    # 6. Resultados
    s += [PageBreak(), P("6. Resultados", "h1"),
          P("6.1 Black-Scholes contra Binance Options", "h2"),
          P(f"Se compararon {int(bs['n'])} contratos de BTC, los más cercanos al dinero, con spot de {bs['spot']:,.0f} USDT. Como el mark price de Binance también sale "
            "de Black-Scholes con su IV de marca, esta prueba verifica la implementación con datos reales."),
          tabla([["Métrica", "Valor"],
                 ["Error absoluto medio del precio (USDT)", f(bs["mae_precio"], 4)], ["Error porcentual mediano del precio", f"{bs['err_precio_pct_mediano']:.4f} %"],
                 ["Error absoluto medio de la IV recuperada", f(bs["mae_iv"], 5)], ["Error absoluto medio de delta / correlación", f"{bs['mae_delta']:.6f} / {bs['corr_delta']:.5f}"],
                 ["Error absoluto medio de gamma / correlación", f"{bs['mae_gamma']:.7f} / {bs['corr_gamma']:.5f}"], ["Error absoluto medio de vega / correlación", f"{bs['mae_vega']:.5f} / {bs['corr_vega']:.5f}"],
                 ["Error absoluto medio de theta / correlación", f"{bs['mae_theta']:.3f} / {bs['corr_theta']:.5f}"]],
                [10 * cm, 6 * cm]),
          P("La theta es la única griega con diferencia apreciable en nivel, aunque su correlación es casi perfecta. La causa es de convención: nosotros la reportamos por "
            "día calendario y Binance usa su propia base de tiempo y tasa. Las colas del histograma corresponden a opciones muy fuera del dinero, con primas de pocos USDT."),
          figura("bs_vs_binance.png", 16.5, "Figura 2. Precio y delta de nuestro Black-Scholes contra Binance, y distribución del error relativo."),
          figura("griegas_por_strike.png", 16.5, "Figura 3. Delta, gamma, theta y vega por strike para calls de BTC."),
          PageBreak(),
          P("6.2 Simulación de hiperparámetros del Random Forest", "h2"),
          P("Para cada activo se probaron 27 combinaciones: n_estimators ∈ {50, 100, 200}, max_depth ∈ {3, 6, 10} y min_samples_leaf ∈ {1, 5, 10}. Se eligió la de mayor F1, "
            "con la accuracy como desempate. La tabla completa está en el Anexo A y en docs/resultados/hiperparametros_rf.csv.")]
    filas = [["Activo", "Hiperparámetros elegidos", "Accuracy", "Precision", "Recall", "F1", "RMSE", "MAE", "SMAPE", "Var. score"]]
    for a in activos:
        r = rf.loc[a]
        h = json.loads(r["hiperparametros"])
        filas.append([a, f"{h['n_estimators']} árboles, prof. {h['max_depth']}, hoja {h['min_samples_leaf']}", f(r["accuracy"]), f(r["precision"]), f(r["recall"]), f(r["f1"]),
                      f(r["rmse"]), f(r["mae"]), f(r["smape"], 1), f(r["variance_score"])])
    s += [KeepTogether([P("Mejor RF por activo, en el tramo de prueba de 443 velas:"), tabla(filas, [1.7 * cm, 3.5 * cm] + [1.52 * cm] * 8)]), Spacer(1, 8),
          figura("hiperparametros_rf.png", 12.5, "Figura 4. F1 y accuracy de las 27 combinaciones por activo; en azul intenso, la elegida.")]

    filas = [["Activo", "Modelo", "Accuracy", "Precision", "Recall", "F1", "RMSE", "MAE", "MAPE", "SMAPE"]]
    for a in activos:
        for nombre, tabla_m in (("Random Forest", rf), ("KNN (k = 15)", knn)):
            r = tabla_m.loc[a]
            filas.append([a, nombre, f(r["accuracy"]), f(r["precision"]), f(r["recall"]), f(r["f1"]), f(r["rmse"]), f(r["mae"]), f(r["mape"], 1), f(r["smape"], 1)])
    tec = met[~met["modelo"].isin(["Random Forest", "KNN (base)"])].pivot_table(index="activo", columns="modelo", values="f1").reindex(activos)
    s += [PageBreak(), P("6.3 Random Forest contra KNN y estrategias técnicas", "h2"), tabla(filas, [1.8 * cm, 2.4 * cm] + [1.6 * cm] * 8),
          P("MAPE y SMAPE son altos porque el objetivo de regresión es un retorno normalizado cercano a cero: dividir entre valores casi nulos infla el error porcentual. "
            "Por eso, para comparar modelos, el informe se apoya en RMSE, MAE, F1 y Diebold-Mariano."),
          P("F1 de las estrategias técnicas en el mismo tramo de prueba. Son estrategias de eventos: casi siempre dicen HOLD, por lo que su recall es muy bajo.")]
    s += [tabla([["Activo"] + list(tec.columns)] + [[a] + [f(v) for v in tec.loc[a]] for a in activos], [3 * cm] + [4.3 * cm] * len(tec.columns)),
          Spacer(1, 6),
          figura("matrices_confusion_prueba.png", 16.5, "Figura 5. Matrices de confusión del mejor RF en el tramo de prueba. Filas: dirección real; columnas: predicción."),
          figura("importancias_rf.png", 11, "Figura 6. Las diez variables más importantes del RF de BTC/USDT.")]

    filas = [["Activo", "Comparación", "DM", "DM HLN", "p-valor", "n", "Resultado (α = 0.05)"]]
    resaltar = []
    for i, r in enumerate(dm.itertuples(), start=1):
        filas.append([r.activo, r.comparacion, f(r.dm), f(r.dm_hln), f(r.p_valor, 4), r.n, r.resultado])
        if r.resultado != "Sin diferencia significativa":
            resaltar.append(i)
    rsi_gana = dm[dm["resultado"].str.contains("RSI")]
    s += [PageBreak(), P("6.4 Prueba de Diebold-Mariano", "h2"),
          P("Pérdida cuadrática, horizonte h = 4 y corrección HLN. Modelo 1 es siempre el Random Forest. Las filas resaltadas son diferencias significativas."),
          tabla(filas, [1.8 * cm, 4.2 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.2 * cm, 5.2 * cm], resaltar),
          P(f"El RF supera de forma significativa a EMA50 + Chandelier en {len(rf_gana)} de {len(activos)} activos, con p-valores menores a 0.001. Frente al KNN no hay "
            "diferencia significativa en ningún activo: con estas variables, los dos modelos extraen una señal igual de débil. "
            + (f"En {', '.join(rsi_gana['activo'])} la reversión por RSI tuvo menor pérdida que el RF (p = {rsi_gana['p_valor'].iloc[0]:.3f}), porque su señal casi siempre es cero "
               "y el error cuadrático de no apostar es bajo cuando los retornos son pequeños." if len(rsi_gana) else "")),
          P("6.5 Validación en tiempo real", "h2"),
          P("Se compararon las señales reales que el RF emitió en la VM con la dirección que tuvo el precio 4 velas después."),
          tabla([["Activo", "Señales evaluadas", "Accuracy", "Precision", "Recall", "F1", "Matriz [[TN, FP], [FN, TP]]"]] +
                [[r.activo, r.senales_evaluadas, f(r.accuracy), f(r.precision), f(r.recall), f(r.f1), r.matriz_confusion] for r in viv.itertuples()],
                [2.2 * cm, 2.4 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 4.6 * cm]),
          figura("matrices_confusion_vivo.png", 15, "Figura 7. Matrices de confusión de las señales en vivo del RF."),
          P("En vivo, la accuracy de BTC y SOL quedó por debajo de la de prueba. La muestra es pequeña, de un día y medio, y las observaciones se traslapan, así que "
            "el resultado no es concluyente, pero confirma que el modelo no tiene una ventaja predictiva estable en este horizonte.")]

    filas = [["Activo", "Estrategia", "Venue", "Ops. válidas", "P&L USD", "ROI %", "Sharpe/op", "Sortino/op", "DD máx. %", "Win rate %"]]
    for r in des.itertuples():
        if r.operaciones_validas == 0:
            continue
        filas.append([r.activo, r.estrategia, r.venue.replace("_", " "), r.operaciones_validas, f(r.pnl_usd, 2), f(r.roi_pct, 2), f(r.sharpe_por_operacion, 2),
                      f(r.sortino_por_operacion, 2), f(r.max_drawdown_pct, 2), f(r.win_rate_pct, 1)])
    cero = int(des["cantidad_cero"].sum())
    vela = int(des["salida_en_vela_de_entrada"].sum())
    s += [PageBreak(), P("6.6 Desempeño del bot en testnet y paper", "h2"),
          P(f"Operaciones cerradas por activo y estrategia, con 1,000 USD de capital virtual cada una. Sharpe y Sortino se calculan por operación, sin anualizar, "
            f"porque la muestra cubre menos de dos días. Total en el Binance spot testnet: <b>{pnl_testnet:+.2f} USD</b>, ROI de <b>{pnl_testnet / cap_testnet * 100:+.3f} %</b>."),
          tabla(filas, [1.9 * cm, 2.8 * cm, 2.9 * cm, 1.3 * cm, 1.4 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm]),
          figura("curvas_capital.png", 16.5, "Figura 8. Capital virtual por activo y estrategia a lo largo de las operaciones cerradas."),
          P(f"Se excluyen del desempeño {cero} operaciones de Alpaca Paper que se guardaron con cantidad cero por un error de registro, y hubo "
            f"{vela} salidas en la misma vela de entrada. Ambos problemas se describen y corrigieron en la sección 7. En Alpaca Paper quedan "
            f"{int(des[des['venue'] == 'alpaca_paper']['operaciones_validas'].sum())} operaciones válidas, con un P&L de "
            f"{des[des['venue'] == 'alpaca_paper']['pnl_usd'].sum():+.2f} USD.")]

    s += [KeepTogether([P("6.7 Copiloto de opciones", "h2"),
          tabla([["Fecha (UTC)", "Activo", "Opinión", "Horizonte", "Presupuesto", "Semáforo", "Estrategia elegida"]] +
                [[r.fecha, r.activo, f"{r.direccion} {r.magnitud_pct:.1f} %", f"{r.horizonte_dias} días", f"{r.presupuesto:,.0f} USD", r.semaforo,
                  r.estrategia_elegida if isinstance(r.estrategia_elegida, str) else "No confirmada"] for r in recs.itertuples()],
                [2.6 * cm, 1.4 * cm, 2.4 * cm, 1.7 * cm, 2 * cm, 1.7 * cm, 5.4 * cm])]),
          Spacer(1, 6),
          tabla([["Posición", "Activo", "Venue", "Costo de entrada USD", "Último valor USD", "Estado"]] +
                [[f"#{r.id}", r.asset, r.venue.replace("_", " "), f(r.entry_cost, 2), f(r.last_value, 2), r.status] for r in ops.itertuples()],
                [1.8 * cm, 1.6 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 2.4 * cm]), Spacer(1, 6),
          P("Las órdenes de BTC se ejecutaron en el testnet de Deribit: cada pata de la cadena de Binance se mapeó al instrumento de Deribit con el mismo vencimiento, "
            "strike y tipo, y la prima en USD se convirtió a BTC con el precio índice. La orden multipata de NVDA se envió a Alpaca Paper como orden mleg con límite neto.")]

    filas_corr = [["", *corr.columns]] + [[i, *[f(v, 2) for v in fila]] for i, fila in zip(corr.index, corr.values)]
    filas_mk = [["Cartera", "Retorno anual", "Volatilidad anual", "Sharpe", *[c.replace("w_", "Peso ") for c in mk.columns if c.startswith("w_")]]]
    for r in mk.to_dict("records"):
        filas_mk.append([r["cartera"], f"{r['ret'] * 100:.1f} %", f"{r['vol'] * 100:.1f} %", f(r["sharpe"], 2), *[f"{r[c] * 100:.1f} %" for c in mk.columns if c.startswith("w_")]])
    s += [PageBreak(), P("6.8 Correlación de Pearson y frontera de Markowitz", "h2"),
          figura("markowitz_pearson.png", 16.5, "Figura 9. Correlación de Pearson y frontera eficiente con un año de retornos diarios."),
          tabla(filas_corr, [2 * cm] + [2.4 * cm] * len(corr.columns)), Spacer(1, 6), tabla(filas_mk),
          P(_texto_markowitz(corr, mk))]

    filas_rc = [["Opinión", "Estrategia", "Pérdida máx.", "Prob. ganancia", "VaR 95 %", "CVaR 95 %", "VaR 99 %", "CVaR 99 %", "CVaR 99 % / máx."]]
    for r in rc.itertuples():
        filas_rc.append([r.opinion, r.estrategia, f(r.perdida_maxima, 1), f"{r.prob_ganancia:.0%}", f(r.var95, 1), f(r.cvar95, 1), f(r.var99, 1), f(r.cvar99, 1), f"{r.cvar99_sobre_perdida_max:.0%}"])
    s += [PageBreak(), P("6.9 Riesgo del copiloto: VaR y CVaR del Monte Carlo", "h2"),
          P("Ejemplo en vivo con BTC, 14 días y 500 USD de presupuesto, para dos opiniones. Pérdidas en USD al horizonte."),
          tabla(filas_rc, [2 * cm, 4.6 * cm, 1.5 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm]),
          P("En las estrategias de débito, como el long put o el bear put spread, más del 5 % de las trayectorias pierde toda la prima, así que el VaR y el CVaR "
            "coinciden con la pérdida máxima. En los iron condor, de crédito, la mayoría de los escenarios gana poco y la cola es más corta: el CVaR al 95 % queda "
            "muy por debajo de la pérdida máxima. Por eso el copiloto muestra las dos cifras juntas."),
          figura("riesgo_copiloto.png", 16.5, "Figura 10. Distribución del P&L simulado con VaR y CVaR al 95 %, y probabilidad acumulada del P&L.")]
    filas_rb = [["Activo", "VaR 95 % hist.", "CVaR 95 % hist.", "VaR 95 % normal", "CVaR 95 % normal", "CVaR 99 % hist.", "Curtosis", "CVaR 95 % por operación (USD)"]]
    for r in rb.itertuples():
        filas_rb.append([r.activo, f"{r.var95_hist_pct:.3f} %", f"{r.cvar95_hist_pct:.3f} %", f"{r.var95_normal_pct:.3f} %", f"{r.cvar95_normal_pct:.3f} %",
                         f"{r.cvar99_hist_pct:.3f} %", f(r.curtosis, 1), f(r.cvar95_operacion_usd, 2)])
    filas_cav = [["Activo", "β1", "β2", "Violaciones dentro", "Violaciones fuera", "Kupiec p", "Christoffersen p", "VaR histórico fuera", "DM HLN (p)"]]
    for r in cav.itertuples():
        filas_cav.append([r.activo, f(r.beta1, 3), f(r.beta2, 3), f"{r.violaciones_dentro_pct:.1f} %", f"{r.violaciones_fuera_pct:.1f} %", f(r.kupiec_p, 3), f(r.christoffersen_p, 3),
                          f"{r.violaciones_historico_pct:.1f} %", f"{r.dm_hln:.2f} ({r.dm_p:.2f})"])
    pasa = cav[cav["kupiec_p"] >= 0.05]
    s += [PageBreak(), P("6.10 Riesgo del bot: VaR, CVaR y CAViaR", "h2"),
          P("VaR y CVaR de los retornos de 15 minutos de cada activo, en la versión histórica y en la normal con la misma media y desviación, y CVaR del P&L por operación "
            "cuando hay al menos cinco operaciones cerradas."),
          tabla(filas_rb, [2 * cm, 1.9 * cm, 2 * cm, 2 * cm, 2.1 * cm, 2 * cm, 1.6 * cm, 3.6 * cm]),
          P("La curtosis está muy por encima de 0 en todos los activos: las colas son más pesadas que las de una normal. Por eso el CVaR histórico al 95 % es mayor que "
            "el normal, sobre todo en acciones, donde los saltos de apertura concentran las pérdidas grandes."),
          figura("var_cvar_bot.png", 16.5, "Figura 11. Retornos de 15 minutos con VaR y CVaR al 95 % y la normal ajustada."),
          P("CAViaR se estima con el 70 % inicial de las velas y se evalúa en el 30 % final, reestimándolo cada 100 velas solo con datos pasados."),
          tabla(filas_cav, [1.9 * cm, 1.3 * cm, 1.3 * cm, 2 * cm, 2 * cm, 1.6 * cm, 2.2 * cm, 2.3 * cm, 2.8 * cm]),
          P(f"CAViaR acierta la cobertura del 5 % en {', '.join(pasa['activo'])}, donde el VaR histórico es demasiado conservador. En las acciones falla por un cambio de "
            "régimen: la volatilidad se duplicó en el tramo de prueba y ningún modelo entrenado en el periodo tranquilo lo anticipa. Diebold-Mariano, con pérdida de "
            "cuantil y h = 1, no encuentra diferencia significativa entre los dos VaR en ningún activo."),
          figura("caviar_btc.png", 16.5, "Figura 12. VaR de CAViaR, VaR histórico móvil y violaciones en BTC/USDT.")]

    # 7. Riesgo
    s += [PageBreak(), P("7. Análisis de riesgo", "h1"),
          P("Riesgo de mercado del copiloto", "h2"),
          P("Todas las estrategias que el copiloto recomienda tienen pérdida máxima acotada y menor o igual al presupuesto del usuario: no se proponen ventas "
            "descubiertas. Mis posiciones agrega las griegas en dólares de todo el portafolio, con delta, theta diaria y vega total, y emite alertas cuando la delta se "
            "desvía, cuando se alcanza el 50 % de la ganancia máxima o cuando faltan tres días o menos para el vencimiento."),
          P("Riesgo del modelo", "h2"),
          P("El Random Forest no muestra una ventaja predictiva robusta: su F1 ronda 0.5 y Diebold-Mariano no lo distingue del KNN. Por eso el bot nunca depende solo "
            "de la predicción: el tamaño por volatilidad limita la pérdida de cada operación al 2 % del capital, el stop protege cada posición y el kill-switch corta "
            "las pérdidas del día. En la mesa del copiloto, la señal del RF solo es contexto para los agentes."),
          P("Riesgo operativo e incidentes corregidos", "h2"),
          lista(["<b>Salidas en la vela de entrada.</b> El bot comparaba el SL y el TP contra la vela en la que acababa de entrar, cuyos extremos ocurrieron antes de la "
                 f"entrada. Esto produjo {vela} cierres un minuto después de abrir. Se corrigió el 18 de noviembre: cada vela se revisa una sola vez y nunca la de entrada.",
                 "<b>Cantidad cero en Alpaca.</b> Alpaca responde al envío de una orden con cero acciones ejecutadas. El bot guardaba ese valor, y al cerrar enviaba órdenes "
                 "de cero acciones que Alpaca rechazaba. El broker acumuló un corto de 72 acciones de NVDA. Se corrigió confirmando el llenado antes de registrar la posición.",
                 "<b>Estrategias que comparten cuenta.</b> Alpaca guarda una sola posición neta por activo, mientras el bot lleva una por estrategia. "
                 "Cuando Alpaca rechazaba una apertura, esta quedaba en paper, pero su cierre se enviaba al broker y dejaba acciones reales sin seguimiento. "
                 "Se corrigió el 19 de noviembre: cada posición se cierra en el mismo venue donde se abrió.",
                 "<b>Órdenes netas en Alpaca.</b> Alpaca guarda una posición neta por acción y el bot una por estrategia. Desde el 19 de noviembre, cada operación "
                 "lleva la cuenta a la suma de las posiciones de todas las estrategias y solo envía la diferencia; cruzar de largo a corto se parte en dos órdenes.",
                 "<b>Mercado cerrado y subyacentes sin opciones.</b> Con la bolsa cerrada el bot ya no abre acciones en paper. En el copiloto, BTC y ETH usan contratos que "
                 "existen en Deribit, y lo que Deribit no cubre va al testnet de Gate.io; como ese libro no tiene contraparte, la orden se cancela a los 10 minutos y pasa a paper.",
                 "<b>Bloqueo geográfico.</b> Binance responde con error 451 a IPs de Estados Unidos, por eso la VM está en Querétaro y no en us-central1.",
                 "<b>Testnets inestables.</b> El testnet de Binance se reinicia periódicamente. La base de datos es la fuente de verdad y el bot reconcilia al arrancar. "
                 "Si un venue rechaza una orden, se usa paper interno y se avisa en #sistema.",
                 "<b>Cuotas de las APIs de IA.</b> La búsqueda de Google de Gemini devolvió error 429 por la cuota del plan. El sentimiento cae a titulares descargados "
                 "y lo indica en pantalla. Si Together falla, la mesa usa Gemini."]),
          P("Seguridad", "h2"),
          P("Todas las llaves viven en un archivo .env fuera del repositorio, también en la VM. El acceso a la app requiere Firebase Authentication. Se configuró una "
            "alerta de presupuesto de 15 USD en Google Cloud y límites de gasto en las APIs de IA.")]

    # 8. Despliegue
    s += [P("8. Despliegue en Google Cloud", "h1"),
          tabla([["Elemento", "Configuración"],
                 ["Máquina", "Compute Engine e2-medium, 2 vCPU y 4 GB de RAM, cubierta por créditos de Google for Education"],
                 ["Sistema y disco", "Ubuntu 22.04 LTS, 30 GB de disco persistente estándar"],
                 ["Región", "northamerica-south1-a, Querétaro, para evitar el bloqueo 451 de Binance"],
                 ["Procesos 24/7", "copiloto-bot.service y copiloto-app.service con Restart=always y EnvironmentFile=.env; sobreviven al reinicio de la VM"],
                 ["Red", "Regla de firewall copiloto-8501 para la URL pública de la app"],
                 ["Registros", "logs/trading_bot.log y logs/app.log con rotación, más la tabla events"],
                 ["Costos", "Alerta de presupuesto de 15 USD"]],
                [3.5 * cm, 13.9 * cm])]

    # 9. Limitaciones y conclusiones
    s += [KeepTogether([P("9. Limitaciones y trabajo futuro", "h1"),
          lista(["El periodo de operación en vivo es corto, menos de dos días, así que el ROI y las matrices en vivo no son estadísticamente concluyentes.",
                 "La liquidez de los testnets es artificial: las señales usan datos del mercado real, pero los precios de ejecución pueden diferir.",
                 "Agregar una red neuronal o gradient boosting y comparar con Diebold-Mariano; incorporar el régimen de volatilidad como variable.",
                 "Usar la frontera de Markowitz para repartir automáticamente el capital del bot entre activos.",
                 "Agregar cierres automáticos con stop en el copiloto de opciones, enviando la orden de cierre al broker."])]),
          P("10. Conclusiones", "h1"),
          P("El proyecto integra en una sola aplicación desplegada en la nube la valuación de derivados, la simulación Monte Carlo, el aprendizaje automático con "
            "validación estadística rigurosa, una mesa de agentes de IA de varios proveedores y la ejecución real en testnets. La implementación de Black-Scholes "
            "quedó verificada contra el mercado con errores de centésimas de punto porcentual."),
          P("El resultado más honesto del componente de aprendizaje automático es negativo: en horizontes de una hora, las variables técnicas no dan una ventaja "
            "predictiva estable, y Diebold-Mariano lo confirma al no distinguir el RF de un KNN simple. Esto refuerza el diseño del sistema: la gestión de riesgo, y no "
            "la predicción, es lo que protege el capital, y la IA se usa para interpretar y explicar, mientras los cálculos los hace Python."),
          P("Referencias", "h2"),
          lista(["Black, F. y Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.",
                 "Hull, J. C. (2022). Options, Futures, and Other Derivatives (11.ª ed.). Pearson.",
                 "Diebold, F. X. y Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.",
                 "Harvey, D., Leybourne, S. y Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.",
                 "Markowitz, H. (1952). Portfolio selection. The Journal of Finance, 7(1), 77-91.",
                 "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
                 "Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.",
                 "Wilder, J. W. (1978). New Concepts in Technical Trading Systems. Trend Research."])]

    # Anexos
    t = hp[hp["activo"] == ("BTC/USDT" if "BTC/USDT" in activos else activos[0])].sort_values("f1", ascending=False)
    filas = [["Árboles", "Prof. máx.", "Mín. hoja", "Accuracy", "Precision", "Recall", "F1", "RMSE", "MAE", "MAPE", "SMAPE", "Var. score"]]
    for r in t.itertuples():
        filas.append([int(r.n_estimators), int(r.max_depth), int(r.min_samples_leaf), f(r.accuracy), f(r.precision), f(r.recall), f(r.f1), f(r.rmse), f(r.mae), f(r.mape, 1), f(r.smape, 1), f(r.variance_score)])
    s += [PageBreak(), P("Anexo A. Tabla completa de hiperparámetros (BTC/USDT)", "h1"),
          P("Ordenada por F1. Las tablas de todos los activos están en docs/resultados/hiperparametros_rf.csv."),
          tabla(filas, [1.7 * cm, 1.5 * cm, 1.9 * cm] + [1.37 * cm] * 9, [1]),
          PageBreak(), P("Anexo B. Estructura del repositorio y reproducibilidad", "h1"),
          tabla([["Carpeta", "Contenido"],
                 ["app/", "Streamlit: login con Firebase, navegación y las cinco pestañas"],
                 ["core/", "Configuración, base de datos, bus de eventos, búsqueda de activos, posiciones y ejecución"],
                 ["data_sources/", "Binance Options, Alpaca, Deribit, Gate.io testnet, yfinance, velas y noticias"],
                 ["quant/", "Black-Scholes, volatilidad, estrategias multipata, selector, Monte Carlo, semáforo, portafolio, VaR y CVaR, y CAViaR"],
                 ["agents/", "Clientes de Together, Gemini y Grok, roles y orquestación de la mesa"],
                 ["bot/", "Runner 24/7, ejecución por venue, riesgo, variables, entrenamiento y estrategias"],
                 ["validation/", "Métricas, Diebold-Mariano, simulación de hiperparámetros y Black-Scholes vs Binance"],
                 ["notebooks/", "validacion_cuantitativa.ipynb, ejecutado con datos reales"],
                 ["scripts/", "check_apis.py, exportar_resultados.py, crear_notebook.py y generar_informe.py"],
                 ["docs/", "Este informe, la presentación y las tablas y gráficas en docs/resultados/"],
                 ["deploy/", "Servicios systemd y setup_vm.sh"],
                 ["tests/", "72 pruebas: Black-Scholes, Diebold-Mariano, semáforo, estrategias, ejecución, trailing stop, VaR, CVaR, CAViaR, Gate.io y órdenes netas en Alpaca"]],
                [3.2 * cm, 14.2 * cm]),
          Spacer(1, 8),
          P("Para reproducir los resultados:"),
          P("python scripts/exportar_resultados.py --db data/app.db<br/>python scripts/crear_notebook.py<br/>python scripts/generar_informe.py", "formula")]
    return s


def main() -> None:
    """Genera el diagrama de arquitectura y el PDF del informe."""
    datos = yaml.safe_load((RAIZ / "docs" / "portada.yaml").read_text(encoding="utf-8"))
    diagrama_arquitectura(RES / "arquitectura.png")
    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"{datos['titulo']}: informe técnico", author=", ".join(datos.get("integrantes", [])), subject=datos["materia"])
    historia = [PageBreak()] + contenido(datos)
    doc.build(historia, onFirstPage=lambda c, d: _portada(c, d, datos), onLaterPages=lambda c, d: _pagina(c, d, datos["titulo"]))
    print(f"Informe generado en {SALIDA}")


if __name__ == "__main__":
    main()
