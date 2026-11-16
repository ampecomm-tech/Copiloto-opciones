"""Entrada de Streamlit: login con Firebase y navegación por pestañas.

Ejecutar desde la raíz del repo: `streamlit run app/main.py`.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

from app import auth, ui  # noqa: E402
from core import db  # noqa: E402
from core.config import cfg  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title=cfg("app", "titulo", default="Copiloto de Opciones"), page_icon=":material/show_chart:", layout="wide")
ui.aplicar_estilo()
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icono.svg"))
db.inicializar()

import importlib  # noqa: E402


def _pagina(modulo: str):
    """Importa `app/pages/<modulo>.py` (los nombres llevan prefijo numérico) y devuelve su `render`."""
    return importlib.import_module(f"app.pages.{modulo}").render


paginas = [
    st.Page(_pagina("1_noticias"), title="Noticias y sentimiento", icon=":material/newspaper:", url_path="noticias-sentimiento", default=True),
    st.Page(_pagina("2_copiloto"), title="Copiloto de opciones", icon=":material/explore:", url_path="copiloto-opciones"),
    st.Page(_pagina("3_posiciones"), title="Mis posiciones", icon=":material/account_balance_wallet:", url_path="mis-posiciones"),
    st.Page(_pagina("4_bot"), title="Bot de trading", icon=":material/smart_toy:", url_path="bot-trading"),
    st.Page(_pagina("5_centro_control"), title="Centro de control", icon=":material/monitoring:", url_path="centro-control"),
]
# st.navigation debe declararse en cada ejecución (si no, Streamlit muestra `pages/` automáticamente).
# Las rutas (url_path) NO deben coincidir con los nombres de archivo de `pages/`, porque una URL
# directa a /copiloto haría que Streamlit ejecutara `2_copiloto.py` como página automática.
usuario = auth.usuario_actual()
pagina = st.navigation(paginas, position="sidebar" if usuario else "hidden")
if not usuario:
    auth.pantalla_login()

with st.sidebar:
    st.divider()
    c_usuario, c_salir = st.columns([4, 1], vertical_alignment="center")
    correo = html.escape(usuario["email"])
    c_usuario.html(f"<div class='cp-usuario' title='{correo}'>{correo}</div>")
    if c_salir.button("", icon=":material/logout:", help="Cerrar sesión", type="tertiary"):
        auth.cerrar_sesion()
        st.rerun()
    st.html("<div class='cp-pie'>Paper trading · nunca dinero real</div>")
pagina.run()
