"""Autenticación con Firebase (email/contraseña) vía la REST API de Identity Toolkit.

Sin `FIREBASE_API_KEY` se activa un modo demo local (cualquier email entra) para
que la app siga siendo usable en desarrollo.
"""
from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from core.config import env

_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
_ERRORES = {
    "EMAIL_EXISTS": "Ese correo ya está registrado.",
    "INVALID_PASSWORD": "Contraseña incorrecta.",
    "INVALID_LOGIN_CREDENTIALS": "Correo o contraseña incorrectos.",
    "EMAIL_NOT_FOUND": "No existe una cuenta con ese correo.",
    "WEAK_PASSWORD : Password should be at least 6 characters": "La contraseña debe tener al menos 6 caracteres.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Demasiados intentos; espera unos minutos.",
    "INVALID_EMAIL": "El correo no es válido.",
}


def firebase_configurado() -> bool:
    """True si hay API key de Firebase."""
    return bool(env("FIREBASE_API_KEY"))


def _llamar(accion: str, email: str, password: str) -> dict[str, Any]:
    """POST a signUp / signInWithPassword; devuelve el JSON o lanza ValueError con mensaje en español."""
    r = requests.post(f"{_BASE}:{accion}", params={"key": env("FIREBASE_API_KEY")}, json={"email": email, "password": password, "returnSecureToken": True}, timeout=15)
    data = r.json()
    if r.status_code != 200:
        msg = (data.get("error") or {}).get("message", "ERROR")
        raise ValueError(_ERRORES.get(msg, f"Error de autenticación: {msg}"))
    return data


def registrar(email: str, password: str) -> dict[str, Any]:
    """Crea la cuenta y devuelve `{user_id, email, token}`."""
    if not firebase_configurado():
        return {"user_id": email.lower(), "email": email, "token": "demo"}
    d = _llamar("signUp", email, password)
    return {"user_id": d["localId"], "email": d["email"], "token": d["idToken"]}


def iniciar_sesion(email: str, password: str) -> dict[str, Any]:
    """Inicia sesión y devuelve `{user_id, email, token}`."""
    if not firebase_configurado():
        return {"user_id": email.lower(), "email": email, "token": "demo"}
    d = _llamar("signInWithPassword", email, password)
    return {"user_id": d["localId"], "email": d["email"], "token": d["idToken"]}


def usuario_actual() -> dict[str, Any] | None:
    """Usuario guardado en la sesión de Streamlit."""
    return st.session_state.get("usuario")


def cerrar_sesion() -> None:
    """Borra la sesión."""
    st.session_state.pop("usuario", None)


def pantalla_login() -> None:
    """Formulario de registro / inicio de sesión. Detiene la ejecución hasta que haya usuario."""
    st.title("Copiloto de Opciones con IA")
    st.caption("Paper trading educativo. Nunca se opera con dinero real.")
    if not firebase_configurado():
        st.info("Firebase no está configurado: modo demo (entra con cualquier correo).")
    tab_in, tab_up = st.tabs(["Iniciar sesión", "Registrarse"])
    with tab_in:
        with st.form("login"):
            email = st.text_input("Correo")
            pw = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Entrar", use_container_width=True):
                try:
                    st.session_state["usuario"] = iniciar_sesion(email.strip(), pw)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with tab_up:
        with st.form("signup"):
            email2 = st.text_input("Correo", key="e2")
            pw2 = st.text_input("Contraseña (mín. 6 caracteres)", type="password", key="p2")
            if st.form_submit_button("Crear cuenta", use_container_width=True):
                try:
                    st.session_state["usuario"] = registrar(email2.strip(), pw2)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    st.stop()
