"""CAViaR: Value at Risk condicional autorregresivo (Engle y Manganelli, 2004).

En lugar de suponer una distribución, CAViaR modela directamente el cuantil θ de los retornos y lo deja
evolucionar en el tiempo. Se implementan dos especificaciones del artículo, para el cuantil q_t (un retorno,
negativo en la cola izquierda):

- Valor absoluto simétrico (SAV):  q_t = β0 + β1·q_{t−1} + β2·|r_{t−1}|
- Pendiente asimétrica (AS):        q_t = β0 + β1·q_{t−1} + β2·max(r_{t−1}, 0) + β3·max(−r_{t−1}, 0)

Los parámetros se estiman por regresión de cuantiles: minimizan la pérdida de cuantil (tick loss)
ρ_θ(u) = u·(θ − 1{u < 0}), con u = r_t − q_t. El VaR se reporta como pérdida positiva: VaR_t = −q_t.

El módulo incluye el backtesting estándar: proporción de violaciones con la prueba de Kupiec (cobertura
incondicional), la de Christoffersen (independencia) y la conjunta de cobertura condicional.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, stats
from scipy.signal import lfilter

MODELOS = {"sav": "Valor absoluto simétrico", "as": "Pendiente asimétrica"}


def perdida_cuantil(r: np.ndarray, q: np.ndarray, theta: float) -> np.ndarray:
    """Pérdida de cuantil (tick loss) de cada observación: (r − q)·(θ − 1{r < q})."""
    u = np.asarray(r, float) - np.asarray(q, float)
    return u * (theta - (u < 0))


def _entradas(r: np.ndarray, beta: np.ndarray, modelo: str) -> np.ndarray:
    """Término que no depende de q_{t−1}: β0 más la respuesta al retorno anterior, para t = 1..n−1."""
    r_prev = r[:-1]
    if modelo == "sav":
        return beta[0] + beta[2] * np.abs(r_prev)
    return beta[0] + beta[2] * np.maximum(r_prev, 0.0) + beta[3] * np.maximum(-r_prev, 0.0)


def recursion(r: np.ndarray, beta: np.ndarray, modelo: str, q0: float) -> np.ndarray:
    """Serie de cuantiles q_t dados los parámetros, con q_0 fijo; usa un filtro lineal (sin bucles en Python)."""
    x = _entradas(r, beta, modelo)
    resto = lfilter([1.0], [1.0, -beta[1]], x, zi=[beta[1] * q0])[0]
    return np.concatenate([[q0], resto])


@dataclass
class ResultadoCAViaR:
    """Modelo CAViaR estimado."""

    modelo: str
    theta: float
    beta: list[float]
    q0: float
    perdida_media: float
    n_estimacion: int
    convergio: bool
    nombres: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convierte a diccionario."""
        return asdict(self)


def estimar(r: np.ndarray, theta: float = 0.05, modelo: str = "sav", n_inicial: int = 300, n_aleatorios: int = 400, semilla: int = 7) -> ResultadoCAViaR:
    """Estima un CAViaR por regresión de cuantiles.

    Siguiendo a Engle y Manganelli, q_0 es el cuantil empírico de las primeras `n_inicial` observaciones. Se evalúan
    `n_aleatorios` vectores de parámetros al azar más uno razonable fijo, y los cinco mejores se refinan con
    Nelder-Mead dos veces seguidas. Se exige |β1| < 1 para que la recursión sea estable.

    Args:
        r: Retornos (en %, para que la escala ayude al optimizador).
        theta: Cuantil a modelar; 0.05 corresponde al VaR al 95 %.
        modelo: "sav" o "as".
    """
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if modelo not in MODELOS:
        raise ValueError(f"modelo desconocido: {modelo}")
    if len(r) < max(100, n_inicial // 2):
        raise ValueError("Hacen falta más observaciones para estimar CAViaR")
    q0 = float(np.quantile(r[: min(n_inicial, len(r))], theta))
    s = float(np.std(r)) or 1.0
    k = 3 if modelo == "sav" else 4

    def objetivo(beta: np.ndarray) -> float:
        if not np.all(np.isfinite(beta)) or abs(beta[1]) >= 0.999:
            return 1e6
        q = recursion(r, beta, modelo, q0)
        if not np.all(np.isfinite(q)):
            return 1e6
        return float(perdida_cuantil(r, q, theta).mean())

    rng = np.random.default_rng(semilla)
    candidatos = [np.array([0.1 * q0, 0.9, -0.3] + ([-0.3] if k == 4 else []))]
    for _ in range(n_aleatorios):
        b = [rng.uniform(-s, 0.0), rng.uniform(0.0, 0.99), rng.uniform(-1.5, 0.0)]
        if k == 4:
            b = [b[0], b[1], rng.uniform(-1.5, 0.5), rng.uniform(-1.5, 0.0)]
        candidatos.append(np.array(b))
    candidatos.sort(key=objetivo)
    mejor, mejor_f, convergio = candidatos[0], objetivo(candidatos[0]), False
    for inicio in candidatos[:5]:
        b = inicio
        for _ in range(2):
            res = optimize.minimize(objetivo, b, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-10})
            b = res.x
        f = objetivo(b)
        if f < mejor_f:
            mejor, mejor_f, convergio = b, f, bool(res.success)
    nombres = ["β0 constante", "β1 VaR anterior", "β2 |retorno anterior|"] if k == 3 else \
        ["β0 constante", "β1 VaR anterior", "β2 subidas", "β3 caídas"]
    return ResultadoCAViaR(modelo, theta, [float(x) for x in mejor], q0, mejor_f, int(len(r)), convergio, nombres)


def pronosticar(r: np.ndarray, modelo: ResultadoCAViaR) -> np.ndarray:
    """Cuantiles q_t con los parámetros fijos sobre una serie (dentro o fuera de la muestra de estimación)."""
    return recursion(np.asarray(r, float), np.asarray(modelo.beta), modelo.modelo, modelo.q0)


def var_historico_movil(r: np.ndarray, theta: float = 0.05, ventana: int = 300) -> np.ndarray:
    """Cuantil histórico con ventana móvil, usando solo datos anteriores a cada observación (NaN al inicio)."""
    r = np.asarray(r, float)
    q = np.full(len(r), np.nan)
    for t in range(ventana, len(r)):
        q[t] = np.quantile(r[t - ventana:t], theta)
    return q


def kupiec(violaciones: np.ndarray, theta: float) -> dict[str, float]:
    """Prueba de Kupiec (1995) de cobertura incondicional: ¿la proporción de violaciones es θ?

    LR = −2·ln[(1−θ)^(n−x)·θ^x] + 2·ln[(1−π)^(n−x)·π^x], con π = x/n; se distribuye χ² con 1 grado de libertad.
    """
    v = np.asarray(violaciones, bool)
    n, x = len(v), int(v.sum())
    if n == 0:
        return {"n": 0, "violaciones": 0, "tasa": float("nan"), "lr": float("nan"), "p_valor": float("nan")}
    pi = x / n
    log_h0 = (n - x) * np.log(1 - theta) + x * np.log(theta)
    log_h1 = ((n - x) * np.log(1 - pi) if pi < 1 else 0.0) + (x * np.log(pi) if x > 0 else 0.0)
    lr = float(-2 * (log_h0 - log_h1))
    return {"n": n, "violaciones": x, "tasa": pi, "lr": lr, "p_valor": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(violaciones: np.ndarray, theta: float) -> dict[str, float]:
    """Prueba de Christoffersen (1998) de independencia y la conjunta de cobertura condicional.

    Compara la probabilidad de violar el VaR justo después de una violación (π11) contra la de violarlo
    después de una vela normal (π01). Si las violaciones llegan en racha, el VaR no se adapta a tiempo.
    """
    v = np.asarray(violaciones, int)
    if len(v) < 3:
        return {"lr_ind": float("nan"), "p_ind": float("nan"), "lr_cc": float("nan"), "p_cc": float("nan")}
    a, b = v[:-1], v[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def xlog(n: int, p: float) -> float:
        return n * np.log(p) if n > 0 and p > 0 else 0.0

    log_h0 = xlog(n00 + n10, 1 - pi) + xlog(n01 + n11, pi)
    log_h1 = xlog(n00, 1 - pi01) + xlog(n01, pi01) + xlog(n10, 1 - pi11) + xlog(n11, pi11)
    lr_ind = float(max(-2 * (log_h0 - log_h1), 0.0))
    lr_cc = kupiec(v.astype(bool), theta)["lr"] + lr_ind
    return {"pi01": pi01, "pi11": pi11, "lr_ind": lr_ind, "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "lr_cc": float(lr_cc), "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2))}


def analizar(r: np.ndarray, theta: float = 0.05, modelo: str = "sav", frac_estimacion: float = 0.7, ventana_historica: int = 300,
             reestimar_cada: int | None = 100) -> dict[str, Any]:
    """Estima CAViaR en el primer tramo, lo evalúa fuera de muestra y lo compara contra el VaR histórico móvil.

    Fuera de muestra el modelo se reestima cada `reestimar_cada` velas con todos los datos disponibles hasta ese
    momento (ventana creciente), como se usaría en producción; cada bloque se pronostica solo con información pasada.
    Así el VaR se adapta a cambios de régimen, por ejemplo cuando la volatilidad se duplica. `None` lo deja fijo.

    Returns:
        Diccionario con el modelo inicial y el último, las series (`r`, `q_caviar`, `q_historico`), el índice donde empieza
        la prueba, el backtesting dentro y fuera de muestra, y Diebold-Mariano sobre la pérdida de cuantil fuera de muestra
        (modelo 1 = CAViaR, modelo 2 = histórico móvil).
    """
    from validation.diebold_mariano import diebold_mariano

    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    corte = int(len(r) * frac_estimacion)
    m = estimar(r[:corte], theta, modelo)
    q_cav = pronosticar(r, m)
    ultimo, n_reest = m, 0
    if reestimar_cada:
        for s in range(corte + reestimar_cada, len(r), reestimar_cada):
            e = min(s + reestimar_cada, len(r))
            ultimo = estimar(r[:s], theta, modelo, n_aleatorios=150)
            q_cav[s:e] = pronosticar(r[:e], ultimo)[s:e]
            n_reest += 1
    q_hist = var_historico_movil(r, theta, ventana_historica)
    dentro, fuera = slice(0, corte), slice(corte, len(r))
    viol = r < q_cav
    out: dict[str, Any] = {
        "modelo": m.to_dict(), "modelo_ultimo": ultimo.to_dict(), "reestimaciones": n_reest, "reestimar_cada": reestimar_cada,
        "r": r, "q_caviar": q_cav, "q_historico": q_hist, "inicio_prueba": corte,
        "kupiec_dentro": kupiec(viol[dentro], theta), "kupiec_fuera": kupiec(viol[fuera], theta),
        "christoffersen_fuera": christoffersen(viol[fuera], theta),
    }
    valido = fuera.start + np.flatnonzero(~np.isnan(q_hist[fuera]))
    if len(valido) > 10:
        out["kupiec_historico_fuera"] = kupiec(r[valido] < q_hist[valido], theta)
        l_cav = perdida_cuantil(r[valido], q_cav[valido], theta)
        l_hist = perdida_cuantil(r[valido], q_hist[valido], theta)
        out["perdida_cuantil_fuera"] = {"caviar": float(l_cav.mean()), "historico": float(l_hist.mean())}
        out["dm_vs_historico"] = diebold_mariano(l_cav, l_hist, h=1, perdida="directa").to_dict()
    return out
