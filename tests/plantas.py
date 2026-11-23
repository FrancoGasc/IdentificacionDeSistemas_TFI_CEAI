"""Plantas sinteticas de prueba, con parametros verdaderos conocidos.

Cada planta recibe el tipo de ruido que la estructura asociada supone en su
hipotesis. De ese modo cada estimador se evalua sobre un escenario coherente
con lo que fue disenado para representar, y no se lo penaliza por un tipo de
ruido que su estructura no puede describir. La comparacion resulta
metodologicamente justa.

Los datos se generan con una semilla fija, de modo que los ensayos son
exactamente reproducibles. El generador PCG64 de NumPy garantiza la misma
secuencia entre versiones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import identificacion_sistemas as ids


@dataclass
class Planta:
    """Un ensayo sintetico con su verdad de terreno."""

    nombre: str
    datos: "ids.DatosExperimento"
    # Polinomios discretos exactos (discretizacion ZOH de la planta continua)
    A: np.ndarray
    B: np.ndarray
    nk: int
    # Parametros fisicos verdaderos
    K: float
    tau: float

    @property
    def ganancia_discreta(self) -> float:
        return float(np.sum(self.B) / np.sum(self.A))


def prbs(n: int, rng: np.random.Generator) -> np.ndarray:
    """Secuencia binaria pseudoaleatoria.

    Es la excitacion habitual en identificacion: su espectro se aproxima al de
    un ruido blanco dentro de una banda, de modo que excita un rango amplio de
    frecuencias, y al tomar solo dos valores es simple de generar y poco
    exigente para el actuador. Un escalon, en cambio, concentra su energia en
    muy pocas frecuencias y no alcanza para identificar dinamicas de orden
    elevado.
    """
    return np.sign(rng.standard_normal(n))


def primer_orden(
    tipo_ruido: str,
    n: int = 1500,
    K: float = 2.0,
    tau: float = 5.0,
    muestras_por_tau: float = 15.0,
    nk: int = 1,
    sigma: float = 0.05,
    seed: int = 20261123,
) -> Planta:
    """Planta de primer orden K/(tau s + 1) discretizada por ZOH.

    El periodo de muestreo se elige como tau/muestras_por_tau, siguiendo la
    regla de tomar del orden de quince a veinte muestras por constante de
    tiempo. Un muestreo demasiado lento pierde informacion de la dinamica; uno
    demasiado rapido acumula los polos cerca de z=1 y degrada el
    condicionamiento del problema.

    tipo_ruido
    ----------
    "blanco" : e(t) directo a la salida            -> hipotesis de OE
    "arx"    : coloreado por 1/A                   -> hipotesis de ARX
    "armax"  : numerador propio C, denominador A   -> hipotesis de ARMAX
    "bj"     : dinamica propia C/D                 -> hipotesis de BJ
    """
    ts = tau / muestras_por_tau
    a = float(np.exp(-ts / tau))
    A = np.array([1.0, -a])
    B = np.array([K * (1.0 - a)])

    rng = np.random.default_rng(seed)
    u = prbs(n, rng)

    # Salida limpia: y(t) = a y(t-1) + K(1-a) u(t-nk)
    y_limpia = np.zeros(n)
    for t in range(max(1, nk), n):
        y_limpia[t] = a * y_limpia[t - 1] + B[0] * u[t - nk]

    e = sigma * rng.standard_normal(n)
    v = np.zeros(n)

    if tipo_ruido == "blanco":
        v = e
    elif tipo_ruido == "arx":
        for t in range(1, n):
            v[t] = a * v[t - 1] + e[t]
    elif tipo_ruido == "armax":
        for t in range(1, n):
            v[t] = a * v[t - 1] + e[t] - 0.5 * e[t - 1]
    elif tipo_ruido == "bj":
        for t in range(1, n):
            v[t] = 0.3 * v[t - 1] + e[t] - 0.5 * e[t - 1]
    else:
        raise ValueError(f"tipo de ruido desconocido: {tipo_ruido!r}")

    datos = ids.DatosExperimento.desde_arrays(
        u, y_limpia + v, ts, nombre=f"primer orden / ruido {tipo_ruido}"
    )
    return Planta(
        nombre=f"primer_orden_{tipo_ruido}",
        datos=datos,
        A=A,
        B=B,
        nk=nk,
        K=K,
        tau=tau,
    )


def segundo_orden_subamortiguado(
    n: int = 1500,
    omega_n: float = 2.0,
    zeta: float = 0.4,
    muestras_por_periodo: float = 20.0,
    sigma: float = 0.02,
    seed: int = 20261123,
) -> Planta:
    """Planta de segundo orden subamortiguada, con ruido de tipo ARMAX.

    Es la dinamica que los metodos no parametricos describen peor: estimar zeta
    a partir del sobrepico de una respuesta ruidosa es notoriamente impreciso,
    mientras la identificacion parametrica la recupera con varias cifras.
    """
    periodo = 2 * np.pi / (omega_n * np.sqrt(1 - zeta ** 2))
    ts = periodo / muestras_por_periodo

    # Discretizacion ZOH de omega_n^2 / (s^2 + 2 zeta omega_n s + omega_n^2)
    wd = omega_n * np.sqrt(1 - zeta ** 2)
    r = float(np.exp(-zeta * omega_n * ts))
    A = np.array([1.0, -2 * r * np.cos(wd * ts), r ** 2])
    # Numerador con ganancia estatica unitaria
    B = np.array([0.0, float(np.sum(A))])

    rng = np.random.default_rng(seed)
    u = prbs(n, rng)

    y_limpia = np.zeros(n)
    for t in range(2, n):
        y_limpia[t] = -A[1] * y_limpia[t - 1] - A[2] * y_limpia[t - 2] + B[1] * u[t - 1]

    e = sigma * rng.standard_normal(n)
    v = np.zeros(n)
    for t in range(2, n):
        v[t] = -A[1] * v[t - 1] - A[2] * v[t - 2] + e[t] - 0.5 * e[t - 1]

    datos = ids.DatosExperimento.desde_arrays(
        u, y_limpia + v, ts, nombre="segundo orden subamortiguado / ruido ARMAX"
    )
    return Planta(
        nombre="segundo_orden_armax",
        datos=datos,
        A=A,
        B=B[1:],
        nk=1,
        K=1.0,
        tau=1.0 / (zeta * omega_n),
    )
