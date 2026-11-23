"""Utilidades de polinomios en el operador de desplazamiento q^-1.

Convencion de la libreria
-------------------------
Un polinomio monico se representa como un array de coeficientes en potencias
crecientes de q^-1, empezando por el 1:

    P(q) = 1 + p1*q^-1 + p2*q^-2 + ... + pn*q^-n   ->   [1, p1, p2, ..., pn]

Los polinomios del numerador (B) no son monicos y no llevan ese 1 inicial:

    B(q) = b1*q^-1 + ... + bnb*q^-nb                ->  [b1, ..., bnb]

con el retardo nk aplicado por separado.

Estabilidad
-----------
Las raices del polinomio caracteristico asociado a [1, p1, ..., pn] son las de

    z^n + p1*z^(n-1) + ... + pn = 0

de modo que np.roots() se aplica directamente al array de coeficientes. El
polinomio es estable cuando todas sus raices quedan dentro del circulo unitario.

Estas funciones estaban duplicadas en los modulos arx, armax, oe y bj de la
implementacion original; aca viven en un solo lugar.
"""

from __future__ import annotations

import numpy as np

# Radio usado como restriccion DURANTE la optimizacion. Es menor que 1 a
# proposito: deja un margen para no quedar pegado al limite de estabilidad,
# donde el predictor recursivo se vuelve numericamente delicado.
RADIO_OPTIMIZACION = 0.999

# Radio usado para juzgar la estabilidad de un polinomio ya estimado.
RADIO_ESTABILIDAD = 1.0


def es_estable(coeficientes, radio: float = RADIO_OPTIMIZACION) -> bool:
    """Indica si 1 + c1*q^-1 + ... tiene todas sus raices de modulo < radio.

    Parametros
    ----------
    coeficientes : array_like
        Coeficientes [c1, c2, ..., cn] SIN el 1 inicial. Una secuencia vacia
        corresponde al polinomio P(q) = 1, que es estable.
    radio : float
        Radio maximo admitido. El valor por omision, 0.999, es la restriccion
        que se aplica durante la optimizacion no lineal; para juzgar un
        polinomio ya estimado corresponde usar RADIO_ESTABILIDAD = 1.0.

    Notas
    -----
    A diferencia de la implementacion original, el radio es un parametro
    explicito. Antes la comprobacion usaba 0.999 y la estabilizacion reflejaba
    con 1.0, de modo que una raiz en 0.9995 era "inestable" para una funcion e
    "intocable" para la otra.
    """
    coeficientes = np.asarray(coeficientes, dtype=float).ravel()
    if coeficientes.size == 0:
        return True
    p = np.concatenate([[1.0], coeficientes])
    try:
        raices = np.roots(p)
    except (np.linalg.LinAlgError, ValueError):
        return False
    if raices.size == 0:
        return True
    return bool(np.all(np.abs(raices) < radio))


def estabilizar(polinomio) -> np.ndarray:
    """Refleja al interior del circulo unitario las raices inestables.

    Recibe el polinomio COMPLETO [1, p1, ..., pn] y devuelve otro de la misma
    forma, monico, cuyas raices cumplen |z| <= 1. La reflexion z -> 1/conj(z)
    preserva la respuesta en magnitud del filtro, de modo que el modelo de
    ruido sigue describiendo el mismo espectro pero su inverso ya no diverge.
    """
    polinomio = np.asarray(polinomio, dtype=float).ravel()
    if polinomio.size <= 1:
        return polinomio
    try:
        raices = np.roots(polinomio)
    except (np.linalg.LinAlgError, ValueError):
        return polinomio

    hubo_cambio = False
    for i, r in enumerate(raices):
        if abs(r) > RADIO_ESTABILIDAD:
            raices[i] = 1.0 / np.conj(r) if abs(r) > 1e-10 else 0.0
            hubo_cambio = True

    if not hubo_cambio:
        return polinomio

    nuevo = np.poly(raices).real
    if abs(nuevo[0]) > 1e-10:
        nuevo = nuevo / nuevo[0]
    return nuevo


def ventana_pasada(senal: np.ndarray, t: int, cantidad: int) -> np.ndarray:
    """Devuelve [x(t-1), x(t-2), ..., x(t-cantidad)].

    Equivale a senal[t-1 : t-cantidad-1 : -1] pero sin el caso borde de ese
    slice: cuando t == cantidad el indice de parada vale -1, que Python
    interpreta como "el penultimo elemento" en lugar de "antes del cero", y el
    slice sale vacio. Ese era el origen del fallo de ARX con na >= nb + nk.
    """
    if cantidad <= 0:
        return np.empty(0, dtype=float)
    if t - cantidad < 0:
        raise IndexError(
            f"se piden {cantidad} muestras pasadas en t={t}: no hay historia suficiente"
        )
    return senal[t - cantidad:t][::-1]


def formatear_polinomio(
    coeficientes,
    retardo_base: int = 0,
    monico: bool = False,
    tolerancia: float = 1e-6,
    decimales: int = 4,
) -> str:
    """Formatea un polinomio como texto legible, p. ej. "1 - 0,3364 z^-1".

    Parametros
    ----------
    coeficientes : array_like
        Coeficientes en potencias crecientes de q^-1.
    retardo_base : int
        Potencia de q^-1 del primer coeficiente. Para B(q) con retardo nk se
        pasa retardo_base=nk.
    monico : bool
        Si es True, el primer coeficiente se imprime como "1" literal (caso de
        A, F, C y D).
    """
    coeficientes = np.asarray(coeficientes, dtype=float).ravel()
    terminos: list[str] = []

    for i, c in enumerate(coeficientes):
        potencia = retardo_base + i
        if monico and i == 0:
            terminos.append("1")
            continue
        if abs(c) < tolerancia:
            continue
        signo = " + " if c >= 0 else " - "
        if potencia == 0:
            termino = f"{abs(c):.{decimales}f}"
        else:
            termino = f"{abs(c):.{decimales}f} z^-{potencia}"
        terminos.append(f"{signo}{termino}")

    if not terminos:
        return "0"

    texto = "".join(terminos)
    if texto.startswith(" + "):
        texto = texto[3:]
    elif texto.startswith(" - "):
        texto = "-" + texto[3:]
    return texto


def formatear_transferencia(
    numerador,
    denominador,
    retardo_base: int = 0,
    decimales: int = 4,
) -> str:
    """Devuelve G(z) = num/den como un bloque de tres lineas."""
    num = formatear_polinomio(numerador, retardo_base=retardo_base, decimales=decimales)
    den = formatear_polinomio(denominador, monico=True, decimales=decimales)
    separador = "-" * max(len(num), len(den), 20)
    return f"G(z) =\n  {num}\n  {separador}\n  {den}"
