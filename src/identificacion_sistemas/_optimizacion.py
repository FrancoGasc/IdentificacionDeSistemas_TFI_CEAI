"""Gauss-Newton amortiguado con busqueda de linea.

Las tres estructuras no lineales -ARMAX, OE y BJ- minimizan la suma de
cuadrados del error de prediccion y comparten el mismo esquema de
optimizacion. Aca vive una sola vez.

Decisiones que importan
-----------------------
*Amortiguamiento de Levenberg-Marquardt.* En cada iteracion se resuelve
(J'J + lam I) delta = -J'e. El parametro lam se reduce cuando el paso mejora el
costo y se aumenta cuando no, lo que da un metodo que se comporta como
Gauss-Newton cerca del optimo y como descenso por gradiente lejos de el
(Marquardt 1963).

*Busqueda de linea sobre una grilla fija de alphas.* En lugar de un algoritmo
de busqueda adaptativo se prueban factores de paso decrecientes y se acepta el
primero que mejora el costo. Es menos elegante y mas predecible: la trayectoria
no depende de tolerancias internas que cada maquina pueda resolver distinto.

*Restriccion de admisibilidad.* El paso se rechaza si lleva los polinomios de
ruido o de planta fuera del circulo unitario. Un modelo de ruido inestable hace
que el predictor recursivo -que filtra por el inverso de esos polinomios-
diverja, produciendo un costo numericamente bajo pero carente de sentido
fisico. Sin esta restriccion la optimizacion puede divergir.

*Implementado a mano, sin scipy.* La razon es la reproducibilidad: el objetivo
era que la misma entrada diera el mismo resultado en Windows y en Linux, con
distintas versiones de BLAS, y coincidiera con la herramienta de referencia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .modelos import (
    MOTIVO_ESTANCAMIENTO,
    MOTIVO_MAX_ITER,
    MOTIVO_SINGULAR,
    MOTIVO_TOLERANCIA,
)

ALPHAS_POR_OMISION = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)


@dataclass
class ResultadoOptimizacion:
    """Salida de una corrida de Gauss-Newton."""

    theta: np.ndarray
    costo: float
    iteraciones: int
    motivo_fin: str

    @property
    def convergio(self) -> bool:
        """True solo si la mejora relativa del costo cayo bajo la tolerancia.

        Un estancamiento -la busqueda de linea sin ningun paso que mejore- no
        es convergencia, aunque el algoritmo se haya detenido igual.
        """
        return self.motivo_fin == MOTIVO_TOLERANCIA


def gauss_newton(
    theta_inicial: np.ndarray,
    evaluar: Callable[[np.ndarray], tuple],
    es_admisible: Callable[[np.ndarray], bool],
    jacobiana: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    lam_inicial: float = 1e-3,
    lam_maximo: float = 1e6,
    alphas: tuple = ALPHAS_POR_OMISION,
) -> ResultadoOptimizacion:
    """Minimiza ||e(theta)||^2 por Gauss-Newton amortiguado.

    Parametros
    ----------
    theta_inicial : np.ndarray
        Punto de arranque.
    evaluar : callable
        evaluar(theta) -> (e, J). Devuelve el vector de errores y, cuando la
        jacobiana se calcula en el mismo paso que el error -caso de ARMAX y OE,
        con jacobiana analitica-, tambien la jacobiana. Si devuelve J=None debe
        suministrarse `jacobiana`.
    es_admisible : callable
        es_admisible(theta) -> bool. Rechaza los pasos que violan la
        restriccion de estabilidad.
    jacobiana : callable, opcional
        jacobiana(theta) -> J, para el caso en que se calcule por separado
        (BJ, con jacobiana numerica).
    tol : float
        Tolerancia de la mejora relativa del costo.

    Devuelve
    --------
    ResultadoOptimizacion
    """
    theta = np.asarray(theta_inicial, dtype=float).copy()
    n_param = theta.size

    e, J = evaluar(theta)
    costo = float(np.sum(e ** 2))
    lam = lam_inicial
    motivo = MOTIVO_MAX_ITER
    iteraciones = 0

    for _ in range(max_iter):
        iteraciones += 1

        if J is None:
            if jacobiana is None:
                raise ValueError(
                    "evaluar() no devolvio jacobiana y no se suministro jacobiana()."
                )
            J = jacobiana(theta)

        try:
            delta = np.linalg.solve(J.T @ J + lam * np.eye(n_param), -J.T @ e)
        except np.linalg.LinAlgError:
            motivo = MOTIVO_SINGULAR
            break

        costo_antes = costo
        mejoro = False

        for alpha in alphas:
            candidato = theta + alpha * delta
            if not es_admisible(candidato):
                continue
            e_test, J_test = evaluar(candidato)
            costo_test = float(np.sum(e_test ** 2))
            if not np.isfinite(costo_test):
                continue
            if costo_test < costo:
                theta = candidato
                e = e_test
                J = J_test
                costo = costo_test
                lam = max(lam * 0.7, 1e-9)
                mejoro = True
                break

        if not mejoro:
            # Ningun paso mejoro: se endurece el amortiguamiento y se reintenta.
            # Si lam llega a su tope, la busqueda esta estancada.
            lam = min(lam * 5.0, lam_maximo)
            if lam >= lam_maximo:
                motivo = MOTIVO_ESTANCAMIENTO
                break
            J = None if jacobiana is not None else J
            continue

        if (costo_antes - costo) / (costo_antes + 1e-15) < tol:
            motivo = MOTIVO_TOLERANCIA
            break

    return ResultadoOptimizacion(
        theta=theta, costo=costo, iteraciones=iteraciones, motivo_fin=motivo
    )
