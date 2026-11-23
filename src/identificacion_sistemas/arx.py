"""Estructura ARX:  A(q) y(t) = B(q) u(t-nk) + e(t).

    y(t) + a1 y(t-1) + ... + a_na y(t-na)
        = b1 u(t-nk) + ... + b_nb u(t-nk-nb+1) + e(t)

ARX es la unica estructura de la familia que es LINEAL en los parametros, de
modo que su estimacion se reduce a un problema de minimos cuadrados con
solucion cerrada y unica: sin iteraciones, sin inicializacion y sin riesgo de
minimos locales. Es por eso el metodo mas rapido y robusto, y el punto de
partida natural de los otros tres.

La contrapartida es su hipotesis de ruido, la mas restrictiva de las cuatro:
supone que el ruido entra filtrado por 1/A, es decir que comparte la dinamica
de la planta. Cuando el ruido real no responde a esa hipotesis la estimacion
resulta sesgada, y el metodo tiende a sobrestimar el orden del modelo
incorporando polos que absorben el ruido y que no pertenecen a la planta
(Ljung 1999; Soderstrom y Stoica 1989).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._polinomios import formatear_transferencia
from .base import IdentificadorBase, descarte_por_omision
from .modelos import MOTIVO_DIRECTO, DatosExperimento, ResultadoARX


class IdentificadorARX(IdentificadorBase):
    """Estima un modelo ARX por minimos cuadrados.

        >>> identificador = IdentificadorARX(datos)
        >>> resultado = identificador.estimar(na=2, nb=2, nk=1)
    """

    def estimar(
        self,
        na: int,
        nb: int,
        nk: int = 1,
        remover_media: bool = True,
        descarte: Optional[int] = None,
    ) -> ResultadoARX:
        """Calcula los parametros del modelo ARX.

        Parametros
        ----------
        na : int
            Orden autorregresivo (polos de la planta). >= 1.
        nb : int
            Orden exogeno (ceros de la planta). >= 1.
        nk : int
            Retardo puro en muestras. >= 0.

            Vale la pena recordar que el equivalente discreto por retenedor de
            orden cero de un proceso continuo estrictamente propio tiene
            siempre, al menos, una muestra de retardo: la salida muestreada en
            k*Ts no puede reflejar la entrada aplicada en ese mismo instante.
            De ahi que nk=1 sea el valor habitual incluso para una planta sin
            retardo de transporte.
        remover_media : bool
            Centra u e y antes de estimar. Conviene dejarlo activado: el offset
            de continua contamina la estimacion.
        descarte : int, opcional
            Muestras iniciales que se descartan al calcular el FIT y el AIC.
            Por omision, max(20, N/20).

        Devuelve
        --------
        ResultadoARX

        Excepciones
        -----------
        ValueError
            Si los ordenes son invalidos o no hay muestras suficientes.
        """
        if na < 1:
            raise ValueError(f"na debe ser al menos 1; se recibio {na}.")
        if nb < 1:
            raise ValueError(f"nb debe ser al menos 1; se recibio {nb}.")
        if nk < 0:
            raise ValueError(f"nk no puede ser negativo; se recibio {nk}.")

        u_c, y_c, u_mean, y_mean = self._centrar(remover_media)
        if descarte is None:
            descarte = descarte_por_omision(self.y.size)

        # --- Minimos cuadrados -------------------------------------------
        # np.linalg.lstsq resuelve por SVD, numericamente mas estable que
        # invertir Phi^T Phi de manera directa.
        Phi, Y = self._construir_matriz_regresion(y_c, u_c, na, nb, nk)
        theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)

        A = np.insert(theta[:na], 0, 1.0)   # A(q) es monico: [1, a1, ...]
        B = theta[na:]

        # --- Metricas -----------------------------------------------------
        # La simulacion sirve para graficar; el FIT y el AIC se calculan sobre
        # la prediccion a un paso, que es el predictor optimo de ARX y es lo
        # que hace la herramienta de referencia. La prediccion es ademas
        # robusta en sistemas lentos (polos cerca de z=1), donde la simulacion
        # pura se degrada.
        y_simulada = self._simular(A, B, nk, u_c, y_c, arranque_caliente=True) + y_mean
        y_prediccion = self._predecir_un_paso_arx(A, B, nk, u_c, y_c) + y_mean

        fit = self._calcular_fit(self.y, y_prediccion, descarte)
        fit_simulacion = self._calcular_fit(self.y, y_simulada, descarte)
        n_params = na + nb
        aic = self._calcular_aic(self.y, y_prediccion, n_params, descarte)

        return ResultadoARX(
            A=A,
            B=B,
            theta=theta,
            na=na,
            nb=nb,
            nk=nk,
            fit=fit,
            ganancia=self._ganancia_estatica(B, A),
            y_simulada=y_simulada,
            u_mean=u_mean,
            y_mean=y_mean,
            aic=aic,
            n_params=n_params,
            y_prediccion=y_prediccion,
            fit_simulacion=fit_simulacion,
            descarte=descarte,
            motivo_fin=MOTIVO_DIRECTO,
            convergio=True,
        )

    # ------------------------------------------------------------------
    # Busqueda del retardo
    # ------------------------------------------------------------------

    def buscar_mejor_nk(
        self,
        na: int,
        nb: int,
        nk_inicial: Optional[int] = None,
        ventana: int = 5,
        remover_media: bool = True,
    ) -> tuple:
        """Prueba valores de nk en una ventana y devuelve el de mejor FIT.

        Si no se indica nk_inicial se parte del lag de maxima correlacion
        cruzada entre entrada y salida, que es la estimacion fisica razonable
        del retardo de transporte.

        Devuelve (mejor_nk, mejor_resultado).
        """
        if nk_inicial is None:
            nk_inicial = max(1, self.lag_dominante())

        mejor_fit = -np.inf
        mejor_nk = nk_inicial
        mejor_resultado = None

        for nk in range(max(0, nk_inicial - ventana), nk_inicial + ventana + 1):
            try:
                resultado = self.estimar(na, nb, nk, remover_media)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if resultado.fit > mejor_fit:
                mejor_fit = resultado.fit
                mejor_nk = nk
                mejor_resultado = resultado

        return mejor_nk, mejor_resultado

    # ------------------------------------------------------------------
    # Presentacion
    # ------------------------------------------------------------------

    def obtener_ecuacion_str(self, resultado: ResultadoARX) -> str:
        """Devuelve G(z) = B(z)/A(z) como texto de tres lineas."""
        return formatear_transferencia(
            resultado.B, resultado.A, retardo_base=resultado.nk
        )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    @staticmethod
    def _construir_matriz_regresion(
        y_c: np.ndarray, u_c: np.ndarray, na: int, nb: int, nk: int
    ) -> tuple:
        """Arma la matriz de regresion Phi y el vector Y.

        Cada fila corresponde a un instante t y tiene la forma

            [-y(t-1), ..., -y(t-na), u(t-nk), ..., u(t-nk-nb+1)]

        El signo negativo de la parte autorregresiva viene de la definicion del
        modelo: de A(q)y = B(q)u se despeja
        y(t) = -a1 y(t-1) - ... + b1 u(t-nk) + ...

        La primera fila utilizable es la de t = max(na, nb+nk), que es el primer
        instante con historia completa de ambas senales.
        """
        n = y_c.size
        inicio = max(na, nb + nk)
        n_filas = n - inicio

        if n_filas <= 0:
            raise ValueError(
                f"no hay suficientes muestras para na={na}, nb={nb}, nk={nk}: "
                f"se necesitan mas de {inicio} y el ensayo tiene {n}."
            )

        t_idx = np.arange(inicio, n)
        Phi = np.empty((n_filas, na + nb), dtype=float)
        for j in range(na):
            Phi[:, j] = -y_c[t_idx - 1 - j]
        for j in range(nb):
            Phi[:, na + j] = u_c[t_idx - nk - j]

        return Phi, y_c[t_idx]


def estimar_arx_lineal(
    u_c: np.ndarray, y_c: np.ndarray, na: int, nb: int, nk: int
) -> tuple:
    """Estimacion ARX por minimos cuadrados sobre senales ya centradas.

    Devuelve (A, B). Es la rutina que ARMAX, OE y BJ usan para construir su
    punto de arranque: al ser lineal y de solucion unica, un ARX da siempre un
    punto inicial razonable para la optimizacion no lineal de los demas.
    """
    Phi, Y = IdentificadorARX._construir_matriz_regresion(y_c, u_c, na, nb, nk)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    A = np.concatenate([[1.0], theta[:na]])
    B = theta[na:na + nb]
    return A, B


def residuos_arx(
    A: np.ndarray, B: np.ndarray, u_c: np.ndarray, y_c: np.ndarray, nk: int
) -> np.ndarray:
    """Residuos de prediccion a un paso de un modelo ARX ya estimado.

    Un ARX de orden elevado deja residuos practicamente blancos, y esos
    residuos son la estimacion del ruido con que arranca Hannan-Rissanen.
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    na, nb = A.size - 1, B.size
    n = y_c.size

    e = np.zeros(n)
    inicio = max(na, nb + nk)
    for t in range(inicio, n):
        prediccion = 0.0
        for k in range(1, na + 1):
            prediccion -= A[k] * y_c[t - k]
        for k in range(nb):
            prediccion += B[k] * u_c[t - nk - k]
        e[t] = y_c[t] - prediccion
    return e


def arx(
    u,
    y,
    ts: float = 1.0,
    *,
    na: int = 1,
    nb: int = 1,
    nk: int = 1,
    remover_media: bool = True,
    descarte: Optional[int] = None,
) -> ResultadoARX:
    """Estima un modelo ARX directamente a partir de las senales.

        >>> r = arx(u, y, ts=0.1, na=2, nb=2, nk=1)
        >>> r.A, r.B, r.fit
    """
    datos = DatosExperimento.desde_arrays(u, y, ts)
    return IdentificadorARX(datos).estimar(
        na=na, nb=nb, nk=nk, remover_media=remover_media, descarte=descarte
    )
