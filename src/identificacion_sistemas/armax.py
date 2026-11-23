"""Estructura ARMAX:  A(q) y(t) = B(q) u(t-nk) + C(q) e(t).

El polinomio C modela el ruido como una media movil: el ruido comparte el
denominador de la planta pero tiene numerador propio. Es la hipotesis adecuada
cuando la perturbacion entra por el mismo camino fisico que la variable
manipulada -una variacion de la alimentacion, una fluctuacion de la presion de
vapor- y por lo tanto atraviesa la misma dinamica.

Estimacion en dos fases
-----------------------
La presencia de C vuelve el problema no lineal en los parametros, de modo que
no admite solucion cerrada.

1. **Hannan-Rissanen** (1982) para el punto de arranque. Se estima primero un
   ARX de orden ELEVADO, cuyos residuos son una estimacion limpia del ruido;
   con esos residuos como regresores adicionales, un minimos cuadrados
   extendido entrega A, B y C. Usar un ARX de orden alto -y no uno del mismo
   orden que el ARMAX final- es lo que hace estable a este arranque.

2. **Error de prediccion (PEM)** para el refinamiento, minimizando

       min_theta  sum_t e(t, theta)^2
       e(t) = y(t) - yhat(t)
       yhat(t) = -sum a_k y(t-k) + sum b_k u(t-nk-k) + sum c_k e(t-k)

   por Gauss-Newton amortiguado, con jacobiana ANALITICA y arranque multiple.

Referencia: Ljung, L. (1999), 2nd ed., cap. 10.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._optimizacion import gauss_newton
from ._polinomios import es_estable, estabilizar, formatear_polinomio
from .arx import estimar_arx_lineal, residuos_arx
from .base import IdentificadorBase, descarte_por_omision
from .modelos import (
    MOTIVO_DIRECTO,
    MOTIVOS_CONVERGENCIA,
    DatosExperimento,
    ResultadoARMAX,
)

# Valores de arranque para el primer coeficiente de C. El barrido cubre el
# rango util de un modelo de media movil estable sin depender de que una unica
# trayectoria converja igual en todas las plataformas.
C_INICIALES = (-0.3, -0.5, -0.7, -0.85, -0.95)


class IdentificadorARMAX(IdentificadorBase):
    """Estima un modelo ARMAX por Hannan-Rissanen mas refinamiento PEM."""

    def estimar(
        self,
        na: int,
        nb: int,
        nc: int,
        nk: int = 1,
        remover_media: bool = True,
        refinar_pem: bool = True,
        max_iter: int = 200,
        tol: float = 1e-8,
        descarte: Optional[int] = None,
    ) -> ResultadoARMAX:
        """Estima A, B y C.

        Parametros
        ----------
        na, nb : int
            Ordenes de la planta (polos y ceros). >= 1.
        nc : int
            Orden del polinomio de ruido. Con nc=0 el modelo se reduce a un ARX
            y la estimacion es directa, sin iteracion.
        nk : int
            Retardo puro en muestras. >= 0.
        refinar_pem : bool
            Si es False se devuelve la estimacion de Hannan-Rissanen sin
            refinar. Util para comparar ambas etapas; no recomendado en uso
            normal.
        max_iter, tol : int, float
            Limite de iteraciones y tolerancia de la mejora relativa del costo
            en cada corrida de Gauss-Newton.
        descarte : int, opcional
            Muestras iniciales descartadas al calcular el FIT y el AIC.
        """
        if na < 1:
            raise ValueError(f"na debe ser al menos 1; se recibio {na}.")
        if nb < 1:
            raise ValueError(f"nb debe ser al menos 1; se recibio {nb}.")
        if nc < 0:
            raise ValueError(f"nc no puede ser negativo; se recibio {nc}.")
        if nk < 0:
            raise ValueError(f"nk no puede ser negativo; se recibio {nk}.")

        u, y, u_mean, y_mean = self._centrar(remover_media)
        if descarte is None:
            descarte = descarte_por_omision(self.y.size)

        traza: list[str] = []
        iteraciones = 0
        arranques = 0

        if nc == 0:
            # Sin polinomio de ruido esto es un ARX puro: solucion cerrada.
            A, B = estimar_arx_lineal(u, y, na, nb, nk)
            C = np.array([1.0])
            motivo = MOTIVO_DIRECTO
            convergio = True
            traza.append("nc=0: ARX puro, sin refinamiento")
        else:
            # --- Fase 1: Hannan-Rissanen -------------------------------
            orden_alto = min(15, max(2 * max(na, nb, nc), 6))
            A_alto, B_alto = estimar_arx_lineal(u, y, orden_alto, orden_alto, nk)
            e_hat = residuos_arx(A_alto, B_alto, u, y, nk)

            A, B, C = self._estimar_armax_lineal(u, y, e_hat, na, nb, nc, nk)
            C = estabilizar(C)
            motivo = MOTIVO_DIRECTO
            convergio = True
            traza.append(f"Hannan-Rissanen con ARX de orden {orden_alto}")

            # --- Fase 2: refinamiento PEM ------------------------------
            if refinar_pem:
                A, B, C, iteraciones, arranques, motivo, mensajes = self._refinar_pem(
                    A, B, C, u, y, na, nb, nc, nk, max_iter=max_iter, tol=tol
                )
                convergio = motivo in MOTIVOS_CONVERGENCIA
                traza.extend(mensajes)

        # --- Metricas ---------------------------------------------------
        # El FIT y el AIC se calculan con el predictor PROPIO de ARMAX, que
        # incluye el termino en C. Usar aca el predictor de ARX ignoraria el
        # modelo del ruido y sesgaria ambas metricas; es ademas el mismo error
        # que minimiza el refinamiento, de modo que la cifra informada
        # corresponde al criterio que se optimizo.
        y_simulada = self._simular_planta(A, B, nk, u) + y_mean
        y_prediccion = self._predecir_un_paso_armax(A, B, C, nk, u, y) + y_mean

        fit = self._calcular_fit(self.y, y_prediccion, descarte)
        fit_simulacion = self._calcular_fit(self.y, y_simulada, descarte)
        n_params = na + nb + nc
        aic = self._calcular_aic(self.y, y_prediccion, n_params, descarte)

        theta = np.concatenate([A[1:], B, C[1:]]) if C.size > 1 else np.concatenate([A[1:], B])

        return ResultadoARMAX(
            A=A,
            B=B,
            C=C,
            theta=theta,
            na=na,
            nb=nb,
            nc=nc,
            nk=nk,
            fit=fit,
            ganancia=self._ganancia_estatica(B, A),
            y_simulada=y_simulada,
            u_mean=u_mean,
            y_mean=y_mean,
            iteraciones=iteraciones,
            arranques=arranques,
            convergio=convergio,
            motivo_fin=motivo,
            aic=aic,
            n_params=n_params,
            debug=traza,
            y_prediccion=y_prediccion,
            fit_simulacion=fit_simulacion,
            descarte=descarte,
        )

    # ------------------------------------------------------------------
    # Fase 1: estimacion lineal extendida
    # ------------------------------------------------------------------

    @staticmethod
    def _estimar_armax_lineal(
        u: np.ndarray,
        y: np.ndarray,
        e_hat: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
    ) -> tuple:
        """Minimos cuadrados de A, B y C tratando e_hat como observado.

        Es el corazon de Hannan-Rissanen: con el ruido estimado de un ARX de
        orden alto, el problema vuelve a ser lineal en los parametros.
        """
        n = y.size
        inicio = max(na, nb + nk, nc)
        n_filas = n - inicio
        if n_filas <= 0:
            raise ValueError(
                f"no hay suficientes muestras para na={na}, nb={nb}, nc={nc}, nk={nk}: "
                f"se necesitan mas de {inicio} y el ensayo tiene {n}."
            )

        t_idx = np.arange(inicio, n)
        Phi = np.empty((n_filas, na + nb + nc), dtype=float)
        col = 0
        for k in range(1, na + 1):
            Phi[:, col] = -y[t_idx - k]
            col += 1
        for k in range(nb):
            Phi[:, col] = u[t_idx - nk - k]
            col += 1
        for k in range(1, nc + 1):
            Phi[:, col] = e_hat[t_idx - k]
            col += 1

        theta, *_ = np.linalg.lstsq(Phi, y[t_idx], rcond=None)
        A = np.concatenate([[1.0], theta[:na]])
        B = theta[na:na + nb]
        C = np.concatenate([[1.0], theta[na + nb:na + nb + nc]])
        return A, B, C

    # ------------------------------------------------------------------
    # Fase 2: refinamiento por error de prediccion
    # ------------------------------------------------------------------

    def _refinar_pem(
        self,
        A0: np.ndarray,
        B0: np.ndarray,
        C0: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iter: int,
        tol: float,
    ) -> tuple:
        """Minimiza el error de prediccion ARMAX desde varios puntos de arranque.

        Se mantienen A y B de Hannan-Rissanen -que ya son buenos- y se varia
        solo el primer coeficiente de C. El arranque multiple cubre el espacio
        de C de manera robusta sin depender de que una sola trayectoria
        converja igual en todas las maquinas.
        """
        n = y.size
        inicio = max(na, nb + nk, nc)
        n_param = na + nb + nc

        if n - inicio <= 0:
            raise ValueError(
                f"no hay suficientes muestras para refinar con na={na}, nb={nb}, "
                f"nc={nc}, nk={nk}."
            )

        def evaluar(th: np.ndarray) -> tuple:
            """Error de prediccion y jacobiana ANALITICA, en una sola pasada.

            Las derivadas del error ARMAX tienen forma cerrada (Ljung 1999,
            ec. 10.50): son las senales de regresion filtradas por 1/C,

                de/da_k =  y(t-k)    filtrado por 1/C
                de/db_k = -u(t-nk-k) filtrado por 1/C
                de/dc_k = -e(t-k)    filtrado por 1/C

            donde el filtrado es la recursion psi(t) = s(t) - sum c_i psi(t-i).

            Usar la jacobiana analitica en lugar de diferencias finitas elimina
            la dependencia del paso eps y del redondeo de la biblioteca de
            algebra lineal. Durante el desarrollo, con jacobiana numerica las
            diferencias de redondeo entre LAPACK en Windows y en Linux se
            amplificaban a lo largo de las iteraciones y los resultados no
            coincidian entre plataformas.
            """
            a = th[:na]
            b = th[na:na + nb]
            c = th[na + nb:na + nb + nc]
            A = np.concatenate([[1.0], a])
            C = np.concatenate([[1.0], c])

            e = np.zeros(n)
            J = np.zeros((n, n_param))

            for t in range(inicio, n):
                yhat = 0.0
                for k in range(1, na + 1):
                    yhat -= A[k] * y[t - k]
                for k in range(nb):
                    yhat += b[k] * u[t - nk - k]
                for k in range(1, nc + 1):
                    yhat += C[k] * e[t - k]
                e[t] = y[t] - yhat

                col = 0
                for k in range(1, na + 1):
                    filtro = 0.0
                    for i in range(1, nc + 1):
                        if t - i >= 0:
                            filtro -= C[i] * J[t - i, col]
                    J[t, col] = y[t - k] + filtro
                    col += 1
                for k in range(nb):
                    filtro = 0.0
                    for i in range(1, nc + 1):
                        if t - i >= 0:
                            filtro -= C[i] * J[t - i, col]
                    J[t, col] = -u[t - nk - k] + filtro
                    col += 1
                for k in range(1, nc + 1):
                    filtro = 0.0
                    for i in range(1, nc + 1):
                        if t - i >= 0:
                            filtro -= C[i] * J[t - i, col]
                    J[t, col] = -e[t - k] + filtro
                    col += 1

            return e[inicio:], J[inicio:]

        def admisible(th: np.ndarray) -> bool:
            # Solo C queda restringido: el predictor filtra por 1/C, de modo
            # que un C inestable lo hace divergir. A no interviene en ese
            # filtrado y puede, legitimamente, describir una planta inestable.
            return es_estable(th[na + nb:na + nb + nc])

        theta_hr = np.concatenate([A0[1:], B0, C0[1:] if C0.size > 1 else []])
        e_hr, _ = evaluar(theta_hr)
        costo_hr = float(np.sum(e_hr ** 2))

        mejor_theta = theta_hr
        mejor_costo = costo_hr
        mejor_motivo = MOTIVO_DIRECTO
        iteraciones = 0
        arranques = 0

        mensajes = [f"HR inicial: costo={costo_hr:.6f}"]

        candidatos = list(C_INICIALES) + [float(C0[1]) if C0.size > 1 else -0.5]
        for c_inicial in candidatos:
            theta_inicial = theta_hr.copy()
            theta_inicial[na + nb] = c_inicial
            if nc > 1:
                theta_inicial[na + nb + 1:na + nb + nc] = 0.0

            resultado = gauss_newton(
                theta_inicial,
                evaluar=evaluar,
                es_admisible=admisible,
                max_iter=max_iter,
                tol=tol,
            )
            arranques += 1
            iteraciones += resultado.iteraciones

            marca = " *" if resultado.costo < mejor_costo else ""
            mensajes.append(
                f"  arranque C={c_inicial:+.2f} -> costo={resultado.costo:.6f} "
                f"({resultado.motivo_fin}){marca}"
            )

            if resultado.costo < mejor_costo:
                mejor_costo = resultado.costo
                mejor_theta = resultado.theta
                mejor_motivo = resultado.motivo_fin

        mensajes.append(f"mejor: costo={mejor_costo:.6f} ({mejor_motivo})")

        A = np.concatenate([[1.0], mejor_theta[:na]])
        B = mejor_theta[na:na + nb]
        C = estabilizar(np.concatenate([[1.0], mejor_theta[na + nb:na + nb + nc]]))

        return A, B, C, iteraciones, arranques, mejor_motivo, mensajes

    # ------------------------------------------------------------------
    # Busqueda del retardo y presentacion
    # ------------------------------------------------------------------

    def buscar_mejor_nk(
        self,
        na: int,
        nb: int,
        nc: int,
        nk_inicial: Optional[int] = None,
        ventana: int = 5,
        remover_media: bool = True,
    ) -> tuple:
        """Busca el nk de mejor FIT alrededor del lag dominante.

        Devuelve (mejor_resultado, lag_correlacion, mejor_nk).
        """
        lag = self.lag_dominante()
        if nk_inicial is None:
            nk_inicial = lag

        mejor_fit = -np.inf
        mejor_resultado = None
        mejor_nk = nk_inicial

        for nk in range(max(0, lag - ventana), lag + ventana + 1):
            try:
                r = self.estimar(na, nb, nc, nk, remover_media)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if r.fit > mejor_fit:
                mejor_fit = r.fit
                mejor_resultado = r
                mejor_nk = nk

        return mejor_resultado, lag, mejor_nk

    def obtener_ecuacion_str(self, resultado: ResultadoARMAX) -> str:
        """Devuelve los tres polinomios como texto."""
        return (
            f"A(q) = {formatear_polinomio(resultado.A, monico=True)}\n"
            f"B(q) = {formatear_polinomio(resultado.B, retardo_base=resultado.nk)}\n"
            f"C(q) = {formatear_polinomio(resultado.C, monico=True)}"
        )


def armax(
    u,
    y,
    ts: float = 1.0,
    *,
    na: int = 1,
    nb: int = 1,
    nc: int = 1,
    nk: int = 1,
    remover_media: bool = True,
    descarte: Optional[int] = None,
) -> ResultadoARMAX:
    """Estima un modelo ARMAX directamente a partir de las senales."""
    datos = DatosExperimento.desde_arrays(u, y, ts)
    return IdentificadorARMAX(datos).estimar(
        na=na, nb=nb, nc=nc, nk=nk, remover_media=remover_media, descarte=descarte
    )
