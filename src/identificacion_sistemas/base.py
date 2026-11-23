"""Clase base de los identificadores y metricas comunes.

Las cuatro estructuras comparten mas de lo que las diferencia: todas reciben un
DatosExperimento, todas devuelven polinomios con un FIT y un AIC, y todas
pueden simular el modelo y calcular la correlacion cruzada entre entrada y
salida. Lo que cambia es COMO estiman los polinomios, y eso es lo unico que
cada subclase implementa.

Predictores
-----------
Cada estructura tiene su propio predictor optimo a un paso, determinado por la
hipotesis de ruido que supone:

    ARX     yhat = (1-A) y + B u(t-nk)
    ARMAX   yhat = (1-A) y + B u(t-nk) + (C-1) e
    OE      yhat = (B/F) u(t-nk)                     (es la simulacion pura)
    BJ      yhat = y - (D/C) [ y - (B/F) u(t-nk) ]

Evaluar las cuatro con un mismo predictor invalida la comparacion por FIT y por
AIC: el AIC solo es una estimacion de la verosimilitud cuando la varianza del
error se calcula con el predictor propio de la estructura. Esta libreria usa
siempre el predictor que corresponde.

Referencia: Ljung, L. (1999), "System Identification: Theory for the User",
2nd ed., cap. 10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ._polinomios import ventana_pasada
from .modelos import DatosExperimento


def descarte_por_omision(n: int) -> int:
    """Muestras iniciales que se descartan al calcular el FIT y el AIC.

    El transitorio de arranque del predictor no describe al modelo, de modo que
    contaminaria la metrica. La regla es max(20, n/20): con 1500 muestras
    descarta 75.

    Conviene tener presente que este valor es una convencion y no una
    definicion: la System Identification Toolbox de Matlab no fija la ventana
    de la misma manera, y parte de las diferencias residuales de FIT entre
    ambas implementaciones proviene de ahi. Por eso todos los metodos admiten
    el parametro `descarte`, y el valor usado queda registrado en el resultado.
    """
    return max(20, int(n) // 20)


class IdentificadorBase(ABC):
    """Base abstracta de los cuatro identificadores."""

    def __init__(self, datos: DatosExperimento):
        if not isinstance(datos, DatosExperimento):
            raise TypeError(
                "se esperaba un DatosExperimento; usar "
                "DatosExperimento.desde_arrays(u, y, ts) para construirlo."
            )
        self.datos = datos
        self.u = datos.u
        self.y = datos.y
        self.t = datos.t
        self.ts = datos.ts

    # ------------------------------------------------------------------
    # Interfaz que cada estructura implementa
    # ------------------------------------------------------------------

    @abstractmethod
    def estimar(self, *args, **kwargs):
        """Estima los polinomios del modelo."""

    @abstractmethod
    def buscar_mejor_nk(self, *args, **kwargs):
        """Busca el retardo nk que maximiza el FIT."""

    @abstractmethod
    def obtener_ecuacion_str(self, resultado) -> str:
        """Devuelve el modelo como texto legible."""

    # ------------------------------------------------------------------
    # Preprocesamiento
    # ------------------------------------------------------------------

    def _centrar(self, remover_media: bool) -> tuple:
        """Devuelve (u_centrada, y_centrada, u_mean, y_mean)."""
        if remover_media:
            u_mean = float(np.mean(self.u))
            y_mean = float(np.mean(self.y))
        else:
            u_mean = 0.0
            y_mean = 0.0
        return self.u - u_mean, self.y - y_mean, u_mean, y_mean

    # ------------------------------------------------------------------
    # Analisis de la senal
    # ------------------------------------------------------------------

    def calcular_correlacion(self, max_lag: Optional[int] = None) -> tuple:
        """Correlacion cruzada normalizada entre u e y, para lags >= 0.

        Es la base sobre la que se decide el retardo nk: el lag del maximo en
        valor absoluto es la primera estimacion razonable.

        Devuelve (lags, correlacion).
        """
        n = self.u.size
        if max_lag is None:
            max_lag = min(n - 1, max(30, n // 2))

        def normalizar(x: np.ndarray) -> np.ndarray:
            desv = np.std(x)
            return (x - np.mean(x)) / desv if desv > 1e-10 else x - np.mean(x)

        corr = np.correlate(normalizar(self.y), normalizar(self.u), mode="full")
        lags = np.arange(-(n - 1), n)
        mascara = (lags >= 0) & (lags <= max_lag)
        return lags[mascara], corr[mascara] / n

    def lag_dominante(self, max_lag: Optional[int] = None) -> int:
        """Lag de maxima correlacion cruzada en valor absoluto."""
        lags, corr = self.calcular_correlacion(max_lag)
        return int(lags[int(np.argmax(np.abs(corr)))])

    # ------------------------------------------------------------------
    # Simulacion y prediccion
    # ------------------------------------------------------------------

    @staticmethod
    def _simular_planta(
        A, B, nk: int, u: np.ndarray, semilla: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Simula y(t) = [B(q)/A(q)] u(t-nk) alimentando solo la entrada.

        El modelo se realimenta con sus propias salidas pasadas, de modo que el
        error se acumula a lo largo del registro. Sirve para A/B (ARX, ARMAX) y
        para F/B (OE, BJ), que tienen la misma forma.

        `semilla` es el arranque en caliente: las primeras muestras se toman de
        la salida real en lugar de arrancar en cero, lo que evita el transitorio
        artificial del comienzo. Importa que esos valores se carguen ANTES de la
        recursion, porque alimentan los primeros terminos autorregresivos;
        escribirlos despues cambia la curva simulada.
        """
        A = np.asarray(A, dtype=float).ravel()
        B = np.asarray(B, dtype=float).ravel()
        na = A.size - 1
        nb = B.size
        n = u.size

        y_sim = np.zeros(n)
        inicio = max(na, nb + nk)

        if semilla is not None and n > inicio:
            y_sim[:inicio] = np.asarray(semilla, dtype=float).ravel()[:inicio]

        for t in range(inicio, n):
            valor = float(np.dot(B, ventana_pasada(u, t - nk + 1, nb)))
            if na > 0:
                valor -= float(np.dot(A[1:], ventana_pasada(y_sim, t, na)))
            y_sim[t] = valor
        return y_sim

    def _simular(
        self,
        A,
        B,
        nk: int,
        u_centrada: np.ndarray,
        y_centrada: Optional[np.ndarray] = None,
        arranque_caliente: bool = False,
    ) -> np.ndarray:
        """Simulacion pura del modelo A(q) y = B(q) u(t-nk).

        Con arranque_caliente=True las primeras muestras se siembran con la
        salida real. Requiere y_centrada.
        """
        if arranque_caliente and y_centrada is None:
            raise ValueError("el arranque caliente necesita y_centrada.")
        semilla = y_centrada if arranque_caliente else None
        return self._simular_planta(A, B, nk, u_centrada, semilla=semilla)

    def _predecir_un_paso_arx(
        self, A, B, nk: int, u_centrada: np.ndarray, y_centrada: np.ndarray
    ) -> np.ndarray:
        """Predictor optimo a un paso de ARX.

        A diferencia de la simulacion, los terminos autorregresivos usan los
        valores REALES de y, de modo que el predictor se reancla al dato en
        cada paso y no arrastra el error. Es el predictor con el que la
        System Identification Toolbox calcula el FIT por omision.
        """
        A = np.asarray(A, dtype=float).ravel()
        B = np.asarray(B, dtype=float).ravel()
        na = A.size - 1
        nb = B.size
        n = u_centrada.size

        yhat = np.zeros(n)
        inicio = max(na, nb + nk)
        for t in range(inicio, n):
            valor = float(np.dot(B, ventana_pasada(u_centrada, t - nk + 1, nb)))
            if na > 0:
                valor -= float(np.dot(A[1:], ventana_pasada(y_centrada, t, na)))
            yhat[t] = valor
        return yhat

    def _predecir_un_paso_armax(
        self, A, B, C, nk: int, u_centrada: np.ndarray, y_centrada: np.ndarray
    ) -> np.ndarray:
        """Predictor optimo a un paso de ARMAX.

            yhat(t) = -sum a_k y(t-k) + sum b_k u(t-nk-k) + sum c_k e(t-k)
            e(t)    = y(t) - yhat(t)

        El termino en C usa los errores de prediccion pasados, que es lo que
        permite anticipar la parte coloreada del ruido. Con C(q) = 1 se reduce
        exactamente al predictor de ARX.
        """
        A = np.asarray(A, dtype=float).ravel()
        B = np.asarray(B, dtype=float).ravel()
        C = np.asarray(C, dtype=float).ravel()
        na, nb, nc = A.size - 1, B.size, C.size - 1
        n = u_centrada.size

        yhat = np.zeros(n)
        e = np.zeros(n)
        inicio = max(na, nb + nk, nc)

        for t in range(inicio, n):
            valor = 0.0
            for k in range(1, na + 1):
                valor -= A[k] * y_centrada[t - k]
            for k in range(nb):
                valor += B[k] * u_centrada[t - nk - k]
            for k in range(1, nc + 1):
                valor += C[k] * e[t - k]
            yhat[t] = valor
            e[t] = y_centrada[t] - valor
        return yhat

    def _predecir_un_paso_bj(
        self, B, F, C, D, nk: int, u_centrada: np.ndarray, y_centrada: np.ndarray
    ) -> np.ndarray:
        """Predictor optimo a un paso de Box-Jenkins.

        Tres pasos: la salida de la planta (B/F), el residuo de planta
        w = y - y_planta, y el filtrado de w por D/C que entrega el error de
        prediccion. Con C = D = 1 el filtrado es la identidad y el predictor se
        reduce a la simulacion B/F, que es el de OE: la familia es coherente.
        """
        B = np.asarray(B, dtype=float).ravel()
        F = np.asarray(F, dtype=float).ravel()
        C = np.asarray(C, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()
        nb, nf, nc, nd = B.size, F.size - 1, C.size - 1, D.size - 1
        n = u_centrada.size

        y_planta = self._simular_planta(B=B, A=F, nk=nk, u=u_centrada)
        w = y_centrada - y_planta

        e = np.zeros(n)
        inicio = max(nb + nk, nf, nc, nd)
        for t in range(inicio, n):
            valor = w[t]
            for k in range(1, nd + 1):
                valor += D[k] * w[t - k]
            for k in range(1, nc + 1):
                valor -= C[k] * e[t - k]
            e[t] = valor

        yhat = np.zeros(n)
        yhat[inicio:] = y_centrada[inicio:] - e[inicio:]
        return yhat

    # ------------------------------------------------------------------
    # Metricas
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_fit(y_real, y_estimada, descarte: Optional[int] = None) -> float:
        """FIT (%) entre la salida medida y la estimada por el modelo.

            FIT = 100 (1 - ||y - yhat|| / ||y - mean(y)||)

        100 % es ajuste perfecto, 0 % equivale a predecir la media y un valor
        negativo es peor que la media. Al estar normalizado, permite comparar
        plantas de distinta escala.
        """
        y_real = np.asarray(y_real, dtype=float).ravel()
        y_estimada = np.asarray(y_estimada, dtype=float).ravel()

        if descarte is None:
            descarte = descarte_por_omision(y_real.size)
        y_real = y_real[descarte:]
        y_estimada = y_estimada[descarte:]

        if y_real.size == 0:
            return 0.0

        norma_error = np.linalg.norm(y_real - y_estimada)
        norma_base = np.linalg.norm(y_real - np.mean(y_real))
        if norma_base < 1e-10:
            return 0.0
        return float(100.0 * (1.0 - norma_error / norma_base))

    @staticmethod
    def _calcular_aic(
        y_real, y_estimada, n_params: int, descarte: Optional[int] = None
    ) -> float:
        """Criterio de informacion de Akaike:  AIC = N ln(V) + 2d.

        V es la varianza del error de prediccion y d el numero de parametros
        libres. El primer termino premia el ajuste y el segundo penaliza la
        complejidad; el mejor modelo es el de AIC MINIMO.

        El AIC es relativo: su valor absoluto no significa nada y solo es licito
        comparar valores obtenidos sobre los mismos datos. Para que la
        comparacion entre estructuras sea valida, y_estimada debe provenir del
        predictor optimo de cada estructura.
        """
        y_real = np.asarray(y_real, dtype=float).ravel()
        y_estimada = np.asarray(y_estimada, dtype=float).ravel()

        if descarte is None:
            descarte = descarte_por_omision(y_real.size)
        y_real = y_real[descarte:]
        y_estimada = y_estimada[descarte:]

        n = y_real.size
        if n <= n_params:
            return float("inf")

        error = y_real - y_estimada
        varianza = float(np.sum(error ** 2) / n)
        varianza = max(varianza, 1e-12)
        return float(n * np.log(varianza) + 2.0 * n_params)

    # ------------------------------------------------------------------
    # Auxiliares
    # ------------------------------------------------------------------

    @staticmethod
    def _ganancia_estatica(numerador, denominador) -> float:
        """Ganancia estatica K = G(1) = sum(B) / sum(A).

        Advertencia: el denominador es una suma de coeficientes de signo
        alterno y magnitud proxima a la unidad, de modo que puede resultar
        varios ordenes de magnitud menor que sus terminos. Cuando eso ocurre
        -tipicamente con polos cerca de z=1, es decir con un periodo de
        muestreo muy fino frente a la dinamica del proceso- un error pequeno
        en los coeficientes se amplifica enormemente en K. Un sum(A) del orden
        de 1e-3 es senal de que la ganancia y las constantes de tiempo
        derivadas del modelo deben tomarse con reserva.
        """
        suma_den = float(np.sum(denominador))
        if abs(suma_den) <= 1e-10:
            return 0.0
        return float(np.sum(numerador)) / suma_den

    def _validar_minimo_muestras(self, inicio: int, etiqueta: str) -> None:
        if self.y.size <= inicio:
            raise ValueError(
                f"no hay suficientes muestras para {etiqueta}: se necesitan al menos "
                f"{inicio + 1} y el ensayo tiene {self.y.size}."
            )
