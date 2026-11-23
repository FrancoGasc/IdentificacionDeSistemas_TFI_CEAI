"""Estructura Box-Jenkins:  y(t) = [B(q)/F(q)] u(t-nk) + [C(q)/D(q)] e(t).

Es el caso general de la familia: la planta (B/F) y el ruido (C/D) tienen
denominadores INDEPENDIENTES, no comparten polos. Eso es lo que la distingue de
ARMAX, donde el ruido hereda el denominador A de la planta.

Al suponer menos que las otras tres, es la mas cara de estimar: cuatro
polinomios, inicializacion en varias etapas y refinamiento conjunto.

Predictor y error de prediccion
-------------------------------
    e(t) = [D(q)/C(q)] [ y(t) - (B(q)/F(q)) u(t-nk) ]

Primero se calcula el residuo de la planta -como en OE- y despues se lo filtra
por D/C, el inverso del modelo de ruido. Con C = D = 1 el filtrado es la
identidad y el predictor se reduce al de OE.

Inicializacion multietapa
-------------------------
1. Estimar la planta B/F por OE, ignorando el modelo de ruido.
2. Calcular los residuos de esa planta, v(t) = y(t) - (B/F) u(t-nk).
3. Modelar esos residuos con un ARMA: D(q) v(t) = C(q) e(t), lo que entrega
   C y D.
4. Refinar los cuatro polinomios en conjunto por Gauss-Newton.

Esta jerarquia tiene una consecuencia observable: el esfuerzo se concentra en
el modelo de la planta y el error residual se acumula preferentemente en C y D.
Para el proposito habitual -obtener un modelo de la planta- es la jerarquia
deseable: el modelo de ruido existe para blanquear el residuo y evitar que la
estimacion de la planta quede sesgada, no como producto final.

A diferencia de ARMAX y OE, aca la jacobiana se calcula de forma NUMERICA: la
complejidad de derivar analiticamente cuatro polinomios acoplados hace que la
aproximacion numerica resulte mas robusta y simple de mantener, a costa de un
mayor tiempo de calculo.

Referencia: Ljung, L. (1999), 2nd ed., cap. 10.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._optimizacion import gauss_newton
from ._polinomios import es_estable, estabilizar, formatear_polinomio
from .arx import estimar_arx_lineal
from .base import IdentificadorBase, descarte_por_omision
from .modelos import MOTIVOS_CONVERGENCIA, DatosExperimento, ResultadoBJ
from .oe import construir_evaluador_oe

MAX_ITER_PLANTA = 100
MAX_ITER_CONJUNTO = 150
EPS_JACOBIANA = 1e-6


class IdentificadorBJ(IdentificadorBase):
    """Estima un modelo Box-Jenkins por error de prediccion."""

    def estimar(
        self,
        nb: int,
        nc: int,
        nd: int,
        nf: int,
        nk: int = 1,
        remover_media: bool = True,
        max_iter: int = MAX_ITER_CONJUNTO,
        tol: float = 1e-8,
        descarte: Optional[int] = None,
    ) -> ResultadoBJ:
        """Estima B, C, D y F.

        Parametros
        ----------
        nb : int
            Orden de B (ceros de la planta). >= 1.
        nc : int
            Orden de C (numerador del ruido). >= 0.
        nd : int
            Orden de D (denominador del ruido). >= 0.
        nf : int
            Orden de F (polos de la planta). >= 1.
        nk : int
            Retardo puro en muestras. >= 0.

        Notas
        -----
        El orden de los parametros es (nb, nc, nd, nf, nk), alfabetico por
        polinomio, no (nb, nf, nc, nd). Conviene pasarlos por nombre.
        """
        if nb < 1:
            raise ValueError(f"nb debe ser al menos 1; se recibio {nb}.")
        if nf < 1:
            raise ValueError(f"nf debe ser al menos 1; se recibio {nf}.")
        if nc < 0 or nd < 0:
            raise ValueError(f"nc y nd no pueden ser negativos; se recibio nc={nc}, nd={nd}.")
        if nk < 0:
            raise ValueError(f"nk no puede ser negativo; se recibio {nk}.")

        u, y, u_mean, y_mean = self._centrar(remover_media)
        n = y.size
        if descarte is None:
            descarte = descarte_por_omision(self.y.size)

        # --- Etapa 1: la planta B/F por OE ------------------------------
        B0, F0 = self._estimar_planta_oe(u, y, nb, nf, nk)

        # --- Etapa 2: residuos de la planta -----------------------------
        v = y - self._simular_planta(F0, B0, nk, u)

        # --- Etapa 3: el ruido v modelado como ARMA (C/D) ---------------
        if nc > 0 or nd > 0:
            C0, D0 = self._estimar_arma(v, nc, nd)
        else:
            C0 = np.array([1.0])
            D0 = np.array([1.0])

        # --- Etapa 4: refinamiento conjunto -----------------------------
        B, F, C, D, iteraciones, motivo = self._refinar_conjunto(
            B0, F0, C0, D0, u, y, nb, nc, nd, nf, nk, max_iter=max_iter, tol=tol
        )

        # --- Metricas ---------------------------------------------------
        y_simulada = self._simular_planta(F, B, nk, u) + y_mean
        y_pred = self._predecir_un_paso_bj(B, F, C, D, nk, u, y)
        y_prediccion = y_pred + y_mean

        fit = self._calcular_fit(self.y, y_prediccion, descarte)
        fit_simulacion = self._calcular_fit(self.y, y_simulada, descarte)
        n_params = nb + nc + nd + nf
        aic = self._calcular_aic(self.y, y_prediccion, n_params, descarte)

        theta = np.concatenate(
            [B, F[1:], C[1:] if C.size > 1 else [], D[1:] if D.size > 1 else []]
        )

        return ResultadoBJ(
            B=B,
            F=F,
            C=C,
            D=D,
            theta=theta,
            nb=nb,
            nc=nc,
            nd=nd,
            nf=nf,
            nk=nk,
            fit=fit,
            ganancia=self._ganancia_estatica(B, F),
            y_simulada=y_simulada,
            u_mean=u_mean,
            y_mean=y_mean,
            iteraciones=iteraciones,
            arranques=1,
            convergio=motivo in MOTIVOS_CONVERGENCIA,
            motivo_fin=motivo,
            aic=aic,
            n_params=n_params,
            y_prediccion=y_prediccion,
            fit_simulacion=fit_simulacion,
            descarte=descarte,
        )

    # ------------------------------------------------------------------
    # Etapa 1
    # ------------------------------------------------------------------

    @staticmethod
    def _estimar_planta_oe(
        u: np.ndarray, y: np.ndarray, nb: int, nf: int, nk: int
    ) -> tuple:
        """Estima la planta B/F por OE, sin modelar el ruido.

        Un solo arranque -el ARX inicial ya es un punto razonable- porque el
        modelo de ruido se incorpora despues y el refinamiento conjunto de la
        etapa 4 corrige lo que quede.
        """
        F_arx, B_arx = estimar_arx_lineal(u, y, nf, nb, nk)

        theta = np.concatenate([B_arx, F_arx[1:]])
        if not es_estable(theta[nb:nb + nf]):
            theta[nb:nb + nf] = 0.0

        evaluar, admisible = construir_evaluador_oe(u, y, nb, nf, nk)
        resultado = gauss_newton(
            theta,
            evaluar=evaluar,
            es_admisible=admisible,
            max_iter=MAX_ITER_PLANTA,
        )

        B = resultado.theta[:nb]
        F = estabilizar(np.concatenate([[1.0], resultado.theta[nb:nb + nf]]))
        return B, F

    # ------------------------------------------------------------------
    # Etapa 3
    # ------------------------------------------------------------------

    @staticmethod
    def _estimar_arma(v: np.ndarray, nc: int, nd: int) -> tuple:
        """Modela los residuos como D(q) v(t) = C(q) e(t).

        Un ARMA es un ARMAX sin entrada, y se estima por el mismo
        Hannan-Rissanen simplificado: un AR de orden alto entrega residuos
        practicamente blancos, y con ellos la regresion de v sobre sus propios
        pasados (para D) y sobre esos residuos (para C) vuelve a ser lineal.

        En forma de regresion:
            v(t) = -sum d_k v(t-k) + sum c_k e(t-k) + e(t)
        """
        n = v.size
        if nc == 0 and nd == 0:
            return np.array([1.0]), np.array([1.0])

        # Paso 1: AR de orden alto para estimar el ruido blanco.
        orden_alto = min(15, max(2 * max(nc, nd), 6))
        if n - orden_alto <= 0:
            raise ValueError(
                f"no hay suficientes muestras para modelar el ruido: se necesitan "
                f"mas de {orden_alto} y hay {n}."
            )

        t_idx = np.arange(orden_alto, n)
        Phi = np.empty((t_idx.size, orden_alto), dtype=float)
        for k in range(1, orden_alto + 1):
            Phi[:, k - 1] = -v[t_idx - k]
        theta_ar, *_ = np.linalg.lstsq(Phi, v[t_idx], rcond=None)
        A_alto = np.concatenate([[1.0], theta_ar])

        e_hat = np.zeros(n)
        for t in range(orden_alto, n):
            prediccion = 0.0
            for k in range(1, orden_alto + 1):
                prediccion -= A_alto[k] * v[t - k]
            e_hat[t] = v[t] - prediccion

        # Paso 2: regresion lineal para C y D.
        inicio = max(nc, nd)
        t_idx = np.arange(inicio, n)
        Phi = np.empty((t_idx.size, nd + nc), dtype=float)
        col = 0
        for k in range(1, nd + 1):
            Phi[:, col] = -v[t_idx - k]
            col += 1
        for k in range(1, nc + 1):
            Phi[:, col] = e_hat[t_idx - k]
            col += 1

        theta, *_ = np.linalg.lstsq(Phi, v[t_idx], rcond=None)
        D = np.concatenate([[1.0], theta[:nd]]) if nd > 0 else np.array([1.0])
        C = np.concatenate([[1.0], theta[nd:nd + nc]]) if nc > 0 else np.array([1.0])

        return estabilizar(C), estabilizar(D)

    # ------------------------------------------------------------------
    # Etapa 4
    # ------------------------------------------------------------------

    def _refinar_conjunto(
        self,
        B0: np.ndarray,
        F0: np.ndarray,
        C0: np.ndarray,
        D0: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        nb: int,
        nc: int,
        nd: int,
        nf: int,
        nk: int,
        max_iter: int,
        tol: float,
    ) -> tuple:
        """Refina B, F, C y D minimizando el error de prediccion de BJ."""
        n = y.size
        n_param = nb + nf + nc + nd
        inicio = max(nb + nk, nf, nc, nd)

        if n - inicio <= 0:
            raise ValueError(
                f"no hay suficientes muestras para nb={nb}, nf={nf}, nc={nc}, "
                f"nd={nd}, nk={nk}."
            )

        def desempaquetar(th: np.ndarray) -> tuple:
            i = 0
            b = th[i:i + nb]; i += nb
            f = th[i:i + nf]; i += nf
            c = th[i:i + nc]; i += nc
            d = th[i:i + nd]
            B = b
            F = np.concatenate([[1.0], f])
            C = np.concatenate([[1.0], c]) if nc > 0 else np.array([1.0])
            D = np.concatenate([[1.0], d]) if nd > 0 else np.array([1.0])
            return B, F, C, D

        def calcular_error(th: np.ndarray) -> np.ndarray:
            B, F, C, D = desempaquetar(th)
            # Residuo de la planta, como en OE.
            w = y - self._simular_planta(F, B, nk, u)
            # Filtrado por D/C: de D w = C e se despeja
            # e(t) = D(q) w(t) - sum c_k e(t-k).
            e = np.zeros(n)
            for t in range(inicio, n):
                valor = w[t]
                for k in range(1, nd + 1):
                    valor += D[k] * w[t - k]
                for k in range(1, nc + 1):
                    valor -= C[k] * e[t - k]
                e[t] = valor
            return e[inicio:]

        m = n - inicio

        def jacobiana(th: np.ndarray) -> np.ndarray:
            """Diferencias finitas centradas.

            Derivar analiticamente cuatro polinomios acoplados es posible pero
            fragil de mantener; la aproximacion numerica es mas simple y, al
            usar diferencias centradas, suficientemente precisa para el paso de
            Gauss-Newton.
            """
            J = np.empty((m, n_param), dtype=float)
            for j in range(n_param):
                mas = th.copy(); mas[j] += EPS_JACOBIANA
                menos = th.copy(); menos[j] -= EPS_JACOBIANA
                J[:, j] = (calcular_error(mas) - calcular_error(menos)) / (2 * EPS_JACOBIANA)
            return J

        def evaluar(th: np.ndarray) -> tuple:
            return calcular_error(th), None

        def admisible(th: np.ndarray) -> bool:
            # F y D son los denominadores: el predictor filtra por 1/F y por
            # D/C, de modo que ambos deben quedar estables.
            _, F, _, D = desempaquetar(th)
            ok_F = es_estable(F[1:]) if nf > 0 else True
            ok_D = es_estable(D[1:]) if nd > 0 else True
            return ok_F and ok_D

        theta_inicial = np.concatenate(
            [B0, F0[1:], C0[1:] if C0.size > 1 else [], D0[1:] if D0.size > 1 else []]
        )

        resultado = gauss_newton(
            theta_inicial,
            evaluar=evaluar,
            es_admisible=admisible,
            jacobiana=jacobiana,
            max_iter=max_iter,
            tol=tol,
        )

        B, F, C, D = desempaquetar(resultado.theta)
        return (
            B,
            estabilizar(F),
            C,
            estabilizar(D),
            resultado.iteraciones,
            resultado.motivo_fin,
        )

    # ------------------------------------------------------------------
    # Busqueda del retardo y presentacion
    # ------------------------------------------------------------------

    def buscar_mejor_nk(
        self,
        nb: int,
        nc: int,
        nd: int,
        nf: int,
        nk_inicial: Optional[int] = None,
        ventana: int = 5,
        remover_media: bool = True,
    ) -> tuple:
        """Busca el nk de mejor FIT alrededor del lag dominante.

        Devuelve (mejor_resultado, lag_correlacion, mejor_nk).
        """
        lag = self.lag_dominante()
        mejor_nk = nk_inicial if nk_inicial is not None else lag

        mejor_fit = -np.inf
        mejor_resultado = None

        for nk in range(max(0, lag - ventana), lag + ventana + 1):
            try:
                r = self.estimar(nb, nc, nd, nf, nk, remover_media)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if r.fit > mejor_fit:
                mejor_fit = r.fit
                mejor_resultado = r
                mejor_nk = nk

        return mejor_resultado, lag, mejor_nk

    def obtener_ecuacion_str(self, resultado: ResultadoBJ) -> str:
        """Devuelve los cuatro polinomios como texto."""
        return (
            f"B(q) = {formatear_polinomio(resultado.B, retardo_base=resultado.nk)}\n"
            f"F(q) = {formatear_polinomio(resultado.F, monico=True)}\n"
            f"C(q) = {formatear_polinomio(resultado.C, monico=True)}\n"
            f"D(q) = {formatear_polinomio(resultado.D, monico=True)}"
        )


def bj(
    u,
    y,
    ts: float = 1.0,
    *,
    nb: int = 1,
    nc: int = 1,
    nd: int = 1,
    nf: int = 1,
    nk: int = 1,
    remover_media: bool = True,
    descarte: Optional[int] = None,
) -> ResultadoBJ:
    """Estima un modelo Box-Jenkins directamente a partir de las senales."""
    datos = DatosExperimento.desde_arrays(u, y, ts)
    return IdentificadorBJ(datos).estimar(
        nb=nb,
        nc=nc,
        nd=nd,
        nf=nf,
        nk=nk,
        remover_media=remover_media,
        descarte=descarte,
    )
