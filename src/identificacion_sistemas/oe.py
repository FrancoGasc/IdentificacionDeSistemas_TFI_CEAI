"""Estructura Output Error:  y(t) = [B(q)/F(q)] u(t-nk) + e(t).

Output Error parte de la hipotesis opuesta a la de ARX: el ruido se suma a la
salida y NO atraviesa la dinamica del proceso. Es el caso del ruido de
instrumento -un transmisor con ruido electrico, un sensor con cuantizacion- y,
al no compartir denominador con la planta, su estimacion de B/F queda libre del
sesgo que ARX introduce en presencia de ruido coloreado.

Una propiedad de esta estructura conviene tenerla presente, porque decide como
se mide el ajuste: **en Output Error el predictor optimo a un paso ES la
simulacion pura**. Como el ruido es blanco y se suma directamente a la salida,
los valores pasados de y no aportan ninguna informacion para anticipar y(t), y
lo mejor que puede hacerse es propagar B/F. No hay, por lo tanto, dos metricas
entre las cuales elegir: el FIT de prediccion y el de simulacion coinciden por
definicion, y `y_prediccion` queda en None para no duplicar la misma curva.

Evaluar OE con el predictor de tipo ARX -usando los valores medidos de y en los
terminos del denominador- no es una aproximacion razonable sino sustituir el
predictor correcto por otro, y hunde el FIT varios puntos porcentuales.

Estimacion
----------
El predictor yhat(t) = [B(q)/F(q)] u(t-nk) es no lineal en los parametros, de
modo que se minimiza el error de prediccion por Gauss-Newton amortiguado con
jacobiana analitica -las derivadas son las senales de regresion filtradas por
1/F (Ljung 1999)-. El arranque es un ARX simple del mismo orden, y se repite
desde varios puntos para evitar los minimos locales, frecuentes en OE cuando el
modelo esta sobreparametrizado.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._optimizacion import gauss_newton
from ._polinomios import es_estable, estabilizar, formatear_transferencia
from .arx import estimar_arx_lineal
from .base import IdentificadorBase, descarte_por_omision
from .modelos import MOTIVOS_CONVERGENCIA, DatosExperimento, ResultadoOE

# Polos reales de arranque para F, cuando el ARX inicial no es buen punto de
# partida. Cubren una dinamica rapida, una media, una lenta y una alternante.
POLOS_INICIALES = (0.5, 0.7, 0.9, -0.5)


def construir_evaluador_oe(
    u: np.ndarray, y: np.ndarray, nb: int, nf: int, nk: int
) -> tuple:
    """Devuelve (evaluar, admisible) para el problema de minimos de OE.

    `evaluar(theta)` entrega el error de prediccion y su jacobiana ANALITICA en
    una sola pasada, con theta = [b1..bnb, f1..fnf].

    El predictor es la recursion

        yhat(t) = sum b_k u(t-nk-k) - sum f_k yhat(t-k)

    y las derivadas del error e = y - yhat son las senales de regresion
    filtradas por 1/F (Ljung 1999):

        de/db_k = -u(t-nk-k)  filtrado por 1/F
        de/df_k = +yhat(t-k)  filtrado por 1/F

    Usar la jacobiana analitica en lugar de diferencias finitas elimina la
    dependencia del paso eps y del redondeo de la biblioteca de algebra lineal,
    que es lo que hacia que los resultados no coincidieran entre plataformas.

    Box-Jenkins reutiliza este evaluador en su primera etapa, donde estima la
    planta B/F por OE ignorando el modelo de ruido.
    """
    n = y.size
    n_param = nb + nf
    inicio = max(nb + nk, nf)

    if n - inicio <= 0:
        raise ValueError(
            f"no hay suficientes muestras para nb={nb}, nf={nf}, nk={nk}: "
            f"se necesitan mas de {inicio} y el ensayo tiene {n}."
        )

    def evaluar(th: np.ndarray) -> tuple:
        b = th[:nb]
        f = th[nb:nb + nf]
        F = np.concatenate([[1.0], f])

        yhat = np.zeros(n)
        J = np.zeros((n, n_param))

        for t in range(inicio, n):
            valor = 0.0
            for k in range(nb):
                valor += b[k] * u[t - nk - k]
            for k in range(1, nf + 1):
                valor -= F[k] * yhat[t - k]
            yhat[t] = valor

            col = 0
            for k in range(nb):
                filtro = 0.0
                for i in range(1, nf + 1):
                    if t - i >= 0:
                        filtro -= F[i] * (-J[t - i, col])
                J[t, col] = -(u[t - nk - k] + filtro)
                col += 1
            for k in range(1, nf + 1):
                filtro = 0.0
                for i in range(1, nf + 1):
                    if t - i >= 0:
                        filtro -= F[i] * (-J[t - i, col])
                J[t, col] = -(-yhat[t - k] + filtro)
                col += 1

        return (y - yhat)[inicio:], J[inicio:]

    def admisible(th: np.ndarray) -> bool:
        return es_estable(th[nb:nb + nf])

    return evaluar, admisible


class IdentificadorOE(IdentificadorBase):
    """Estima un modelo Output Error por minimizacion del error de prediccion."""

    def estimar(
        self,
        nb: int,
        nf: int,
        nk: int = 1,
        remover_media: bool = True,
        max_iter: int = 200,
        tol: float = 1e-8,
        descarte: Optional[int] = None,
    ) -> ResultadoOE:
        """Estima B y F.

        Parametros
        ----------
        nb : int
            Orden de B (ceros de la planta). >= 1.
        nf : int
            Orden de F (polos de la planta). >= 1.
        nk : int
            Retardo puro en muestras. >= 0.
        """
        if nb < 1:
            raise ValueError(f"nb debe ser al menos 1; se recibio {nb}.")
        if nf < 1:
            raise ValueError(f"nf debe ser al menos 1; se recibio {nf}.")
        if nk < 0:
            raise ValueError(f"nk no puede ser negativo; se recibio {nk}.")

        u, y, u_mean, y_mean = self._centrar(remover_media)
        if descarte is None:
            descarte = descarte_por_omision(self.y.size)

        # Arranque: un ARX del mismo orden. Su A sirve como F inicial.
        F0, B0 = estimar_arx_lineal(u, y, nf, nb, nk)

        B, F, iteraciones, arranques, motivo = self._refinar(
            B0, F0, u, y, nb, nf, nk, max_iter=max_iter, tol=tol
        )

        # --- Metricas ---------------------------------------------------
        # El predictor optimo de OE es la simulacion, de modo que ambas
        # metricas coinciden y se informa una sola.
        y_sim = self._simular_planta(F, B, nk, u)
        y_simulada = y_sim + y_mean

        fit = self._calcular_fit(self.y, y_simulada, descarte)
        n_params = nb + nf
        aic = self._calcular_aic(self.y, y_simulada, n_params, descarte)

        return ResultadoOE(
            B=B,
            F=F,
            theta=np.concatenate([B, F[1:]]),
            nb=nb,
            nf=nf,
            nk=nk,
            fit=fit,
            ganancia=self._ganancia_estatica(B, F),
            y_simulada=y_simulada,
            u_mean=u_mean,
            y_mean=y_mean,
            iteraciones=iteraciones,
            arranques=arranques,
            convergio=motivo in MOTIVOS_CONVERGENCIA,
            motivo_fin=motivo,
            aic=aic,
            n_params=n_params,
            y_prediccion=None,
            fit_simulacion=fit,
            descarte=descarte,
        )

    # ------------------------------------------------------------------
    # Refinamiento no lineal
    # ------------------------------------------------------------------

    def _refinar(
        self,
        B0: np.ndarray,
        F0: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
        nb: int,
        nf: int,
        nk: int,
        max_iter: int,
        tol: float,
    ) -> tuple:
        """Gauss-Newton con jacobiana analitica y arranque multiple."""
        evaluar, admisible = construir_evaluador_oe(u, y, nb, nf, nk)

        # Candidato principal: el ARX inicial. Si su F no es estable se lo
        # lleva a cero, que corresponde a F(q) = 1 y es trivialmente estable.
        theta_arx = np.concatenate([B0, F0[1:]])
        if not es_estable(theta_arx[nb:nb + nf]):
            theta_arx[nb:nb + nf] = 0.0

        candidatos = [theta_arx]
        for polo in POLOS_INICIALES:
            variante = theta_arx.copy()
            coeficientes_f = np.zeros(nf)
            coeficientes_f[0] = -polo
            if es_estable(coeficientes_f):
                variante[nb:nb + nf] = coeficientes_f
                candidatos.append(variante)

        mejor_theta = theta_arx
        mejor_costo = np.inf
        mejor_motivo = ""
        iteraciones = 0
        arranques = 0

        for theta_inicial in candidatos:
            resultado = gauss_newton(
                theta_inicial,
                evaluar=evaluar,
                es_admisible=admisible,
                max_iter=max_iter,
                tol=tol,
            )
            arranques += 1
            iteraciones += resultado.iteraciones
            if resultado.costo < mejor_costo:
                mejor_costo = resultado.costo
                mejor_theta = resultado.theta
                mejor_motivo = resultado.motivo_fin

        B = mejor_theta[:nb]
        F = estabilizar(np.concatenate([[1.0], mejor_theta[nb:nb + nf]]))
        return B, F, iteraciones, arranques, mejor_motivo

    # ------------------------------------------------------------------
    # Busqueda del retardo y presentacion
    # ------------------------------------------------------------------

    def buscar_mejor_nk(
        self,
        nb: int,
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
                r = self.estimar(nb, nf, nk, remover_media)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if r.fit > mejor_fit:
                mejor_fit = r.fit
                mejor_resultado = r
                mejor_nk = nk

        return mejor_resultado, lag, mejor_nk

    def obtener_ecuacion_str(self, resultado: ResultadoOE) -> str:
        """Devuelve G(z) = B(z)/F(z) como texto de tres lineas."""
        return formatear_transferencia(
            resultado.B, resultado.F, retardo_base=resultado.nk
        )


def oe(
    u,
    y,
    ts: float = 1.0,
    *,
    nb: int = 1,
    nf: int = 1,
    nk: int = 1,
    remover_media: bool = True,
    descarte: Optional[int] = None,
) -> ResultadoOE:
    """Estima un modelo Output Error directamente a partir de las senales."""
    datos = DatosExperimento.desde_arrays(u, y, ts)
    return IdentificadorOE(datos).estimar(
        nb=nb, nf=nf, nk=nk, remover_media=remover_media, descarte=descarte
    )
