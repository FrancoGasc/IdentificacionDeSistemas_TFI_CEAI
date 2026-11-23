"""Nivel 1: cada estructura frente a los parametros verdaderos de la planta.

Es el nivel mas exigente, porque no admite que un error compartido con una
herramienta de referencia pase inadvertido: las plantas son sinteticas y sus
parametros se conocen por construccion.

Los umbrales no se derivan de los resultados obtenidos sino del uso al que el
modelo esta destinado. Un modelo identificado sirve, tipicamente, para
sintonizar un controlador, y las reglas de sintonia calculan las ganancias a
partir de K y tau mediante expresiones en las que esos parametros intervienen
de manera directa: un error relativo en la ganancia del proceso se traslada, en
primera aproximacion, a un error del mismo orden en Kp. Un 5 % deja al lazo
dentro del regimen en que las familias de sintonia habituales se comportan de
manera previsible.

Los polinomios del modelo de ruido quedan deliberadamente fuera del criterio:
C y D existen para blanquear el residuo y evitar que la estimacion de la planta
resulte sesgada; son un instrumento para obtener el modelo, no el producto.
"""

from __future__ import annotations

import numpy as np
import pytest

import identificacion_sistemas as ids

from plantas import primer_orden, segundo_orden_subamortiguado

TOLERANCIA_PARAMETRO = 0.05   # 5 % en ganancia y constantes de tiempo


def tau_desde_polo(A: np.ndarray, ts: float) -> float:
    """Constante de tiempo a partir del polo dominante: tau = -Ts / ln|z|."""
    polos = np.roots(A)
    dominante = sorted(polos, key=np.abs, reverse=True)[0]
    return float(-ts / np.log(abs(dominante)))


def error_relativo(estimado: float, verdadero: float) -> float:
    return abs(estimado - verdadero) / abs(verdadero)


# ----------------------------------------------------------------------
# Cada estructura sobre la planta cuyo ruido corresponde a su hipotesis
# ----------------------------------------------------------------------

def test_arx_recupera_la_planta_con_ruido_coloreado_por_la_dinamica():
    planta = primer_orden("arx")
    r = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)

    assert error_relativo(r.ganancia, planta.K) < TOLERANCIA_PARAMETRO
    assert error_relativo(
        tau_desde_polo(r.A, planta.datos.ts), planta.tau
    ) < TOLERANCIA_PARAMETRO


def test_armax_recupera_la_dinamica_oscilatoria():
    planta = segundo_orden_subamortiguado()
    r = ids.IdentificadorARMAX(planta.datos).estimar(na=2, nb=2, nc=1, nk=1)

    # omega_n y zeta se derivan del par de polos complejos conjugados.
    polo = np.roots(r.A)[0]
    ts = planta.datos.ts
    omega_n = abs(np.log(polo) / ts)
    zeta = -np.real(np.log(polo) / ts) / omega_n

    assert error_relativo(float(omega_n), 2.0) < TOLERANCIA_PARAMETRO
    assert error_relativo(float(zeta), 0.4) < TOLERANCIA_PARAMETRO
    assert error_relativo(r.ganancia, 1.0) < TOLERANCIA_PARAMETRO


def test_oe_recupera_la_planta_con_ruido_de_medicion():
    planta = primer_orden("blanco")
    r = ids.IdentificadorOE(planta.datos).estimar(nb=1, nf=1, nk=1)

    assert error_relativo(r.ganancia, planta.K) < TOLERANCIA_PARAMETRO
    assert error_relativo(
        tau_desde_polo(r.F, planta.datos.ts), planta.tau
    ) < TOLERANCIA_PARAMETRO


def test_bj_recupera_la_planta_con_ruido_de_dinamica_propia():
    planta = primer_orden("bj")
    r = ids.IdentificadorBJ(planta.datos).estimar(nb=1, nc=1, nd=1, nf=1, nk=1)

    assert error_relativo(r.ganancia, planta.K) < TOLERANCIA_PARAMETRO
    assert error_relativo(
        tau_desde_polo(r.F, planta.datos.ts), planta.tau
    ) < TOLERANCIA_PARAMETRO


# ----------------------------------------------------------------------
# Los polinomios discretos frente a la discretizacion exacta
# ----------------------------------------------------------------------

def test_los_polinomios_estimados_se_acercan_a_la_discretizacion_exacta():
    planta = primer_orden("arx")
    r = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)

    assert np.allclose(r.A, planta.A, atol=5e-3)
    assert np.allclose(r.B, planta.B, atol=5e-3)


# ----------------------------------------------------------------------
# Un FIT bajo no significa un modelo malo
# ----------------------------------------------------------------------

def test_un_fit_moderado_puede_acompanar_parametros_exactos():
    """El FIT queda limitado por el ruido, no por el modelo.

    Con ruido de dinamica propia y amplitud apreciable, una parte considerable
    de la varianza de la salida no es explicable por NINGUN modelo de la planta.
    El FIT cae y los parametros, en cambio, se recuperan bien. Es la razon por
    la que un FIT moderado no autoriza, por si solo, a descartar un modelo.
    """
    planta = primer_orden("bj", sigma=0.25)
    r = ids.IdentificadorBJ(planta.datos).estimar(nb=1, nc=1, nd=1, nf=1, nk=1)

    assert r.fit < 95.0                                   # el ruido lo limita
    assert error_relativo(r.ganancia, planta.K) < 0.05    # y aun asi acierta


# ----------------------------------------------------------------------
# El sesgo de ARX cuando su hipotesis de ruido no se cumple
# ----------------------------------------------------------------------

def test_arx_sesga_la_ganancia_cuando_el_ruido_tiene_dinamica_propia():
    """Documenta el sesgo, que es una propiedad del metodo y no un defecto.

    ARX dispone de un unico polinomio A que debe ser a la vez denominador de la
    planta e inverso del filtro de ruido. Cuando el ruido tiene dinamica propia,
    A la absorbe y los polos resultantes no son los de la planta. BJ, que
    modela ambas dinamicas por separado, no tiene ese problema sobre los mismos
    datos.
    """
    planta = primer_orden("bj", sigma=0.25)

    r_arx = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)
    r_bj = ids.IdentificadorBJ(planta.datos).estimar(nb=1, nc=1, nd=1, nf=1, nk=1)

    error_arx = error_relativo(r_arx.ganancia, planta.K)
    error_bj = error_relativo(r_bj.ganancia, planta.K)

    assert error_bj < error_arx, (
        "sobre ruido con dinamica propia, BJ deberia recuperar la ganancia "
        f"mejor que ARX (ARX {error_arx:.1%}, BJ {error_bj:.1%})"
    )


# ----------------------------------------------------------------------
# El retardo
# ----------------------------------------------------------------------

@pytest.mark.parametrize("nk_verdadero", [1, 3, 7])
def test_el_retardo_se_recupera_por_correlacion_y_por_fit(nk_verdadero):
    planta = primer_orden("blanco", nk=nk_verdadero, sigma=0.02)
    identificador = ids.IdentificadorARX(planta.datos)

    mejor_nk, resultado = identificador.buscar_mejor_nk(na=1, nb=1, ventana=4)

    assert mejor_nk == nk_verdadero
    assert resultado is not None
    assert error_relativo(resultado.ganancia, planta.K) < TOLERANCIA_PARAMETRO
