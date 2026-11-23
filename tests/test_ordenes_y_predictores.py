"""Combinaciones de ordenes, predictores y barrido de AIC.

El primer bloque cubre un defecto concreto de la implementacion original, y
conviene dejar escrito cual, porque es el tipo de falla que un test de
regresion existe para impedir.

El predictor de ARX recorria los valores pasados de la salida con un slice
invertido de la forma `y[t-1 : t-na-1 : -1]`. El bucle arrancaba en
t = max(na, nb+nk), de modo que cuando na >= nb + nk ese indice de parada valia
-1 en la primera iteracion; Python lo interpreta como "el penultimo elemento" y
no como "antes del cero", el slice salia vacio y la estimacion fallaba con un
error de broadcasting. Barriendo una grilla de 6x6 con nk en {1,2}, fallaban 25
de 72 combinaciones.

Lo serio era la consecuencia indirecta: el barrido de ordenes atrapaba la
excepcion y seguia de largo, de modo que las combinaciones afectadas
desaparecian del resultado sin dejar rastro. Sobre una grilla de 4x4 devolvia
diez candidatos de dieciseis, y los seis ausentes eran exactamente los de
na > nb. Un barrido incompleto que se presenta como completo es peor que uno
que falla.
"""

from __future__ import annotations

import numpy as np
import pytest

import identificacion_sistemas as ids

from plantas import primer_orden


@pytest.fixture(scope="module")
def datos():
    return primer_orden("arx").datos


# ----------------------------------------------------------------------
# Regresion: todas las combinaciones de ordenes deben estimarse
# ----------------------------------------------------------------------

@pytest.mark.parametrize("na", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("nb", [1, 2, 3])
@pytest.mark.parametrize("nk", [0, 1, 2])
def test_arx_estima_con_cualquier_combinacion_de_ordenes(datos, na, nb, nk):
    """Incluye el caso na >= nb + nk, que antes fallaba."""
    r = ids.IdentificadorARX(datos).estimar(na=na, nb=nb, nk=nk)

    assert r.A.size == na + 1
    assert r.B.size == nb
    assert np.isfinite(r.fit)
    assert np.isfinite(r.aic)
    assert r.A[0] == 1.0


def test_el_barrido_cubre_la_grilla_completa(datos):
    """Las 16 combinaciones de una grilla 4x4, no solo las de na <= nb."""
    sugerencia = ids.SugeridorOrden(datos).sugerir("ARX", nk=1, rango_planta=(1, 4))

    presentes = {(c.ordenes["na"], c.ordenes["nb"]) for c in sugerencia.candidatos}
    esperadas = {(na, nb) for na in range(1, 5) for nb in range(1, 5)}

    assert presentes == esperadas
    assert sugerencia.fallidos == []


def test_las_combinaciones_fallidas_se_registran_con_su_motivo():
    """Una combinacion que no se puede estimar queda documentada, no oculta.

    Con un registro corto, los ordenes altos no tienen historia suficiente: es
    una falla legitima, y lo que se verifica aca es que el barrido la informe
    en lugar de hacerla desaparecer.
    """
    corto = primer_orden("arx", n=40).datos
    sugerencia = ids.SugeridorOrden(corto).sugerir(
        "ARX", nk=1, rango_planta=(1, 45)
    )

    assert sugerencia.fallidos, "se esperaban combinaciones inviables registradas"
    assert all(f.error for f in sugerencia.fallidos)
    assert all("no hay suficientes muestras" in f.error for f in sugerencia.fallidos)
    total = len(sugerencia.candidatos) + len(sugerencia.fallidos)
    assert total == 45 * 45, "ninguna combinacion debe desaparecer del barrido"


# ----------------------------------------------------------------------
# Predictores
# ----------------------------------------------------------------------

def test_en_oe_el_predictor_optimo_es_la_simulacion(datos):
    """Propiedad de la estructura, no un detalle de implementacion.

    Como el ruido es blanco y se suma directo a la salida, los valores pasados
    de y no aportan informacion para anticipar y(t): el mejor predictor es
    propagar B/F. De ahi que ambas metricas coincidan por definicion y que no
    haya una segunda curva que graficar.
    """
    r = ids.IdentificadorOE(datos).estimar(nb=1, nf=1, nk=1)

    assert r.fit == r.fit_simulacion
    assert r.y_prediccion is None


def test_armax_con_nc_cero_coincide_con_arx(datos):
    """Con C(q) = 1 el predictor de ARMAX se reduce exactamente al de ARX."""
    r_armax = ids.IdentificadorARMAX(datos).estimar(na=2, nb=2, nc=0, nk=1)
    r_arx = ids.IdentificadorARX(datos).estimar(na=2, nb=2, nk=1)

    assert np.allclose(r_armax.A, r_arx.A, atol=1e-12)
    assert np.allclose(r_armax.B, r_arx.B, atol=1e-12)
    assert r_armax.fit == pytest.approx(r_arx.fit, abs=1e-10)


def test_bj_con_c_y_d_unitarios_coincide_con_oe(datos):
    """Con C = D = 1 el filtrado de BJ es la identidad y queda el predictor de OE."""
    r_bj = ids.IdentificadorBJ(datos).estimar(nb=1, nc=0, nd=0, nf=1, nk=1)
    r_oe = ids.IdentificadorOE(datos).estimar(nb=1, nf=1, nk=1)

    # La coincidencia no es exacta porque BJ llega por otro camino: refina los
    # cuatro polinomios en conjunto con jacobiana numerica, de modo que su
    # trayectoria de optimizacion difiere de la de OE aunque el modelo al que
    # converge sea el mismo.
    assert np.allclose(r_bj.B, r_oe.B, atol=1e-6)
    assert np.allclose(r_bj.F, r_oe.F, atol=1e-6)
    assert r_bj.fit == pytest.approx(r_oe.fit, abs=1e-4)


def test_la_prediccion_a_un_paso_no_es_peor_que_la_simulacion(datos):
    """El predictor se reancla al dato medido en cada paso; la simulacion no.

    Por eso la prediccion es una prueba menos exigente, y su FIT no puede ser
    inferior al de simulacion sobre el mismo modelo.
    """
    r = ids.IdentificadorARX(datos).estimar(na=1, nb=1, nk=1)
    assert r.fit >= r.fit_simulacion


# ----------------------------------------------------------------------
# AIC y parsimonia
# ----------------------------------------------------------------------

def test_el_aic_penaliza_los_parametros_de_mas(datos):
    """Entre dos modelos de ajuste equivalente, gana el mas simple."""
    sugerencia = ids.SugeridorOrden(datos).sugerir("ARX", nk=1, rango_planta=(1, 3))
    recomendado = sugerencia.recomendado

    assert recomendado is not None
    # El orden verdadero de la planta es na=1, nb=1: no deberia recomendar
    # mas parametros que los necesarios cuando la estructura es la correcta.
    assert recomendado.n_params <= 3


def test_la_parsimonia_prefiere_el_modelo_simple_dentro_de_la_banda():
    """Verifica el criterio en forma aislada, con candidatos construidos."""
    candidatos = [
        ids.CandidatoOrden(ordenes={"na": 3}, n_params=6, fit=97.5, aic=-1000.5),
        ids.CandidatoOrden(ordenes={"na": 1}, n_params=2, fit=97.4, aic=-999.8),
        ids.CandidatoOrden(ordenes={"na": 5}, n_params=10, fit=97.6, aic=-995.0),
    ]
    elegido = ids.SugeridorOrden._elegir_por_parsimonia(candidatos)

    # -999,8 esta a 0,7 del minimo: dentro de la banda de equivalencia de 2
    # unidades, de modo que gana el de menos parametros.
    assert elegido.n_params == 2


def test_un_aic_no_finito_se_descarta():
    candidatos = [
        ids.CandidatoOrden(ordenes={"na": 1}, n_params=2, fit=0.0, aic=float("inf")),
        ids.CandidatoOrden(ordenes={"na": 2}, n_params=4, fit=90.0, aic=-500.0),
    ]
    elegido = ids.SugeridorOrden._elegir_por_parsimonia(candidatos)
    assert elegido.n_params == 4


def test_el_recorrido_del_aic_delata_la_estructura_equivocada():
    """Diagnostico por la forma del barrido, no por su minimo.

    Sobre ruido con dinamica propia, ARX usa cada orden adicional para absorber
    un poco mas del ruido que no modela: el AIC decrece sin detenerse y recorre
    cientos de unidades. BJ, que modela el ruido por separado, no necesita
    orden de mas y su barrido resulta comparativamente plano. La amplitud del
    recorrido es la senal de que la estructura, y no el orden, es lo que hay
    que cambiar.
    """
    planta = primer_orden("bj", sigma=0.25)

    barrido_arx = ids.SugeridorOrden(planta.datos).sugerir(
        "ARX", nk=1, rango_planta=(1, 5)
    )
    barrido_bj = ids.SugeridorOrden(planta.datos).sugerir(
        "BJ", nk=1, rango_planta=(1, 3), rango_ruido=(1, 1)
    )

    assert barrido_arx.recorrido_aic() > barrido_bj.recorrido_aic()
