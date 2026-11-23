"""Valores de referencia fijos: el estimador no debe cambiar sin que se note.

Estos numeros no son "lo correcto" en ningun sentido absoluto; son lo que esta
implementacion produce hoy sobre datos generados con una semilla fija. Su
funcion es detectar que un refactor, una vectorizacion o un cambio de version
de NumPy modifiquen el resultado del estimador, que es precisamente lo que un
usuario no espera de una actualizacion menor.

Hay una razon concreta para fijarlos con tantas cifras. El objetivo de diseno
de esta implementacion era que la misma entrada diera el mismo resultado en
plataformas distintas y con distintas versiones de BLAS, y a eso apuntan la
jacobiana analitica, la restriccion de estabilidad y el arranque multiple. Un
test que tolerara diferencias en la cuarta cifra no verificaria esa propiedad.

La tolerancia no es, de todos modos, la misma para las cuatro estructuras:
Box-Jenkins calcula su jacobiana por diferencias finitas, de modo que amplifica
las diferencias de redondeo mas que las otras tres y admite un margen mayor.
"""

from __future__ import annotations

import numpy as np
import pytest

import identificacion_sistemas as ids

from plantas import primer_orden, segundo_orden_subamortiguado

# Tolerancia relativa por estructura.
TOLERANCIA = {
    "ARX": 1e-12,       # solucion cerrada por minimos cuadrados
    "ARMAX": 1e-9,      # Gauss-Newton con jacobiana analitica
    "OE": 1e-9,         # idem
    "BJ": 1e-6,         # jacobiana numerica: mas sensible al redondeo
}

REFERENCIA = {
    "arx_ruido_arx": dict(
        A=[1.0, -0.9360373636014867],
        B=[0.12703838843787715],
        fit=86.63040515170665,
        aic=-8483.23330088051,
        ganancia=1.9861343370272642,
        n_params=2,
    ),
    # Esta combinacion -na > nb- es la que la implementacion original no podia
    # calcular. Queda fijada para que no vuelva a perderse.
    "arx_na_mayor_nb": dict(
        A=[1.0, -0.9391498333043498, 0.013024874164660343, -0.010373543800959836],
        B=[0.1269624487677151],
        fit=86.64411134074047,
        aic=-8482.156551429749,
        ganancia=1.999361505588624,
        n_params=4,
    ),
    "armax_segundo_orden": dict(
        A=[1.0, -1.654814460856433, 0.7564852256169208],
        B=[0.10096495924638688, 0.0007133014284942695],
        C=[1.0, -0.4814717517573814],
        fit=95.44206166643015,
        aic=-11090.739200294012,
        ganancia=1.0000737273336249,
        n_params=5,
    ),
    "oe_ruido_blanco": dict(
        B=[0.12883930245008],
        F=[1.0, -0.9354849916502409],
        fit=85.83152878898558,
        aic=-8473.040619854555,
        ganancia=1.9970438777841542,
        n_params=2,
    ),
    "bj_ruido_propio": dict(
        B=[0.12901715210780326],
        C=[1.0, -0.5187427567317089],
        D=[1.0, -0.36630579980662925],
        F=[1.0, -0.9354303587626954],
        fit=85.82375091388256,
        aic=-8464.165521031282,
        ganancia=1.9981085481587695,
        n_params=4,
    ),
}

CASOS = [
    ("arx_ruido_arx", lambda: primer_orden("arx"), "ARX", {"na": 1, "nb": 1, "nk": 1}),
    ("arx_na_mayor_nb", lambda: primer_orden("arx"), "ARX", {"na": 3, "nb": 1, "nk": 1}),
    (
        "armax_segundo_orden",
        segundo_orden_subamortiguado,
        "ARMAX",
        {"na": 2, "nb": 2, "nc": 1, "nk": 1},
    ),
    ("oe_ruido_blanco", lambda: primer_orden("blanco"), "OE", {"nb": 1, "nf": 1, "nk": 1}),
    (
        "bj_ruido_propio",
        lambda: primer_orden("bj"),
        "BJ",
        {"nb": 1, "nc": 1, "nd": 1, "nf": 1, "nk": 1},
    ),
]


@pytest.mark.parametrize("nombre, fabrica, metodo, ordenes", CASOS)
def test_los_valores_de_referencia_se_reproducen(nombre, fabrica, metodo, ordenes):
    esperado = REFERENCIA[nombre]
    tolerancia = TOLERANCIA[metodo]

    resultado = ids.estimar_con_ordenes(fabrica().datos, metodo, ordenes)

    for polinomio in ("A", "B", "C", "D", "F"):
        if polinomio not in esperado:
            continue
        obtenido = np.asarray(getattr(resultado, polinomio), dtype=float)
        referencia = np.asarray(esperado[polinomio], dtype=float)
        assert obtenido.shape == referencia.shape, f"{polinomio}: cambio la forma"
        assert np.allclose(obtenido, referencia, rtol=tolerancia, atol=tolerancia), (
            f"{nombre}: el polinomio {polinomio} cambio\n"
            f"  obtenido:  {obtenido}\n"
            f"  referencia:{referencia}"
        )

    assert resultado.fit == pytest.approx(esperado["fit"], rel=tolerancia)
    assert resultado.aic == pytest.approx(esperado["aic"], rel=tolerancia)
    assert resultado.ganancia == pytest.approx(esperado["ganancia"], rel=tolerancia)
    assert resultado.n_params == esperado["n_params"]


def test_el_polinomio_monico_arranca_exactamente_en_uno():
    """No en 0,9999999: A, F, C y D deben ser monicos por construccion."""
    planta = primer_orden("bj")
    r = ids.IdentificadorBJ(planta.datos).estimar(nb=1, nc=1, nd=1, nf=1, nk=1)

    for polinomio in (r.F, r.C, r.D):
        assert polinomio[0] == 1.0


def test_los_polinomios_de_ruido_quedan_estables():
    """El predictor filtra por su inverso: si no son estables, diverge."""
    planta = primer_orden("bj", sigma=0.3)
    r = ids.IdentificadorBJ(planta.datos).estimar(nb=2, nc=2, nd=2, nf=2, nk=1)

    assert ids.es_estable(r.C[1:], radio=1.0)
    assert ids.es_estable(r.D[1:], radio=1.0)
    assert ids.es_estable(r.F[1:], radio=1.0)


def test_el_motivo_de_fin_distingue_convergencia_de_estancamiento():
    """`convergio` no debe activarse por el solo hecho de haberse detenido.

    En la implementacion original un unico indicador se activaba tanto cuando
    la mejora relativa del costo caia bajo la tolerancia como cuando la
    busqueda de linea se estancaba sin encontrar ningun paso que mejorara, y el
    usuario no tenia manera de distinguir las dos situaciones sin leer el
    codigo.
    """
    planta = primer_orden("armax")
    r = ids.IdentificadorARMAX(planta.datos).estimar(na=1, nb=1, nc=1, nk=1)

    assert r.motivo_fin in (
        ids.MOTIVO_TOLERANCIA,
        ids.MOTIVO_ESTANCAMIENTO,
        ids.MOTIVO_MAX_ITER,
        ids.MOTIVO_SINGULAR,
        ids.MOTIVO_DIRECTO,
    )
    assert r.convergio == (r.motivo_fin in (ids.MOTIVO_TOLERANCIA, ids.MOTIVO_DIRECTO))
    assert r.iteraciones > 0
    assert r.arranques > 0


def test_arx_no_itera_y_lo_informa():
    """ARX tiene solucion cerrada: no hay optimizacion que converger."""
    planta = primer_orden("arx")
    r = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)

    assert r.motivo_fin == ids.MOTIVO_DIRECTO
    assert r.convergio is True
