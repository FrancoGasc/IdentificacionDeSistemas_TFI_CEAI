"""La superficie publica: construccion de datos, validaciones y comparacion."""

from __future__ import annotations

import numpy as np
import pytest

import identificacion_sistemas as ids

from plantas import primer_orden


# ----------------------------------------------------------------------
# DatosExperimento
# ----------------------------------------------------------------------

def test_desde_arrays_reconstruye_el_vector_de_tiempo():
    u = np.ones(100)
    u[::2] = -1.0
    datos = ids.DatosExperimento.desde_arrays(u, np.arange(100.0), ts=0.25)

    assert datos.n_muestras == 100
    assert datos.t[0] == 0.0
    assert datos.t[-1] == pytest.approx(99 * 0.25)


def test_longitudes_distintas_se_rechazan():
    with pytest.raises(ValueError, match="deben coincidir"):
        ids.DatosExperimento.desde_arrays(np.zeros(10), np.zeros(11), ts=1.0)


@pytest.mark.parametrize("ts", [0.0, -1.0, float("nan")])
def test_un_periodo_de_muestreo_invalido_se_rechaza(ts):
    with pytest.raises(ValueError, match="positivo"):
        ids.DatosExperimento.desde_arrays(np.zeros(10), np.zeros(10), ts=ts)


def test_recortar_devuelve_un_tramo_con_el_tiempo_rereferenciado():
    datos = primer_orden("blanco", n=200).datos
    tramo = datos.recortar(50, 149)

    assert tramo.n_muestras == 100
    assert tramo.t[0] == 0.0
    assert np.array_equal(tramo.u, datos.u[50:150])
    # El original no se modifica.
    assert datos.n_muestras == 200


def test_recortar_valida_los_indices():
    datos = primer_orden("blanco", n=100).datos
    with pytest.raises(ValueError, match="fuera de rango"):
        datos.recortar(0, 500)


# ----------------------------------------------------------------------
# Validaciones de ordenes
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, mensaje",
    [
        ({"na": 0, "nb": 1, "nk": 1}, "na debe ser al menos 1"),
        ({"na": 1, "nb": 0, "nk": 1}, "nb debe ser al menos 1"),
        ({"na": 1, "nb": 1, "nk": -1}, "nk no puede ser negativo"),
    ],
)
def test_arx_rechaza_ordenes_invalidos(kwargs, mensaje):
    datos = primer_orden("blanco", n=200).datos
    with pytest.raises(ValueError, match=mensaje):
        ids.IdentificadorARX(datos).estimar(**kwargs)


def test_sin_muestras_suficientes_el_error_lo_dice():
    datos = ids.DatosExperimento.desde_arrays(
        np.sign(np.sin(np.arange(25.0))), np.arange(25.0), ts=1.0
    )
    with pytest.raises(ValueError, match="no hay suficientes muestras"):
        ids.IdentificadorARX(datos).estimar(na=20, nb=20, nk=5)


def test_pasar_arrays_en_lugar_del_objeto_de_datos_se_explica():
    with pytest.raises(TypeError, match="desde_arrays"):
        ids.IdentificadorARX(np.zeros(100))


# ----------------------------------------------------------------------
# Validacion de datos
# ----------------------------------------------------------------------

def test_una_entrada_constante_se_rechaza():
    with pytest.raises(ids.ErrorValidacion, match="constante"):
        ids.validar_datos_crudos(None, np.ones(100), np.arange(100.0))


def test_los_nan_se_rechazan():
    y = np.arange(100.0)
    y[7] = np.nan
    u = np.sign(np.sin(np.arange(100.0)))
    with pytest.raises(ids.ErrorValidacion, match="NaN"):
        ids.validar_datos_crudos(None, u, y)


def test_un_muestreo_irregular_se_detecta():
    t = np.arange(100.0)
    t[50:] += 3.0           # un salto en la base de tiempo
    u = np.sign(np.sin(np.arange(100.0)))
    with pytest.raises(ids.ErrorValidacion, match="no es constante"):
        ids.validar_datos_crudos(t, u, np.arange(100.0))


def test_contar_muestras_repetidas_delata_una_fuente_congelada():
    """Lo que la verificacion de uniformidad no puede ver.

    Un registro en el que la fuente dejo de actualizarse y el cliente volvio a
    leer el mismo valor retenido es, en el eje temporal, perfectamente
    uniforme: pasa la verificacion del periodo de muestreo sin observaciones.
    El recuento de muestras consecutivas identicas si lo detecta, porque en una
    senal analogica con ruido no deberian existir.
    """
    limpia = primer_orden("blanco").datos.y
    assert ids.contar_muestras_repetidas(limpia) == 0

    congelada = limpia.copy()
    congelada[200:250] = congelada[199]     # 50 muestras retenidas
    assert ids.contar_muestras_repetidas(congelada) >= 49


def test_validar_devuelve_el_ts_medido():
    datos = primer_orden("blanco", n=300).datos
    assert ids.validar(datos) == pytest.approx(datos.ts, rel=1e-12)


# ----------------------------------------------------------------------
# Comparacion de estructuras
# ----------------------------------------------------------------------

def test_comparar_las_cuatro_devuelve_las_cuatro_ordenadas_por_aic():
    datos = primer_orden("arx").datos
    comparadas = ids.comparar_las_cuatro(datos, na=1, nb=1, nc=1, nd=1, nf=1, nk=1)

    assert len(comparadas) == 4
    assert {c.metodo for c in comparadas} == {"ARX", "ARMAX", "OE", "BJ"}
    assert all(c.ok for c in comparadas)

    aics = [c.aic for c in comparadas]
    assert aics == sorted(aics)


def test_el_numerador_comparado_lleva_el_retardo_incorporado():
    datos = primer_orden("blanco", nk=3).datos
    comparada = ids.estimar_configuracion(
        datos, "ARX", {"na": 1, "nb": 1, "nk": 3}
    )

    etiquetas = dict(comparada.polinomios)
    assert etiquetas["B(q)"].size == 1 + 3
    assert np.all(etiquetas["B(q)"][:3] == 0.0)


def test_una_configuracion_fallida_se_informa_y_no_rompe_la_comparacion():
    datos = primer_orden("blanco", n=60).datos
    comparadas = ids.comparar(
        datos,
        [
            ("ARX", {"na": 1, "nb": 1, "nk": 1}),
            ("ARX", {"na": 100, "nb": 100, "nk": 1}),   # inviable con 60 muestras
        ],
    )

    assert len(comparadas) == 2
    assert comparadas[0].ok
    assert not comparadas[-1].ok
    assert "no hay suficientes muestras" in comparadas[-1].error


def test_un_metodo_desconocido_se_rechaza():
    datos = primer_orden("blanco", n=200).datos
    with pytest.raises(ValueError, match="metodo desconocido"):
        ids.estimar_con_ordenes(datos, "NARMAX", {"na": 1, "nb": 1, "nk": 1})


# ----------------------------------------------------------------------
# Utilidades de polinomios
# ----------------------------------------------------------------------

def test_es_estable_usa_el_radio_que_se_le_pide():
    # Raiz en 0,9995: estable respecto del circulo unitario, no respecto del
    # radio de 0,999 que se usa como restriccion durante la optimizacion.
    coeficientes = [-0.9995]
    assert ids.es_estable(coeficientes, radio=1.0)
    assert not ids.es_estable(coeficientes, radio=0.999)
    assert ids.es_estable([])        # P(q) = 1


def test_estabilizar_refleja_las_raices_de_afuera_preservando_la_magnitud():
    inestable = np.array([1.0, -2.0])        # raiz en z = 2
    estabilizado = ids.estabilizar(inestable)

    raices = np.roots(estabilizado)
    assert np.all(np.abs(raices) <= 1.0)
    assert estabilizado[0] == pytest.approx(1.0)
    assert abs(raices[0]) == pytest.approx(0.5)   # 1/2, el reflejo de 2


def test_formatear_polinomio_omite_los_coeficientes_despreciables():
    texto = ids.formatear_polinomio([1.0, -0.3364, 1e-9], monico=True)
    assert texto == "1 - 0.3364 z^-1"


# ----------------------------------------------------------------------
# Atajos funcionales
# ----------------------------------------------------------------------

def test_los_atajos_coinciden_con_las_clases():
    planta = primer_orden("arx")
    u, y, ts = planta.datos.u, planta.datos.y, planta.datos.ts

    por_atajo = ids.arx(u, y, ts, na=1, nb=1, nk=1)
    por_clase = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)

    assert np.array_equal(por_atajo.A, por_clase.A)
    assert por_atajo.fit == por_clase.fit


def test_el_descarte_de_la_metrica_es_configurable_y_queda_registrado():
    """La ventana de descarte es una convencion, no una definicion.

    La herramienta de referencia no la fija de la misma manera, y parte de las
    diferencias residuales de FIT entre implementaciones proviene de ahi. Por
    eso es un parametro y queda anotada en el resultado.
    """
    planta = primer_orden("arx")

    por_omision = ids.IdentificadorARX(planta.datos).estimar(na=1, nb=1, nk=1)
    assert por_omision.descarte == ids.descarte_por_omision(planta.datos.n_muestras)
    assert por_omision.descarte == 75          # max(20, 1500/20)

    otro = ids.IdentificadorARX(planta.datos).estimar(
        na=1, nb=1, nk=1, descarte=200
    )
    assert otro.descarte == 200
    assert otro.fit != por_omision.fit         # distinta ventana, distinto FIT
    assert np.array_equal(otro.A, por_omision.A)   # el modelo no cambia
