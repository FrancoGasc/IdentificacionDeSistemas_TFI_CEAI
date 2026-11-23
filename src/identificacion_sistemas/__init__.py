"""IdentificacionSistemasPy — identificacion parametrica de sistemas SISO.

Cuatro estructuras de la familia de modelos de error de prediccion, estimadas
en tiempo discreto a partir de datos de entrada y salida:

    ARX     A(q) y(t) = B(q) u(t-nk) + e(t)
    ARMAX   A(q) y(t) = B(q) u(t-nk) + C(q) e(t)
    OE      y(t) = [B(q)/F(q)] u(t-nk) + e(t)
    BJ      y(t) = [B(q)/F(q)] u(t-nk) + [C(q)/D(q)] e(t)

Las cuatro son casos particulares de una misma formulacion general (Ljung
1999), y difieren en una sola cosa: la hipotesis que hacen sobre el camino por
el cual el ruido entra al proceso. La eleccion de la estructura es, por lo
tanto, una decision sobre la fisica del proceso y no un problema de ajuste.

Uso rapido
----------
    import identificacion_sistemas as ids

    r = ids.arx(u, y, ts=0.1, na=2, nb=2, nk=1)
    print(r.A, r.B, r.fit)

Uso con el objeto de datos
--------------------------
    datos = ids.DatosExperimento.desde_arrays(u, y, ts=0.1)

    r = ids.IdentificadorARMAX(datos).estimar(na=2, nb=2, nc=1, nk=1)

    sugerencia = ids.SugeridorOrden(datos).sugerir("ARMAX", nk=1)
    print(sugerencia.recomendado)

    for c in ids.comparar_las_cuatro(datos, na=1, nb=1, nf=1, nk=1):
        print(c)
"""

from __future__ import annotations

from ._polinomios import es_estable, estabilizar, formatear_polinomio
from .armax import IdentificadorARMAX, armax
from .arx import IdentificadorARX, arx
from .base import IdentificadorBase, descarte_por_omision
from .bj import IdentificadorBJ, bj
from .comparacion import (
    ConfiguracionComparada,
    comparar,
    comparar_las_cuatro,
    estimar_configuracion,
)
from .modelos import (
    MOTIVO_DIRECTO,
    MOTIVO_ESTANCAMIENTO,
    MOTIVO_MAX_ITER,
    MOTIVO_SINGULAR,
    MOTIVO_TOLERANCIA,
    DatosExperimento,
    ResultadoARMAX,
    ResultadoARX,
    ResultadoBJ,
    ResultadoOE,
)
from .oe import IdentificadorOE, oe
from .orden import (
    DELTA_AIC_EQUIVALENTE,
    CandidatoOrden,
    OrdenFallido,
    ResultadoSugerencia,
    SugeridorOrden,
    estimar_con_ordenes,
    sugerir_orden,
)
from .validacion import (
    ErrorValidacion,
    contar_muestras_repetidas,
    validar,
    validar_datos_crudos,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # datos
    "DatosExperimento",
    # estimadores: atajos funcionales
    "arx",
    "armax",
    "oe",
    "bj",
    # estimadores: clases
    "IdentificadorARX",
    "IdentificadorARMAX",
    "IdentificadorOE",
    "IdentificadorBJ",
    "IdentificadorBase",
    # resultados
    "ResultadoARX",
    "ResultadoARMAX",
    "ResultadoOE",
    "ResultadoBJ",
    # seleccion de orden
    "SugeridorOrden",
    "sugerir_orden",
    "CandidatoOrden",
    "OrdenFallido",
    "ResultadoSugerencia",
    "DELTA_AIC_EQUIVALENTE",
    "estimar_con_ordenes",
    # comparacion de estructuras
    "comparar",
    "comparar_las_cuatro",
    "estimar_configuracion",
    "ConfiguracionComparada",
    # validacion
    "validar",
    "validar_datos_crudos",
    "contar_muestras_repetidas",
    "ErrorValidacion",
    # utilidades
    "es_estable",
    "estabilizar",
    "formatear_polinomio",
    "descarte_por_omision",
    # motivos de finalizacion
    "MOTIVO_TOLERANCIA",
    "MOTIVO_ESTANCAMIENTO",
    "MOTIVO_MAX_ITER",
    "MOTIVO_SINGULAR",
    "MOTIVO_DIRECTO",
]
