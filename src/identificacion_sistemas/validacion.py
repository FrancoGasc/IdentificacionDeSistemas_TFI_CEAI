"""Validaciones de los datos de un ensayo de identificacion.

Todas las funciones siguen el mismo patron: si algo esta mal lanzan una
excepcion con un mensaje que dice que revisar; si todo esta bien no hacen nada,
o devuelven el dato derivado.

Las comprobaciones son deliberadamente baratas y previas al calculo. No
reemplazan al juicio sobre la calidad del ensayo: un registro puede pasar todas
estas verificaciones y seguir siendo inservible para identificar -por ejemplo
si la entrada no excita las frecuencias de interes, o si el periodo de muestreo
es demasiado fino frente a la dinamica del proceso-.

Una limitacion que conviene conocer: `validar_periodo_muestreo` comprueba la
regularidad del reloj con que se registraron las muestras, no la tasa de
actualizacion de la fuente que las produjo. Una captura en la que el proceso
dejo de actualizar sus variables -y en la que, por lo tanto, muchas muestras
son repeticiones de la anterior- es perfectamente uniforme en el eje temporal y
pasa esta verificacion sin observaciones. Para ese caso esta
`contar_muestras_repetidas`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .modelos import DatosExperimento


class ErrorValidacion(ValueError):
    """Los datos del ensayo no son validos para identificar.

    Hereda de ValueError porque conceptualmente lo es, pero tiene nombre propio
    para poder distinguirlo de cualquier otro error de valor.
    """


def validar_longitudes_iguales(u, y) -> None:
    """Entrada y salida deben tener la misma cantidad de muestras."""
    u = np.asarray(u).ravel()
    y = np.asarray(y).ravel()
    if u.size != y.size:
        raise ErrorValidacion(
            f"la entrada u tiene {u.size} muestras y la salida y tiene {y.size}; "
            "deben coincidir."
        )


def validar_minimo_muestras(x, minimo: int = 20) -> None:
    """Con menos de ~20 muestras la estimacion no tiene sentido."""
    x = np.asarray(x).ravel()
    if x.size < minimo:
        raise ErrorValidacion(
            f"se necesitan al menos {minimo} muestras para identificar; hay {x.size}."
        )


def validar_sin_nan(nombre: str, x) -> None:
    """No puede haber valores NaN ni infinitos.

    Los NaN aparecen tipicamente al leer celdas vacias de un archivo.
    """
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        cantidad = int(np.sum(~np.isfinite(x)))
        raise ErrorValidacion(
            f"la senal '{nombre}' tiene {cantidad} valor(es) NaN o infinitos."
        )


def validar_variacion_entrada(u, umbral: float = 1e-10) -> None:
    """La entrada no puede ser constante.

    Si u no varia no hay informacion con que identificar el sistema: ninguna
    cantidad de datos compensa una entrada que no excita al proceso.
    """
    u = np.asarray(u, dtype=float).ravel()
    if np.std(u) < umbral:
        raise ErrorValidacion(
            "la senal de entrada u es constante: no contiene informacion para "
            "identificar el sistema."
        )


def validar_periodo_muestreo(t, tolerancia: float = 0.01) -> float:
    """Calcula Ts y verifica que sea constante. Devuelve Ts.

    La tolerancia es el coeficiente de variacion admitido entre intervalos
    consecutivos; 0,01 equivale a un 1 %.
    """
    t = np.asarray(t, dtype=float).ravel()
    if t.size < 2:
        raise ErrorValidacion("se necesitan al menos dos marcas de tiempo.")

    diferencias = np.diff(t)
    if np.any(diferencias <= 0):
        raise ErrorValidacion(
            "la columna de tiempo no es estrictamente creciente; verificar el orden."
        )

    ts_medio = float(np.mean(diferencias))
    variacion = float(np.std(diferencias) / ts_medio)
    if variacion > tolerancia:
        raise ErrorValidacion(
            f"el periodo de muestreo no es constante (variacion {variacion * 100:.1f} %); "
            f"Ts medio {ts_medio:.6g} s."
        )
    return ts_medio


def contar_muestras_repetidas(x) -> int:
    """Cuenta muestras consecutivas numericamente identicas.

    En una senal analogica con ruido de medicion no deberian existir. Un
    recuento alto delata una adquisicion en la que la fuente dejo de
    actualizarse y el cliente volvio a leer el mismo valor retenido, defecto
    que la verificacion de uniformidad del periodo de muestreo no puede
    detectar porque un registro de valores retenidos es, en el eje temporal,
    perfectamente uniforme.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        return 0
    return int(np.sum(np.diff(x) == 0.0))


def validar_datos_crudos(t, u, y, tolerancia_ts: float = 0.01) -> Optional[float]:
    """Ejecuta todas las validaciones y devuelve Ts.

    Si t es None se omite la verificacion del periodo de muestreo y se devuelve
    None: el resto de las comprobaciones se aplica igual.
    """
    validar_longitudes_iguales(u, y)
    validar_minimo_muestras(u)
    validar_sin_nan("u", u)
    validar_sin_nan("y", y)
    validar_variacion_entrada(u)

    if t is None:
        return None

    validar_sin_nan("t", t)
    validar_longitudes_iguales(t, u)
    return validar_periodo_muestreo(t, tolerancia_ts)


def validar(datos: DatosExperimento, tolerancia_ts: float = 0.01) -> float:
    """Valida un DatosExperimento ya construido y devuelve su Ts medido.

        >>> ts_medido = validar(datos)

    Notar que el Ts devuelto es el medido a partir de t, que puede diferir del
    Ts nominal guardado en el objeto. Las magnitudes expresadas en segundos
    -constantes de tiempo, retardos- heredan esa diferencia dígito a dígito.
    """
    ts_medido = validar_datos_crudos(datos.t, datos.u, datos.y, tolerancia_ts)
    return ts_medido if ts_medido is not None else datos.ts
