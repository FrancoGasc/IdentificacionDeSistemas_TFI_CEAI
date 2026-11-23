"""Comparacion de varias estructuras sobre un mismo ensayo.

Estima varias configuraciones -cada una con su metodo y sus ordenes- y las
devuelve en un formato uniforme, ordenadas por AIC.

Por que la comparacion es valida
--------------------------------
Cada estructura se evalua con su propio predictor optimo. Esa es la condicion
que vuelve licito comparar los AIC entre estructuras: el AIC solo estima la
verosimilitud cuando la varianza del error se calcula con el predictor que la
hipotesis de ruido de la estructura determina.

Evaluar las cuatro con un unico predictor -tipicamente el de ARX, que es el
mas simple de implementar- produce un resultado degenerado. Los polinomios de
ruido no intervienen en ese predictor, de modo que ARMAX, OE y BJ producen
practicamente el mismo residuo y el AIC no tiene con que distinguirlas: las
ordena solo por la penalizacion 2d, es decir por cantidad de parametros, y
siempre gana ARX. Y gana incluso cuando su modelo es inservible: sobre ruido
con dinamica propia puede errar la ganancia estatica en un 30 % o estimarla con
el signo invertido, mientras las estructuras que si modelan el ruido quedan
relegadas por tener dos parametros mas.

Como leer el resultado
----------------------
El menor AIC es un indicio, no un veredicto. Dos verificaciones baratas e
independientes conviene hacerlas siempre:

* **La ganancia estatica.** El ingeniero de proceso casi siempre sabe
  aproximadamente cuanto se mueve la salida por unidad de entrada. Un modelo
  cuya ganancia es incompatible con eso se descarta sin mirar ninguna metrica.
* **La magnitud de sum(A) o sum(F).** Si es del orden de 1e-3, la ganancia y
  las constantes de tiempo derivadas del modelo estan mal condicionadas y
  deben tomarse con reserva por alto que sea el FIT.

Y conviene recordar que un FIT moderado no autoriza, por si solo, a concluir
que el modelo sea malo: cuando una parte considerable de la varianza de la
salida proviene del ruido, ningun modelo de la planta puede explicarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .modelos import DatosExperimento
from .orden import estimar_con_ordenes


@dataclass
class ConfiguracionComparada:
    """Una configuracion estimada, en formato uniforme para las cuatro.

    `polinomios` lleva los coeficientes con el retardo ya incorporado al
    numerador, de modo que las cuatro estructuras se presentan igual aunque
    ARX y ARMAX se expresen en terminos de A y OE y BJ en terminos de F.
    """

    metodo: str
    ordenes: dict
    polinomios: list = field(default_factory=list)
    fit: float = float("nan")
    aic: float = float("inf")
    n_params: int = 0
    ganancia: float = float("nan")
    convergio: bool = False
    motivo_fin: str = ""
    resultado: object = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def ordenes_str(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.ordenes.items())

    def __repr__(self) -> str:
        if self.error is not None:
            return f"ConfiguracionComparada({self.metodo} {self.ordenes_str()}, error={self.error!r})"
        return (
            f"ConfiguracionComparada({self.metodo} {self.ordenes_str()}, "
            f"d={self.n_params}, fit={self.fit:.4f}%, aic={self.aic:.2f}, "
            f"K={self.ganancia:.6f})"
        )


def estimar_configuracion(
    datos: DatosExperimento,
    metodo: str,
    ordenes: dict,
    remover_media: bool = True,
) -> ConfiguracionComparada:
    """Estima una configuracion y la devuelve en el formato uniforme.

    Propaga las excepciones: para capturarlas por configuracion esta
    `comparar`.
    """
    resultado = estimar_con_ordenes(datos, metodo, ordenes, remover_media)
    metodo = metodo.upper()
    nk = ordenes["nk"]
    B_con_retardo = np.concatenate([np.zeros(nk), resultado.B])

    if metodo == "ARX":
        polinomios = [("A(q)", resultado.A), ("B(q)", B_con_retardo)]
    elif metodo == "ARMAX":
        polinomios = [
            ("A(q)", resultado.A),
            ("B(q)", B_con_retardo),
            ("C(q)", resultado.C),
        ]
    elif metodo == "OE":
        polinomios = [("B(q)", B_con_retardo), ("F(q)", resultado.F)]
    else:
        polinomios = [
            ("B(q)", B_con_retardo),
            ("F(q)", resultado.F),
            ("C(q)", resultado.C),
            ("D(q)", resultado.D),
        ]

    return ConfiguracionComparada(
        metodo=metodo,
        ordenes=dict(ordenes),
        polinomios=polinomios,
        fit=resultado.fit,
        aic=resultado.aic if resultado.aic is not None else float("inf"),
        n_params=resultado.n_params,
        ganancia=resultado.ganancia,
        convergio=getattr(resultado, "convergio", True),
        motivo_fin=getattr(resultado, "motivo_fin", ""),
        resultado=resultado,
    )


def comparar(
    datos: DatosExperimento,
    configuraciones: list,
    remover_media: bool = True,
    ordenar: bool = True,
) -> list:
    """Estima varias configuraciones y las devuelve ordenadas por AIC.

    Parametros
    ----------
    configuraciones : list of (str, dict)
        Pares (metodo, ordenes). Por ejemplo:

            [("ARX",   {"na": 1, "nb": 1, "nk": 1}),
             ("ARMAX", {"na": 1, "nb": 1, "nc": 1, "nk": 1}),
             ("OE",    {"nb": 1, "nf": 1, "nk": 1}),
             ("BJ",    {"nb": 1, "nf": 1, "nc": 1, "nd": 1, "nk": 1})]

    ordenar : bool
        Si es True las configuraciones validas se devuelven de menor a mayor
        AIC y las fallidas al final.

    Las configuraciones que fallan se devuelven igual, con su campo `error`
    cargado: una configuracion que desaparece de la comparacion sin dejar
    rastro oculta la razon por la que no se pudo estimar.
    """
    comparadas = []
    for metodo, ordenes in configuraciones:
        try:
            comparadas.append(
                estimar_configuracion(datos, metodo, ordenes, remover_media)
            )
        except Exception as error:  # noqa: BLE001 - se registra, no se oculta
            comparadas.append(
                ConfiguracionComparada(
                    metodo=str(metodo).upper(),
                    ordenes=dict(ordenes),
                    error=f"{type(error).__name__}: {error}",
                )
            )

    if ordenar:
        comparadas.sort(
            key=lambda c: (not c.ok, c.aic if np.isfinite(c.aic) else float("inf"))
        )
    return comparadas


def comparar_las_cuatro(
    datos: DatosExperimento,
    *,
    na: int = 1,
    nb: int = 1,
    nc: int = 1,
    nd: int = 1,
    nf: int = 1,
    nk: int = 1,
    remover_media: bool = True,
) -> list:
    """Compara las cuatro estructuras al mismo orden de planta.

    Atajo para la pregunta habitual: con esta planta y este orden, que
    estructura describe mejor el ruido del proceso.

        >>> for c in comparar_las_cuatro(datos, na=1, nb=1, nf=1, nk=1):
        ...     print(c)
    """
    configuraciones = [
        ("ARX", {"na": na, "nb": nb, "nk": nk}),
        ("ARMAX", {"na": na, "nb": nb, "nc": nc, "nk": nk}),
        ("OE", {"nb": nb, "nf": nf, "nk": nk}),
        ("BJ", {"nb": nb, "nf": nf, "nc": nc, "nd": nd, "nk": nk}),
    ]
    return comparar(datos, configuraciones, remover_media=remover_media)
