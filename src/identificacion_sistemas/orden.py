"""Seleccion del orden del modelo por AIC con desempate por parsimonia.

Barre combinaciones de ordenes para una estructura dada, estima cada una,
calcula su AIC y recomienda la mejor.

Criterio de decision
--------------------
Se elige el modelo de menor AIC, SALVO que exista uno mas simple cuyo AIC este
suficientemente cerca. El umbral es delta_AIC < 2: segun Burnham y Anderson
(2002) por debajo de esa diferencia no hay evidencia sustancial de que un
modelo sea mejor que otro, y en ese caso se prefiere el de menos parametros
(navaja de Occam).

Espacio de busqueda
-------------------
El retardo nk NO se barre: es una propiedad fisica del proceso y se determina
mejor por correlacion cruzada. Los ordenes de la planta se barren en un rango
chico y los del ruido en uno aun mas chico, porque el modelo de ruido rara vez
necesita orden alto. Sin esa acotacion la combinatoria explota: para BJ, barrer
cinco ordenes de 1 a 5 serian 3125 combinaciones.

Que puede y que no puede hacer el AIC
-------------------------------------
El AIC esta concebido para elegir el ORDEN dentro de una estructura fija, y
para eso funciona bien. Usarlo para elegir entre ESTRUCTURAS exige que cada una
se evalue con su propio predictor optimo -esta libreria lo hace- y aun asi el
criterio es un auxiliar de la decision estructural, no un sustituto: la
estructura la decide el camino por el cual el ruido entra al proceso.

La forma del barrido es, en si misma, un diagnostico util:

* Un salto grande seguido de un minimo claro indica que el orden era
  insuficiente y que ya se alcanzo el adecuado.
* Un descenso monotono que recorre cientos de unidades sin encontrar minimo
  indica que la ESTRUCTURA es la equivocada: el modelo esta usando el orden
  para suplir lo que su hipotesis de ruido no contempla. No corresponde seguir
  agregando polos sino cambiar de estructura.
* Un paisaje plano, con el orden verdadero a unas pocas unidades del minimo,
  indica que el orden ya es suficiente y que las diferencias restantes estan
  dentro del ruido del procedimiento de optimizacion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .armax import IdentificadorARMAX
from .arx import IdentificadorARX
from .bj import IdentificadorBJ
from .modelos import DatosExperimento
from .oe import IdentificadorOE

# Burnham y Anderson (2002): por debajo de 2 unidades de AIC no hay evidencia
# sustancial de que un modelo sea mejor que otro.
DELTA_AIC_EQUIVALENTE = 2.0

METODOS = ("ARX", "ARMAX", "OE", "BJ")


@dataclass
class CandidatoOrden:
    """Un modelo probado durante el barrido."""

    ordenes: dict
    n_params: int
    fit: float
    aic: float
    es_recomendado: bool = False
    convergio: bool = True
    motivo_fin: str = ""

    def ordenes_str(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.ordenes.items())

    def __repr__(self) -> str:
        marca = " <-- recomendado" if self.es_recomendado else ""
        return (
            f"CandidatoOrden({self.ordenes_str()}, d={self.n_params}, "
            f"fit={self.fit:.4f}%, aic={self.aic:.2f}{marca})"
        )


@dataclass
class OrdenFallido:
    """Una combinacion que no pudo estimarse, con el motivo.

    Se registran en lugar de descartarse en silencio: una combinacion que
    desaparece del barrido sin dejar rastro oculta tanto un orden inviable
    para los datos como un defecto de la implementacion.
    """

    ordenes: dict
    error: str

    def ordenes_str(self) -> str:
        return " ".join(f"{k}={v}" for k, v in self.ordenes.items())


@dataclass
class ResultadoSugerencia:
    """Resultado completo de un barrido de ordenes."""

    metodo: str
    nk: int
    candidatos: list = field(default_factory=list)
    fallidos: list = field(default_factory=list)
    recomendado: Optional[CandidatoOrden] = None

    def tabla_ordenada(self) -> list:
        """Candidatos de AIC finito, ordenados de mejor a peor."""
        validos = [c for c in self.candidatos if np.isfinite(c.aic)]
        return sorted(validos, key=lambda c: c.aic)

    def recorrido_aic(self) -> float:
        """Amplitud total del AIC a lo largo del barrido.

        Es la cifra que permite leer el diagnostico descripto en el modulo:
        cientos de unidades indican que la estructura no describe el ruido;
        unas pocas, que el orden ya es suficiente.
        """
        validos = [c.aic for c in self.candidatos if np.isfinite(c.aic)]
        if not validos:
            return 0.0
        return float(max(validos) - min(validos))

    def __repr__(self) -> str:
        recomendado = (
            self.recomendado.ordenes_str() if self.recomendado else "ninguno"
        )
        return (
            f"ResultadoSugerencia({self.metodo}, nk={self.nk}, "
            f"{len(self.candidatos)} candidatos, {len(self.fallidos)} fallidos, "
            f"recomendado: {recomendado})"
        )


class SugeridorOrden:
    """Busca el mejor orden de modelo para una estructura dada.

        >>> sugerencia = SugeridorOrden(datos).sugerir("ARMAX", nk=1)
        >>> sugerencia.recomendado
    """

    def __init__(self, datos: DatosExperimento):
        self.datos = datos

    def sugerir(
        self,
        metodo: str,
        nk: int = 1,
        remover_media: bool = True,
        rango_planta: tuple = (1, 4),
        rango_ruido: tuple = (1, 2),
        callback_progreso: Optional[Callable[[int, int], None]] = None,
    ) -> ResultadoSugerencia:
        """Barre ordenes para el metodo dado y recomienda el mejor por AIC.

        Parametros
        ----------
        metodo : str
            "ARX", "ARMAX", "OE" o "BJ".
        nk : int
            Retardo, fijo durante todo el barrido.
        rango_planta : (int, int)
            Minimo y maximo de los ordenes de la planta (na/nb/nf), inclusive.
        rango_ruido : (int, int)
            Minimo y maximo de los ordenes del ruido (nc/nd), inclusive.
        callback_progreso : callable, opcional
            Se llama como callback(actual, total) despues de cada estimacion.
            El barrido puede tardar -sobre todo en BJ-, de modo que conviene
            usarlo para informar avance.
        """
        metodo = metodo.upper()
        if metodo not in METODOS:
            raise ValueError(
                f"metodo desconocido: {metodo!r}. Opciones: {', '.join(METODOS)}."
            )

        combinaciones = self._grilla(metodo, nk, rango_planta, rango_ruido)
        total = len(combinaciones)

        candidatos: list[CandidatoOrden] = []
        fallidos: list[OrdenFallido] = []

        for indice, ordenes in enumerate(combinaciones, start=1):
            try:
                resultado = estimar_con_ordenes(
                    self.datos, metodo, ordenes, remover_media
                )
            except Exception as error:  # noqa: BLE001 - se registra, no se oculta
                fallidos.append(OrdenFallido(ordenes=ordenes, error=f"{type(error).__name__}: {error}"))
            else:
                candidatos.append(
                    CandidatoOrden(
                        ordenes=ordenes,
                        n_params=resultado.n_params,
                        fit=resultado.fit,
                        aic=resultado.aic if resultado.aic is not None else float("inf"),
                        convergio=getattr(resultado, "convergio", True),
                        motivo_fin=getattr(resultado, "motivo_fin", ""),
                    )
                )
            if callback_progreso is not None:
                callback_progreso(indice, total)

        sugerencia = ResultadoSugerencia(
            metodo=metodo, nk=nk, candidatos=candidatos, fallidos=fallidos
        )
        recomendado = self._elegir_por_parsimonia(candidatos)
        if recomendado is not None:
            recomendado.es_recomendado = True
        sugerencia.recomendado = recomendado
        return sugerencia

    # ------------------------------------------------------------------
    # Criterio de seleccion
    # ------------------------------------------------------------------

    @staticmethod
    def _elegir_por_parsimonia(candidatos: list) -> Optional[CandidatoOrden]:
        """Menor AIC, con desempate por cantidad de parametros.

        Se descartan unicamente los candidatos de AIC no finito. Un modelo que
        no convergio formalmente -que agoto sus iteraciones- puede aun asi
        tener un AIC valido y ser un buen candidato: la convergencia es
        informativa, no un criterio de exclusion.
        """
        validos = [c for c in candidatos if np.isfinite(c.aic)]
        if not validos:
            return None

        aic_minimo = min(c.aic for c in validos)
        equivalentes = [
            c for c in validos if (c.aic - aic_minimo) < DELTA_AIC_EQUIVALENTE
        ]
        equivalentes.sort(key=lambda c: (c.n_params, c.aic))
        return equivalentes[0]

    # ------------------------------------------------------------------
    # Grilla de busqueda
    # ------------------------------------------------------------------

    @staticmethod
    def _grilla(metodo: str, nk: int, rango_planta: tuple, rango_ruido: tuple) -> list:
        """Enumera las combinaciones de ordenes a probar."""
        lo_p, hi_p = rango_planta
        lo_r, hi_r = rango_ruido
        planta = range(lo_p, hi_p + 1)
        ruido = range(lo_r, hi_r + 1)

        if metodo == "ARX":
            return [
                {"na": na, "nb": nb, "nk": nk} for na in planta for nb in planta
            ]
        if metodo == "OE":
            return [
                {"nb": nb, "nf": nf, "nk": nk} for nf in planta for nb in planta
            ]
        if metodo == "ARMAX":
            return [
                {"na": na, "nb": nb, "nc": nc, "nk": nk}
                for na in planta
                for nb in planta
                for nc in ruido
            ]
        return [
            {"nb": nb, "nf": nf, "nc": nc, "nd": nd, "nk": nk}
            for nf in planta
            for nb in planta
            for nc in ruido
            for nd in ruido
        ]


def estimar_con_ordenes(
    datos: DatosExperimento,
    metodo: str,
    ordenes: dict,
    remover_media: bool = True,
):
    """Estima un modelo del metodo indicado con los ordenes dados.

    Es el punto unico donde se traduce un nombre de metodo y un diccionario de
    ordenes a la llamada concreta de cada identificador. Devuelve el objeto
    Resultado* correspondiente y propaga cualquier excepcion.
    """
    metodo = metodo.upper()

    if metodo == "ARX":
        return IdentificadorARX(datos).estimar(
            na=ordenes["na"], nb=ordenes["nb"], nk=ordenes["nk"],
            remover_media=remover_media,
        )
    if metodo == "ARMAX":
        return IdentificadorARMAX(datos).estimar(
            na=ordenes["na"], nb=ordenes["nb"], nc=ordenes["nc"], nk=ordenes["nk"],
            remover_media=remover_media,
        )
    if metodo == "OE":
        return IdentificadorOE(datos).estimar(
            nb=ordenes["nb"], nf=ordenes["nf"], nk=ordenes["nk"],
            remover_media=remover_media,
        )
    if metodo == "BJ":
        return IdentificadorBJ(datos).estimar(
            nb=ordenes["nb"], nc=ordenes["nc"], nd=ordenes["nd"],
            nf=ordenes["nf"], nk=ordenes["nk"],
            remover_media=remover_media,
        )
    raise ValueError(f"metodo desconocido: {metodo!r}. Opciones: {', '.join(METODOS)}.")


def sugerir_orden(
    datos: DatosExperimento,
    metodo: str,
    nk: int = 1,
    **kwargs,
) -> ResultadoSugerencia:
    """Atajo de SugeridorOrden(datos).sugerir(metodo, nk, ...)."""
    return SugeridorOrden(datos).sugerir(metodo, nk=nk, **kwargs)
