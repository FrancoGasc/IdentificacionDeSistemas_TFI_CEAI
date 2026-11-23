"""Estructuras de datos de la libreria.

Son el vocabulario comun: un objeto de entrada con las senales del ensayo y un
objeto de resultado por estructura de modelo. Ningun modulo inventa sus propias
tuplas o diccionarios para representar lo mismo.

Motivos de finalizacion de la optimizacion
------------------------------------------
Los tres metodos iterativos informan en `motivo_fin` por que se detuvieron.
Importa distinguirlos: en la implementacion original un unico indicador
`convergio` se activaba tanto cuando la mejora relativa del costo caia por
debajo de la tolerancia (convergencia genuina) como cuando el amortiguamiento
alcanzaba su tope y la busqueda de linea ya no encontraba ningun paso que
mejorara el costo (estancamiento), que son situaciones muy distintas.

    MOTIVO_TOLERANCIA   la mejora relativa del costo cayo bajo la tolerancia
    MOTIVO_ESTANCAMIENTO  la busqueda de linea no encontro ningun paso util
    MOTIVO_MAX_ITER     se agoto el numero maximo de iteraciones
    MOTIVO_SINGULAR     el sistema normal resulto singular
    MOTIVO_DIRECTO      no hubo optimizacion (solucion cerrada o nc=0)

`convergio` es True unicamente para MOTIVO_TOLERANCIA y MOTIVO_DIRECTO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MOTIVO_TOLERANCIA = "tolerancia"
MOTIVO_ESTANCAMIENTO = "estancamiento"
MOTIVO_MAX_ITER = "max_iter"
MOTIVO_SINGULAR = "singular"
MOTIVO_DIRECTO = "directo"

MOTIVOS_CONVERGENCIA = (MOTIVO_TOLERANCIA, MOTIVO_DIRECTO)


@dataclass
class DatosExperimento:
    """Un ensayo de identificacion listo para estimar.

    Parametros
    ----------
    t : np.ndarray
        Vector de tiempo [s]. Se admite None cuando no se dispone de el: en ese
        caso se reconstruye como k*ts.
    u : np.ndarray
        Senal de entrada (variable manipulada).
    y : np.ndarray
        Senal de salida (variable medida).
    ts : float
        Periodo de muestreo [s].
    fuente, nombre : str
        Metadatos opcionales, sin efecto sobre el calculo.

    Para el caso habitual conviene el constructor alternativo:

        datos = DatosExperimento.desde_arrays(u, y, ts=0.1)
    """

    t: Optional[np.ndarray]
    u: np.ndarray
    y: np.ndarray
    ts: float
    fuente: str = ""
    nombre: str = ""

    n_muestras: int = field(init=False)

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()

        if self.u.size != self.y.size:
            raise ValueError(
                f"u tiene {self.u.size} muestras e y tiene {self.y.size}; deben coincidir."
            )
        if not np.isfinite(self.ts) or self.ts <= 0:
            raise ValueError(f"ts debe ser un numero positivo; se recibio {self.ts!r}.")

        if self.t is None:
            self.t = np.arange(self.u.size, dtype=float) * self.ts
        else:
            self.t = np.asarray(self.t, dtype=float).ravel()
            if self.t.size != self.u.size:
                raise ValueError(
                    f"t tiene {self.t.size} muestras y u tiene {self.u.size}; deben coincidir."
                )

        self.n_muestras = int(self.u.size)

    @classmethod
    def desde_arrays(
        cls,
        u,
        y,
        ts: float = 1.0,
        *,
        nombre: str = "",
        fuente: str = "arrays",
    ) -> "DatosExperimento":
        """Construye el objeto a partir de u, y y el periodo de muestreo."""
        return cls(t=None, u=u, y=y, ts=ts, fuente=fuente, nombre=nombre)

    def recortar(self, inicio: int, fin: int) -> "DatosExperimento":
        """Devuelve un ensayo nuevo con el tramo [inicio, fin] inclusive.

        El tiempo se re-referencia a cero. El objeto original no se modifica.
        """
        n = self.n_muestras
        if inicio < 0 or inicio >= n:
            raise ValueError(f"inicio={inicio} fuera de rango [0, {n - 1}].")
        if fin < inicio or fin >= n:
            raise ValueError(f"fin={fin} fuera de rango [{inicio}, {n - 1}].")

        t = self.t[inicio:fin + 1].copy()
        return DatosExperimento(
            t=t - t[0],
            u=self.u[inicio:fin + 1].copy(),
            y=self.y[inicio:fin + 1].copy(),
            ts=self.ts,
            fuente=self.fuente,
            nombre=f"{self.nombre}  [{inicio}:{fin}]",
        )

    def __repr__(self) -> str:
        return (
            f"DatosExperimento(nombre={self.nombre!r}, "
            f"n_muestras={self.n_muestras}, ts={self.ts:.6g}s)"
        )


@dataclass
class ResultadoARX:
    """Resultado de una estimacion ARX:  A(q) y(t) = B(q) u(t-nk) + e(t)."""

    A: np.ndarray
    B: np.ndarray
    theta: np.ndarray

    na: int
    nb: int
    nk: int

    fit: float
    ganancia: float

    y_simulada: np.ndarray

    u_mean: float = 0.0
    y_mean: float = 0.0

    aic: Optional[float] = None
    n_params: int = 0

    y_prediccion: Optional[np.ndarray] = None
    fit_simulacion: Optional[float] = None

    descarte: int = 0
    motivo_fin: str = MOTIVO_DIRECTO
    convergio: bool = True

    @property
    def ordenes(self) -> dict:
        return {"na": self.na, "nb": self.nb, "nk": self.nk}

    def __repr__(self) -> str:
        return (
            f"ResultadoARX(na={self.na}, nb={self.nb}, nk={self.nk}, "
            f"fit={self.fit:.2f}%, K={self.ganancia:.4f})"
        )


@dataclass
class ResultadoARMAX:
    """Resultado ARMAX:  A(q) y(t) = B(q) u(t-nk) + C(q) e(t)."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    theta: np.ndarray

    na: int
    nb: int
    nc: int
    nk: int

    fit: float
    ganancia: float

    y_simulada: np.ndarray

    u_mean: float = 0.0
    y_mean: float = 0.0

    iteraciones: int = 0
    arranques: int = 0
    convergio: bool = False
    motivo_fin: str = ""

    aic: Optional[float] = None
    n_params: int = 0

    debug: list = field(default_factory=list)

    y_prediccion: Optional[np.ndarray] = None
    fit_simulacion: Optional[float] = None

    descarte: int = 0

    @property
    def ordenes(self) -> dict:
        return {"na": self.na, "nb": self.nb, "nc": self.nc, "nk": self.nk}

    def __repr__(self) -> str:
        return (
            f"ResultadoARMAX(na={self.na}, nb={self.nb}, nc={self.nc}, nk={self.nk}, "
            f"fit={self.fit:.2f}%, K={self.ganancia:.4f}, "
            f"iters={self.iteraciones}, {self.motivo_fin})"
        )


@dataclass
class ResultadoOE:
    """Resultado Output Error:  y(t) = [B(q)/F(q)] u(t-nk) + e(t).

    No hay polinomio A: los polos de la planta estan en F, y el ruido se suma
    directo a la salida.
    """

    B: np.ndarray
    F: np.ndarray
    theta: np.ndarray

    nb: int
    nf: int
    nk: int

    fit: float
    ganancia: float

    y_simulada: np.ndarray

    u_mean: float = 0.0
    y_mean: float = 0.0

    iteraciones: int = 0
    arranques: int = 0
    convergio: bool = False
    motivo_fin: str = ""

    aic: Optional[float] = None
    n_params: int = 0

    y_prediccion: Optional[np.ndarray] = None
    fit_simulacion: Optional[float] = None

    descarte: int = 0

    @property
    def ordenes(self) -> dict:
        return {"nb": self.nb, "nf": self.nf, "nk": self.nk}

    def __repr__(self) -> str:
        return (
            f"ResultadoOE(nb={self.nb}, nf={self.nf}, nk={self.nk}, "
            f"fit={self.fit:.2f}%, K={self.ganancia:.4f}, {self.motivo_fin})"
        )


@dataclass
class ResultadoBJ:
    """Resultado Box-Jenkins:  y(t) = [B/F] u(t-nk) + [C/D] e(t).

    Planta y ruido tienen denominadores independientes; es lo que distingue a
    esta estructura de ARMAX.
    """

    B: np.ndarray
    F: np.ndarray
    C: np.ndarray
    D: np.ndarray
    theta: np.ndarray

    nb: int
    nc: int
    nd: int
    nf: int
    nk: int

    fit: float
    ganancia: float

    y_simulada: np.ndarray

    u_mean: float = 0.0
    y_mean: float = 0.0

    iteraciones: int = 0
    arranques: int = 0
    convergio: bool = False
    motivo_fin: str = ""

    aic: Optional[float] = None
    n_params: int = 0

    y_prediccion: Optional[np.ndarray] = None
    fit_simulacion: Optional[float] = None

    descarte: int = 0

    @property
    def ordenes(self) -> dict:
        return {
            "nb": self.nb,
            "nf": self.nf,
            "nc": self.nc,
            "nd": self.nd,
            "nk": self.nk,
        }

    def __repr__(self) -> str:
        return (
            f"ResultadoBJ(nb={self.nb}, nf={self.nf}, nc={self.nc}, nd={self.nd}, "
            f"nk={self.nk}, fit={self.fit:.2f}%, {self.motivo_fin})"
        )
