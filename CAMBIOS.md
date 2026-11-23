# Cambios respecto de la implementación original

Esta librería es el núcleo de identificación de la aplicación del TFI, extraído
a un paquete independiente. El comportamiento numérico se conservó: sobre las
combinaciones de órdenes que la implementación original podía calcular, la
diferencia máxima entre ambas es de **1,8e-10**, y proviene exclusivamente del
caso de Box–Jenkins con jacobiana numérica, donde las diferencias de redondeo se
amplifican. En ARX, ARMAX y OE las diferencias quedan en el épsilon de la
máquina (1e-16 a 1e-13).

Los números del informe se reproducen con esta librería.

## Corrección de un defecto

**El predictor de ARX fallaba cuando `na >= nb + nk`.**

El recorrido de los valores pasados de la salida usaba un slice invertido de la
forma `y[t-1 : t-na-1 : -1]`. El bucle arranca en `t = max(na, nb+nk)`, de modo
que cuando `na >= nb+nk` el índice de parada vale `-1` en la primera iteración;
Python lo interpreta como «el penúltimo elemento» y no como «antes del cero», el
slice sale vacío y la estimación aborta con
`ValueError: operands could not be broadcast together with shapes (na,) (0,)`.

Barriendo una grilla de 6×6 con `nk ∈ {1,2}` fallaban **25 de 72**
combinaciones. Con `nk=1` la condición se reduce a `na > nb`.

La consecuencia indirecta era la seria: el barrido de órdenes atrapaba la
excepción con un `except Exception: pass` y seguía de largo, de modo que las
combinaciones afectadas desaparecían del resultado sin dejar rastro. Sobre una
grilla de 4×4 devolvía 10 candidatos de 16, y los 6 ausentes eran exactamente
los de `na > nb`.

Esto se puede verificar en el anexo de tablas del informe: el barrido ARX sobre
la Planta 4 tiene 21 filas y en ninguna `na > nb`. Veintiuno es justo la
cantidad de combinaciones con `na ≤ nb` en una grilla de 6×6. **La conclusión de
ese barrido no cambia** —el mínimo en `na=2 nb=2` es el orden verdadero de la
planta— pero la tabla cubre media grilla, no la grilla completa.

Corregido reemplazando el slice por `y[t-na:t][::-1]`, que no tiene caso borde.
ARMAX, OE y BJ no estaban afectados porque usan bucles explícitos.

## Cambios de comportamiento

**`convergio` ya no se activa por estancamiento.** La rutina de Gauss-Newton se
detiene por dos caminos muy distintos: cuando la mejora relativa del costo cae
bajo la tolerancia, que es convergencia genuina, y cuando el amortiguamiento
alcanza su tope y la búsqueda de línea ya no encuentra ningún paso que mejore,
que es un estancamiento. El indicador original no los distinguía, de modo que
señalaba que la optimización había terminado y no que el modelo fuera utilizable.
Ahora `motivo_fin` dice cuál de los dos ocurrió (`tolerancia`,
`estancamiento`, `max_iter`, `singular` o `directo`) y `convergio` es verdadero
solo para convergencia real.

**El barrido de órdenes registra las combinaciones fallidas.** En
`ResultadoSugerencia.fallidos`, cada una con su mensaje de error. Nada
desaparece en silencio.

**`iteraciones` cuenta iteraciones.** En BJ devolvía la constante `1` y en ARMAX
contaba arranques múltiples. Ahora `iteraciones` es la suma de iteraciones de
Gauss-Newton y `arranques` la cantidad de puntos de partida probados.

**ARX acepta `nk = 0`.** Antes exigía `nk >= 1` mientras ARMAX, OE y BJ
aceptaban `0`. La asimetría no tenía fundamento. Conviene recordar, de todos
modos, que el equivalente discreto por ZOH de un proceso estrictamente propio
tiene siempre al menos una muestra de retardo, así que `nk=1` sigue siendo el
valor habitual.

**La ventana de descarte del FIT y del AIC es un parámetro.** Antes estaba
fijada en `max(20, N/20)` dentro del cálculo de la métrica. Ese valor sigue
siendo el de omisión y queda anotado en el resultado (`descarte`), pero ahora se
puede cambiar. Es una convención y no una definición: la herramienta de
referencia no la fija de la misma manera, y parte de las diferencias residuales
de FIT entre ambas implementaciones proviene de ahí.

**`max_iter` y `tol` de ARMAX ahora hacen algo.** Eran parámetros de
`estimar()` que no se usaban en el cuerpo; quedaron conectados al refinamiento
PEM.

**`es_estable` toma el radio como parámetro.** Antes la comprobación usaba
`0.999` y la estabilización reflejaba con `1.0`, de modo que una raíz en
`0.9995` era «inestable» para una función e «intocable» para la otra. Ahora el
radio es explícito: `0.999` como restricción durante la optimización —para dejar
margen respecto del límite— y `1.0` para juzgar un polinomio ya estimado.

## Limpieza

- Las funciones de estabilidad, reflexión de raíces y formateo de polinomios
  estaban duplicadas en los cuatro módulos; ahora viven una sola vez en
  `_polinomios.py`.
- El Gauss-Newton amortiguado estaba triplicado (ARMAX, OE y una copia
  condensada dentro de BJ); ahora está una sola vez en `_optimizacion.py`.
- El evaluador de OE —predictor más jacobiana analítica— lo comparten OE y la
  primera etapa de BJ, que antes tenía su propia copia.
- Eliminado el `except:` desnudo de `bj._gauss_newton_oe`, que capturaba también
  `KeyboardInterrupt`.
- Eliminado el `IdentificadorOE` que `bj.estimar` instanciaba sin usar, y el
  `_calcular_residuos_armax` que quedó muerto tras el refactor del PEM.
- Las matrices de regresión se arman vectorizadas en lugar de fila por fila.
  Los valores son idénticos, de modo que `lstsq` devuelve el mismo resultado.
- `Optional[...]` donde el tipo declarado era `float` con omisión `None`.
- Se agregó `DatosExperimento.desde_arrays(u, y, ts)` y atajos funcionales
  `arx()`, `armax()`, `oe()`, `bj()` para no tener que construir el objeto de
  datos a mano.
- Se agregó `contar_muestras_repetidas()`, que detecta una adquisición en la que
  la fuente dejó de actualizarse: el caso que la verificación de uniformidad del
  período de muestreo no puede ver, porque un registro de valores retenidos es
  perfectamente uniforme en el eje temporal.
- Se agregó `ResultadoSugerencia.recorrido_aic()`, que expone la amplitud del
  barrido como diagnóstico de la estructura.

## Qué no se trajo

Por decisión de alcance, la librería es solo identificación. Quedaron fuera la
estimación FOPDT, la sintonía PID y la simulación del lazo cerrado, los lectores
de OPC UA, MQTT y puerto serie, la generación del reporte PDF y toda la capa de
interfaz. Los `QObject` de Qt que envolvían el sugeridor y el comparador
tampoco: la librería no depende de Qt.

## Pendiente

Mejoras identificadas y no incluidas en esta versión:

- **Errores estándar de los parámetros.** Para ARX salen de `σ²(ΦᵀΦ)⁻¹` y para
  los PEM de `JᵀJ`, que ya se calcula. Es lo que más se busca y no se encuentra
  en las alternativas disponibles en Python.
- **Prueba de blancura del residuo.** Detectaría de manera directa el caso en
  que la estructura elegida no modela el ruido, sin necesidad de interpretar la
  forma del barrido de AIC. Ya figura como trabajo futuro en el informe.
- **Rendimiento.** Los predictores son recursiones en bucles de Python;
  `scipy.signal.lfilter` cubre casi todas y las ventanas de regresión se pueden
  armar con `sliding_window_view`. La jacobiana numérica de BJ es el camino más
  lento. Cualquier cambio acá tiene que pasar los valores de referencia de
  `tests/test_regresion.py`.
- **Advertencia automática de mal condicionamiento** cuando `sum(A)` o `sum(F)`
  resultan del orden de `1e-3`.
