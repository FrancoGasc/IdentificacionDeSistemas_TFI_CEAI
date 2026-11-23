# IdentificacionSistemasPy

Identificación paramétrica de sistemas SISO en tiempo discreto: **ARX, ARMAX,
Output Error y Box–Jenkins** por minimización del error de predicción.

Solo depende de NumPy.

```bash
pip install IdentificacionSistemasPy
```

```python
import identificacion_sistemas as ids

r = ids.arx(u, y, ts=0.1, na=2, nb=2, nk=1)
print(r.A, r.B, r.fit, r.aic)
```

---

## Qué tiene de distinto

Implementar los cuatro modelos polinomiales es la parte fácil. Lo difícil es
que la estimación sea **reproducible** y que las métricas sean **comparables**,
y es ahí donde está el trabajo de esta librería.

**Jacobiana analítica en ARMAX y en OE.** Las derivadas del error de predicción
tienen forma cerrada: son las señales de regresión filtradas por `1/C` en ARMAX
y por `1/F` en OE (Ljung 1999, ec. 10.50). Usarlas en lugar de diferencias
finitas no es una optimización de velocidad sino de reproducibilidad. Con
jacobiana numérica, las diferencias de redondeo entre LAPACK en Windows y en
Linux se amplifican a lo largo de las iteraciones y el resultado deja de ser el
mismo en dos máquinas distintas.

**Restricción de estabilidad por reflexión de raíces.** Durante la
optimización, los polinomios de ruido pueden tomar valores con raíces fuera del
círculo unitario. Un modelo de ruido inestable hace que el predictor recursivo
—que filtra por el inverso de esos polinomios— diverja, produciendo un costo
numéricamente bajo pero sin sentido físico. Cada vez que eso ocurre, las raíces
se reflejan al interior con `z → 1/z̄`, lo que preserva la respuesta en
magnitud. Sin esta restricción la optimización puede divergir.

**Arranque múltiple.** Los tres métodos no lineales se inicializan con una
estimación más simple —un ARX de orden elevado en ARMAX, un ARX simple en OE,
la secuencia OE → ARMA en BJ— y la optimización se repite desde varios puntos,
conservando el de menor costo. Además de mejorar la solución, evita que una
única trayectoria quede atrapada en un óptimo distinto según el redondeo de cada
máquina.

**Cada estructura se evalúa con su propio predictor óptimo.** Este es el punto
que más consecuencias tiene y el que las implementaciones ingenuas suelen errar.

| Estructura | Predictor óptimo a un paso |
|---|---|
| ARX | `ŷ = (1−A) y + B u(t−nk)` |
| ARMAX | `ŷ = (1−A) y + B u(t−nk) + (C−1) ε` |
| OE | `ŷ = (B/F) u(t−nk)` — **es la simulación pura** |
| BJ | `ŷ = y − (D/C)·[ y − (B/F) u(t−nk) ]` |

El AIC solo estima la verosimilitud cuando la varianza del error se calcula con
el predictor que la hipótesis de ruido de cada estructura determina. Evaluar las
cuatro con un único predictor —típicamente el de ARX, que es el más simple de
programar— produce un resultado degenerado: los polinomios de ruido no
intervienen, ARMAX, OE y BJ producen prácticamente el mismo residuo, y el AIC
las ordena solo por cantidad de parámetros. Gana siempre ARX, incluso cuando su
modelo es inservible. Medido sobre cuatro plantas con cuatro tipos de ruido
distintos, el modelo así elegido erraba la ganancia estática en un 31,98 % en un
caso y la estimaba **con el signo invertido** en otro, mientras las estructuras
que sí modelan el ruido quedaban relegadas por tener dos parámetros más.

## Validación contra la System Identification Toolbox de Matlab

Seis plantas sintéticas de parámetros conocidos, excitadas con PRBS de 1500
muestras y con el tipo de ruido que la estructura de cada una supone. Los mismos
datos y los mismos órdenes en ambas herramientas.

| Planta | Estructura | Esta librería | Matlab | FIT propio | FIT Matlab |
|---|---|---|---|---|---|
| 1 — primer orden | ARX | A=[1 −0,9342] | A=[1 −0,9342] | 97,1153 | 97,2270 |
| 3 — 2.º orden subamortiguado | ARMAX | A=[1 −1,6590 0,7603] | A=[1 −1,6590 0,7602] | 97,8758 | 97,8784 |
| 4 — dos polos separados | ARX | A=[1 −1,8839 0,8869] | A=[1 −1,8840 0,8871] | 97,7154 | 97,6635 |
| 5 — fase no mínima | OE | F=[1 −1,8978 0,8995] | F=[1 −1,8993 0,9009] | 90,4113 | 92,3563 |
| 6 — ruido con dinámica propia | BJ | F=[1 −0,9357] | F=[1 −0,9358] | 82,3642 | 83,0082 |

Los coeficientes del modelo de la planta difieren **a lo sumo en 0,0015**, y en
cuatro de las cinco en 0,0002 o menos. Tratándose de implementaciones
independientes, ese acuerdo es la evidencia más directa de que el motor resuelve
correctamente el problema de estimación.

La sexta planta —primer orden con tiempo muerto— es el caso en que las dos
herramientas **no** coinciden, y vale la pena por lo que muestra: esta librería
estima `nk=7` y la referencia `nk=6`. El retardo discreto verdadero es de 7
muestras, 6 del transporte físico y una que introduce el retenedor de orden
cero, porque el equivalente discreto por ZOH de un proceso estrictamente propio
tiene siempre al menos una muestra de retardo. Con una muestra de menos, la
referencia debe compensar distorsionando el polo: su constante de tiempo resulta
de 8,84 s contra el valor verdadero de 5 s.

## Cosas que conviene saber antes de confiar en un resultado

**Un FIT alto no garantiza un buen modelo.** Sobre una planta de dos polos
muestreada a `Ts = τ₁/50`, el FIT por predicción da 97,72 % y el de simulación
17,60 % con el mismo modelo y los mismos datos. La causa no es el estimador sino
el condicionamiento del experimento: con un período de muestreo muy fino frente
a la dinámica del proceso, los polos se acumulan cerca de `z = 1` y su posición
queda determinada por cifras que el ruido no permite resolver. La referencia de
Matlab sufre el mismo efecto sobre esos datos.

**Un indicador barato del problema: `sum(A)`.** La ganancia estática se evalúa
como `sum(B)/sum(A)`, y ese denominador es una suma de coeficientes de signo
alterno y magnitud próxima a la unidad. Si resulta del orden de `1e-3`, un error
pequeño en los coeficientes se amplifica enormemente en `K`, y tanto la ganancia
como las constantes de tiempo derivadas deben tomarse con reserva.

**Un FIT moderado tampoco significa que el modelo sea malo.** Cuando buena parte
de la varianza de la salida proviene del ruido, ningún modelo de la planta puede
explicarla. En la planta 6 de la validación, el FIT más bajo de las seis (82,36 %)
corresponde a la identificación que recupera los parámetros físicos con mayor
exactitud: 0,30 % de error en la ganancia y 0,24 % en la constante de tiempo.

**La estructura la decide la física del ruido, no el criterio de ajuste.**

| Situación del proceso | Estructura |
|---|---|
| Primera aproximación, ruido simple o poco relevante | ARX |
| Perturbaciones que ingresan a través de la dinámica de la planta | ARMAX |
| Ruido de medición dominante sobre una planta poco perturbada | OE |
| Perturbaciones con dinámica propia, independiente de la planta | BJ |

El AIC sirve para elegir el **orden** dentro de una estructura. Para elegir
entre estructuras es un auxiliar: las ordena correctamente solo porque la
hipótesis de ruido de cada una ya está incorporada en el predictor con que se
la evalúa.

## API

### Atajos funcionales

```python
r = ids.arx(u, y, ts=0.1, na=2, nb=2, nk=1)
r = ids.armax(u, y, ts=0.1, na=2, nb=2, nc=1, nk=1)
r = ids.oe(u, y, ts=0.1, nb=2, nf=2, nk=1)
r = ids.bj(u, y, ts=0.1, nb=1, nc=1, nd=1, nf=1, nk=1)
```

Cada uno devuelve un `Resultado*` con los polinomios, `theta`, `fit`, `aic`,
`n_params`, `ganancia`, `y_simulada`, `y_prediccion` y, en los iterativos,
`iteraciones`, `arranques`, `convergio` y `motivo_fin`.

### Con el objeto de datos

```python
datos = ids.DatosExperimento.desde_arrays(u, y, ts=0.1)

identificador = ids.IdentificadorARMAX(datos)
r = identificador.estimar(na=2, nb=2, nc=1, nk=1)
print(identificador.obtener_ecuacion_str(r))

# retardo por correlación cruzada
nk = identificador.lag_dominante()
r, lag, mejor_nk = identificador.buscar_mejor_nk(na=2, nb=2, nc=1)
```

### Selección de orden

```python
s = ids.SugeridorOrden(datos).sugerir("ARMAX", nk=1, rango_planta=(1, 4))
print(s.recomendado)            # menor AIC con desempate por parsimonia
print(s.recorrido_aic())        # amplitud del barrido: ver el diagnóstico
for c in s.tabla_ordenada()[:5]:
    print(c)
for f in s.fallidos:            # las que no se pudieron estimar, con el motivo
    print(f.ordenes_str(), f.error)
```

La amplitud del barrido es un diagnóstico de la estructura: un descenso
monótono de cientos de unidades de AIC sin encontrar mínimo no pide más orden,
avisa que la estructura elegida no describe el ruido del proceso.

### Comparación de estructuras

```python
for c in ids.comparar_las_cuatro(datos, na=1, nb=1, nf=1, nk=1):
    print(c)
```

### Validación de los datos

```python
ids.validar(datos)                        # lanza ErrorValidacion si algo falla
ids.contar_muestras_repetidas(datos.y)    # delata una fuente que dejó de actualizarse
```

## Alcance y límites

Sistemas de **una entrada y una salida**, lineales e invariantes en el tiempo,
estimados directamente en tiempo discreto. No hay MIMO, no hay modelos no
lineales y no hay identificación en espacio de estados.

El rendimiento está pensado para registros de ensayo, del orden de miles de
muestras. Los predictores son recursiones implementadas en bucles de Python;
con registros muy largos conviene esperar tiempos de cálculo apreciables, sobre
todo en Box–Jenkins, cuya jacobiana es numérica.

## Origen y cita

El código proviene del núcleo de cálculo de la herramienta desarrollada para el
Trabajo Final Integrador *«Identificación de sistemas y gemelos digitales: una
plataforma para la validación segura de lazos de control»*, de la Carrera de
Especialización en Automatización Industrial de la FIUBA. Lo que se publica acá
son los algoritmos de identificación; la aplicación completa —adquisición por
OPC UA, MQTT y puerto serie, reducción FOPDT, sintonía PID y simulación de lazo
cerrado— no forma parte de esta librería.

Si la usás en un trabajo publicado, el archivo `CITATION.cff` tiene los datos de
la cita.

## Referencias

- Ljung, L. (1999). *System Identification: Theory for the User*, 2nd ed. Prentice Hall.
- Söderström, T. y Stoica, P. (1989). *System Identification*. Prentice Hall.
- Hannan, E. J. y Rissanen, J. (1982). Recursive estimation of mixed autoregressive-moving average order. *Biometrika*, 69(1).
- Marquardt, D. W. (1963). An algorithm for least-squares estimation of nonlinear parameters. *J. SIAM*, 11(2).
- Burnham, K. P. y Anderson, D. R. (2002). *Model Selection and Multimodel Inference*, 2nd ed. Springer.
- Åström, K. J. y Wittenmark, B. (1997). *Computer-Controlled Systems*, 3rd ed. Prentice Hall.

## Licencia

MIT.

---

### In English

Parametric identification of SISO discrete-time systems in Python: ARX, ARMAX,
Output Error and Box–Jenkins, estimated by prediction-error minimization. NumPy
is the only dependency.

What sets it apart: analytic Jacobians for ARMAX and OE (so results reproduce
across platforms and BLAS implementations), root-reflection stability
constraints during optimization, multi-start initialization, and — most
importantly — **each structure is scored with its own optimal one-step
predictor**, which is the condition that makes FIT and AIC comparable across
structures. Validated against MATLAB's System Identification Toolbox on six
synthetic plants: plant-model coefficients agree to within 0.0015.

The documentation and the API are in Spanish.
