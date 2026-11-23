"""Recorrido completo sobre una planta simulada de primer orden con retardo.

Genera un ensayo, identifica con las cuatro estructuras, compara, elige el
orden por AIC y muestra como leer cada cifra. Se ejecuta con

    python ejemplos/ejemplo_basico.py
"""

from __future__ import annotations

import numpy as np

import identificacion_sistemas as ids

# ----------------------------------------------------------------------
# 1. El ensayo
# ----------------------------------------------------------------------
# Planta: 2 / (5s + 1) con dos segundos de tiempo muerto, discretizada por
# retenedor de orden cero. Se excita con una PRBS, que es la senal habitual en
# identificacion porque excita un rango amplio de frecuencias.
#
# El periodo de muestreo se elige como tau/15. No es un detalle: sobremuestrear
# acerca los polos discretos a z=1 y degrada el condicionamiento del problema,
# con el efecto doble de inflar el FIT por prediccion y arruinar la estimacion
# de los parametros fisicos.

K_VERDADERO, TAU_VERDADERO, THETA_VERDADERO = 2.0, 5.0, 2.0
TS = TAU_VERDADERO / 15.0
N = 1500

rng = np.random.default_rng(20261123)
u = np.sign(rng.standard_normal(N))

a = np.exp(-TS / TAU_VERDADERO)
b = K_VERDADERO * (1.0 - a)

# El retardo discreto total tiene dos componentes: el transporte fisico
# (theta/Ts = 6 muestras) y una muestra mas que introduce el propio retenedor
# de orden cero, porque la salida muestreada en k*Ts no puede reflejar la
# entrada aplicada en ese mismo instante. El retardo verdadero es 7, no 6.
nk_verdadero = int(round(THETA_VERDADERO / TS)) + 1

y = np.zeros(N)
for t in range(nk_verdadero, N):
    y[t] = a * y[t - 1] + b * u[t - nk_verdadero]

# Ruido de medicion: se suma directo a la salida, sin atravesar la planta.
# Es la hipotesis de Output Error.
y = y + 0.05 * rng.standard_normal(N)

datos = ids.DatosExperimento.desde_arrays(u, y, TS, nombre="FOPDT simulada")
print(datos)
print(f"verdad de terreno: K={K_VERDADERO}  tau={TAU_VERDADERO} s  "
      f"theta={THETA_VERDADERO} s  nk={nk_verdadero} muestras")
print(f"Ts = {TS:.6f} s  ({TAU_VERDADERO / TS:.0f} muestras por constante de tiempo)")

# Vale la pena validar antes de estimar.
ids.validar(datos)
print(f"muestras consecutivas identicas: {ids.contar_muestras_repetidas(datos.y)}")

# ----------------------------------------------------------------------
# 2. El retardo, por correlacion cruzada
# ----------------------------------------------------------------------
print("\n--- retardo ---")
identificador = ids.IdentificadorARX(datos)
print(f"lag de maxima correlacion cruzada: {identificador.lag_dominante()}")

nk, _ = identificador.buscar_mejor_nk(na=1, nb=1, ventana=4)
print(f"nk de mejor FIT: {nk}   (verdadero: {nk_verdadero})")

# ----------------------------------------------------------------------
# 3. Identificacion con las cuatro estructuras
# ----------------------------------------------------------------------
print("\n--- las cuatro estructuras al mismo orden ---")
print(f"{'estructura':<10} {'d':>2} {'FIT [%]':>9} {'AIC':>12} {'K':>9} {'err K':>8}")
for comparada in ids.comparar_las_cuatro(
    datos, na=1, nb=1, nc=1, nd=1, nf=1, nk=nk
):
    error_k = abs(comparada.ganancia - K_VERDADERO) / K_VERDADERO
    print(
        f"{comparada.metodo:<10} {comparada.n_params:>2} {comparada.fit:>9.4f} "
        f"{comparada.aic:>12.2f} {comparada.ganancia:>9.4f} {error_k:>7.2%}"
    )

# El ruido de este ensayo es de medicion, de modo que la hipotesis que
# corresponde es la de Output Error. Box-Jenkins contiene a OE como caso
# particular (C = D = 1), asi que tambien es una eleccion razonable: la
# diferencia es de dos parametros, no de hipotesis.

# ----------------------------------------------------------------------
# 4. El modelo elegido, en detalle
# ----------------------------------------------------------------------
print("\n--- Output Error ---")
identificador_oe = ids.IdentificadorOE(datos)
r = identificador_oe.estimar(nb=1, nf=1, nk=nk)

print(identificador_oe.obtener_ecuacion_str(r))
print(f"\nFIT      {r.fit:.4f} %")
print(f"AIC      {r.aic:.2f}  ({r.n_params} parametros)")
print(f"K        {r.ganancia:.6f}   (verdadero {K_VERDADERO})")

tau = -TS / np.log(abs(np.roots(r.F)[0]))
print(f"tau      {tau:.6f} s   (verdadero {TAU_VERDADERO})")
print(f"theta    {nk * TS:.6f} s   (verdadero {THETA_VERDADERO})")
print(f"         descontando la muestra del retenedor: {(nk - 1) * TS:.6f} s")
print(f"fin      {r.motivo_fin}, {r.iteraciones} iteraciones en "
      f"{r.arranques} arranques")

# En OE el predictor optimo es la simulacion, de modo que hay un solo FIT.
print(f"\ny_prediccion es {r.y_prediccion}: en OE el predictor optimo ES la simulacion")

# Verificacion de condicionamiento: si sum(F) es del orden de 1e-3, la ganancia
# y la constante de tiempo derivadas del modelo deben tomarse con reserva por
# alto que sea el FIT.
suma_f = float(np.sum(r.F))
print(f"sum(F) = {suma_f:.6f}  ->  "
      f"{'bien condicionado' if abs(suma_f) > 1e-2 else 'CUIDADO: mal condicionado'}")

# ----------------------------------------------------------------------
# 5. Eleccion del orden por AIC
# ----------------------------------------------------------------------
print("\n--- barrido de ordenes (OE) ---")
sugerencia = ids.SugeridorOrden(datos).sugerir(
    "OE", nk=nk, rango_planta=(1, 4)
)
print(f"recomendado: {sugerencia.recomendado.ordenes_str()}  "
      f"(d={sugerencia.recomendado.n_params})")
print(f"recorrido total del AIC: {sugerencia.recorrido_aic():.1f} unidades")
print("\nlos cinco mejores:")
for candidato in sugerencia.tabla_ordenada()[:5]:
    print(f"  {candidato}")

if sugerencia.fallidos:
    print("\ncombinaciones que no se pudieron estimar:")
    for fallido in sugerencia.fallidos:
        print(f"  {fallido.ordenes_str()}: {fallido.error}")

# La amplitud del barrido es un diagnostico: unas pocas unidades indican que el
# orden ya es suficiente; un descenso de cientos de unidades sin minimo avisa
# que la estructura elegida no describe el ruido del proceso y que no
# corresponde seguir agregando polos.
