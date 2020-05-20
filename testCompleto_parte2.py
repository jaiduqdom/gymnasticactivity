#!/usr/bin/env python
# -*- coding: utf-8 -*-
import textdistance
import csv
import cv2
import numpy as np

# Este segundo proceso permite aplicar la distancia de Levenshtein al problema de Viterbi
# Se ha probado con un video que dura 5:26 y se han obtenido 9741 secuencias en Markov
# La frecuencia es 9741 / (5*60+26) = 29.88Hz

FRECUENCIA = float(29.88)

# Leemos el fichero de Markov y lo transformamos en pares [[postura, repeticiones]*]
ENTRADA_MARKOV = "salidaMarkov_Gimnasia_Jaime_4.dat"
VIDEO_ENTRADA = "VIDEO_4_PASO_0_INICIAL.mp4"
VIDEO_SALIDA_EJERCICIOS = "VIDEO_4_PASO_2_EJERCICIOS_MARKOV_LEVENSHTEIN.avi"
n_posturas = 19

elementos = -1
POSTURA = [""] * 1000000
REPETICIONES = [int(0)] * 1000000

with open(ENTRADA_MARKOV) as File:
    reader = csv.reader(File, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        pos = str(row[0])
        if elementos == -1:
            elementos = 0
            POSTURA[elementos] = pos
            REPETICIONES[elementos] = 1
        else:
            if pos == POSTURA[elementos]:
                REPETICIONES[elementos] = REPETICIONES[elementos] + 1
            else:
                elementos = elementos + 1
                POSTURA[elementos] = pos
                REPETICIONES[elementos] = 1            
    elementos = elementos + 1

# Aplicamos un filtro para suprimir valores espúreos
##########################################################
for i in range(1, elementos):
    if REPETICIONES[i] <= 2:
        POSTURA[i] = POSTURA[i-1]
n_elementos = -1
n_POSTURA = [""] * 1000000
n_REPETICIONES = [int(0)] * 1000000   
for i in range(0, elementos):
    if i == 0:
        n_elementos = 0
        n_POSTURA[n_elementos] = POSTURA[i]
        n_REPETICIONES[n_elementos] = REPETICIONES[i]
        n_elementos = n_elementos + 1
    else:
        if POSTURA[i] == n_POSTURA[n_elementos-1]:
            n_REPETICIONES[n_elementos-1] = n_REPETICIONES[n_elementos-1] + REPETICIONES[i]
        else:
            n_POSTURA[n_elementos] = POSTURA[i]
            n_REPETICIONES[n_elementos] = REPETICIONES[i]
            n_elementos = n_elementos + 1
elementos = n_elementos
POSTURA = n_POSTURA
REPETICIONES = n_REPETICIONES
##########################################################

# Tabla para guardar los ejercicios que se han realizado
t_ejercicios = -1
t_tipo_ejercicio = [int(0)] * 1000
t_frame_inicio = [int(0)] * 1000
t_frame_final = [int(0)] * 1000

EJERCICIOS_POSIBLES = [[["C",2], ["B",2], ["J",1], ["A",3], ["J",1], ["B",2], ["C",2]],
[["C",2], ["B",2], ["J",1], ["A",1], ["Q",2], ["R",1], ["Q",1], ["S",1], ["Q",1], ["R",1], ["Q",1], ["S",1], ["Q",2], ["A",1], ["J",1], ["B",2], ["C",2]],
[["C",2], ["N",2], ["O",1], ["N",1], ["P",1], ["N",1], ["O",1], ["N",1], ["P",1], ["N",2], ["C",2]],
[["C",2], ["G",2], ["K",1], ["D",3], ["K",1], ["G",2], ["C",2]],
[["C",2], ["H",2], ["I",1], ["E",3], ["I",1], ["H",2], ["C",2]],
[["C",2], ["B",2], ["F",2], ["B",2], ["C",2]],
[["C",2], ["G",2], ["K",2], ["M",2], ["D",3], ["K",2], ["G",2], ["C",2]],
[["C",2], ["H",2], ["I",2], ["L",2], ["E",3], ["I",2], ["H",2], ["C",2]],
[["C",2], ["G",2], ["K",2], ["M",2], ["J",2], ["A",3], ["J",2], ["L",2], ["I",2], ["H",2], ["C",2]],
[["C",2], ["B",2], ["J",1], ["A",2], ["Q",2], ["A",2], ["Q",2], ["A",2], ["Q",2], ["A",2], ["J",1], ["B",2], ["C",2]]]

# Creamos una matriz de confusion: Valores Reales/Valores Obtenidos
n_ejercicios = len(EJERCICIOS_POSIBLES)
MATRIZ_CONFUSION = np.zeros((n_ejercicios, n_ejercicios), dtype=np.double)

# Relacion de posturas
POSTURAS = [ "Arms up",
            "Medium arms",
            "Arms down",
            "Left arm up",
            "Right arm up",
            "Arms forward",
            "Left arm in the middle",
            "Right arm in the middle",
            "Right arm half up/left dow",
            "Arms half up",            
            "Left arm half up/right down",
            "Right arm half up/left middle",
            "Left arm half up/right middle",
            "Arms akimbo",
            "Arms akimbo to the left",
            "Arms akimbo to the right",
            "Arms behind head",
            "Arms behind head to the left",
            "Arms behind head to the right"]

# Para facilitar la busqueda de Levenshtein, creamos un array de strings
e_posibles = [""] * len(EJERCICIOS_POSIBLES)
for i in range(0, len(EJERCICIOS_POSIBLES)):
    for j in range(0, len(EJERCICIOS_POSIBLES[i])):
        e_posibles[i] = e_posibles[i] + EJERCICIOS_POSIBLES[i][j][0]
    print(e_posibles[i])
    
# Ahora generamos un video con la salida
# Primero obtenemos la secuencia de posturas
def valor(dato):
    if dato == "A":
        return 0
    if dato == "B":
        return 1
    if dato == "C":
        return 2
    if dato == "D":
        return 3
    if dato == "E":
        return 4
    if dato == "F":
        return 5
    if dato == "G":
        return 6
    if dato == "H":
        return 7
    if dato == "I":
        return 8
    if dato == "J":
        return 9
    if dato == "K":
        return 10
    if dato == "L":
        return 11
    if dato == "M":
        return 12
    if dato == "N":
        return 13
    if dato == "O":
        return 14
    if dato == "P":
        return 15
    if dato == "Q":
        return 16
    if dato == "R":
        return 17
    if dato == "S":
        return 18
    return -1
    
"""def calcular_distancia(cadena1, cadena2):
    # Calculamos la distancia de Levenshtein y a mayores una distancia propia que mira a ver
    # si todas las posturas de la cadena del ejercicio optimo se han ejecutado
    d1 = textdistance.levenshtein.normalized_distance(cadena1, cadena2)

    v1 = [int(0)] * 100    
    v2 = [int(0)] * 100    

    for x in range(0, len(cadena1)):
        v1[valor(cadena1[x])] = 1
    
    for x in range(0, len(cadena2)):
        v2[valor(cadena2[x])] = 1
        
    c = float(0.0)
    t = float(0.0)
    for x in range(0, 100):
        if v2[x] == 1:
            t = t + 1
            if v1[x] == 1:
                c = c + 1
    d2 = c / t
    d_total = 0.5 * float(d1) + 0.5 * (1 - d2)
    return d_total
"""    
"""    
def calcular_distancia(cadena1, cadena2):
    # Calculamos la distancia de Levenshtein y a mayores una distancia propia que mira a ver
    # si todas las posturas de la cadena del ejercicio optimo se han ejecutado
    d1 = textdistance.levenshtein.normalized_distance(cadena1, cadena2)

    v1 = [int(0)] * 100    
    v2 = [int(0)] * 100    

    for x in range(0, len(cadena1)):
        v1[valor(cadena1[x])] = v1[valor(cadena1[x])] + 1
    
    for x in range(0, len(cadena2)):
        v2[valor(cadena2[x])] = v2[valor(cadena2[x])] + 1
        
    c = float(0.0)
    t = float(0.0)
    for x in range(0, 100):
        if v2[x] > 0:
            t = t + 1
            d_tmp = abs(float(v2[x]) - float(v1[x])) / (abs(float(v2[x]) - 0))
            if d_tmp > 1:
                d_tmp = 1
            c = c + d_tmp
    d2 = c / t
    d_total = 0.5 * float(d1) + 0.5 * (d2)
    return d_total    
"""

def calcular_distancia(cadena1, cadena2):
    # Calculamos la distancia de Levenshtein y a mayores una distancia propia que mira a ver
    # si todas las posturas de la cadena del ejercicio optimo se han ejecutado
    d1 = textdistance.levenshtein.normalized_distance(cadena1, cadena2)

    v1 = [int(0)] * 100    
    v2 = [int(0)] * 100    

    for x in range(0, len(cadena1)):
        v1[valor(cadena1[x])] = v1[valor(cadena1[x])] + 1
    
    for x in range(0, len(cadena2)):
        v2[valor(cadena2[x])] = v2[valor(cadena2[x])] + 1
        
    c = float(0.0)
    t = float(0.0)
    for x in range(0, 100):
        if v2[x] > 0:
            t = t + 1
            d_tmp = abs(float(v2[x]) - float(v1[x])) / (abs(float(v2[x]) - 0))
            if d_tmp > 1:
                d_tmp = 1
            c = c + d_tmp
    d2 = c / t
    
    c = float(0.0)
    t = float(0.0)
    d_tmp = float(0.0)
    for x in range(0, 100):
        t = t + v1[x]
        if v2[x] == 0 and v1[x] > 0:
            d_tmp = d_tmp + abs(float(v1[x]))
    d3 = d_tmp / t
    
    d_total = 0.5 * float(d1) + 0.25 * (d2) + 0.25 * (d3)
    #d_total = 0.5 * float(d1) + 0.5 * (d2)
    #d_total = 1 * float(d1)
    return d_total    

# Esta parte nos permite calcular la situacion de deteccion actual. Durante el ejercicio intenta
# detectar el ejercicio realizado
c_ejercicios = -1
c_elementos = [int(0)] * 1000 
c_tipo_ejercicio = [int(0)] * 1000
c_frame_origen = [int(0)] * 1000
c_frame_inicio = [int(0)] * 1000
c_frame_final = [int(0)] * 1000
c_frame_comienzo = 0

def procesar_cadena_temporal(s_elementos, S_POSTURA, frame_actual, frame_origen):
    global c_ejercicios, c_tipo_ejercicio, c_frame_inicio, c_frame_final, c_frame_comienzo, c_frame_origen
        
    # Procesamos un ejercicio encontrado para localizarlo en la tabla de ejercicios 
    # posible utilizando la cadena de Levenshtein
    cadena_a_buscar = ""
    for j in range(0, s_elementos):
        cadena_a_buscar = cadena_a_buscar + S_POSTURA[j]
    # print("Buscando cadena " + cadena_a_buscar)  

    # Evaluamos Levenshtein (al estar normalizado la distancia maxima es 1)
    # Utilizamos la libreria textdistance para evaluar Levenshtein:
    # https://pypi.org/project/textdistance/
    # a = textdistance.hamming('test', 'text')
    # print(str(a))
    # a = textdistance.levenshtein.normalized_distance('My string', 'My syrixg22')
    # print(str(a))      
    distancia_maxima = 1
    tipo = -1
    for j in range(0, len(EJERCICIOS_POSIBLES)):
        distancia = calcular_distancia(cadena_a_buscar, e_posibles[j])
        if distancia < distancia_maxima:
            distancia_maxima = distancia
            tipo = j
    #print("Tipo ejercicio: " + str(tipo + 1) + ", con distancia = " + str(distancia_maxima))
    
    # Podemos utilizar un umbral
    THRESHOLD = 1
    if distancia_maxima < THRESHOLD:
        c_ejercicios = c_ejercicios + 1
        c_tipo_ejercicio[c_ejercicios] = tipo + 1
        c_frame_inicio[c_ejercicios] = c_frame_comienzo
        c_frame_origen[c_ejercicios] = frame_origen        
        c_frame_final[c_ejercicios] = frame_actual
        c_elementos[c_ejercicios] = s_elementos
        c_frame_comienzo = frame_actual + 1        

def procesar_cadena(s_elementos, S_POSTURA, S_REPETICIONES, frame_actual):
    global t_ejercicios, t_tipo_ejercicio, t_frame_inicio, t_frame_final
        
    # Procesamos un ejercicio encontrado para localizarlo en la tabla de ejercicios 
    # posible utilizando la cadena de Levenshtein
    cadena_a_buscar = ""
    for j in range(0, s_elementos):
        cadena_a_buscar = cadena_a_buscar + S_POSTURA[j]
    print("Buscando cadena " + cadena_a_buscar)  

    # Evaluamos Levenshtein (al estar normalizado la distancia maxima es 1)
    # Utilizamos la libreria textdistance para evaluar Levenshtein:
    # https://pypi.org/project/textdistance/
    # a = textdistance.hamming('test', 'text')
    # print(str(a))
    # a = textdistance.levenshtein.normalized_distance('My string', 'My syrixg22')
    # print(str(a))      
    distancia_maxima = 1
    tipo = -1
    for j in range(0, len(EJERCICIOS_POSIBLES)):
        distancia = calcular_distancia(cadena_a_buscar, e_posibles[j])
        if distancia < distancia_maxima:
            distancia_maxima = distancia
            tipo = j
    print("Tipo ejercicio: " + str(tipo + 1) + ", con distancia = " + str(distancia_maxima))
    
    # Podemos utilizar un umbral
    THRESHOLD = 1
    if distancia_maxima < THRESHOLD:
        t_ejercicios = t_ejercicios + 1
        t_tipo_ejercicio[t_ejercicios] = tipo + 1
        t_frame_inicio[t_ejercicios] = frame_actual
        if t_ejercicios > 0:
            t_frame_inicio[t_ejercicios] = frame_actual
            t_frame_final[t_ejercicios - 1] = frame_actual - 1

# Dividimos las cadenas separadas por grupos de 20 Cs (brazos abajo)
# Los nuevos grupos se denominan S_POSTURA y S_REPETICIONES
cuenta = 0
inicio_encontrado = False
s_elementos = 0
S_POSTURA = [""] * 1000
S_REPETICIONES = [int(0)] * 1000
# Estas variables nos valen para controlar el frame de cara a generar un video
frame_actual = 0
frame_inicio = 0
for i in range(0, elementos):
    if inicio_encontrado == False:
        # Primero limpiamos el array temporal
        s_elementos = 0        
        S_POSTURA = [""] * 1000
        S_REPETICIONES = [int(0)] * 1000
        # Si es postura C y tiene 20 repeticiones comenzamos
        if POSTURA[i] == "C" and REPETICIONES[i] >= 20:
            inicio_encontrado = True
            frame_inicio = frame_actual
            c_frame_comienzo = frame_actual
            S_POSTURA[s_elementos] = POSTURA[i]
            S_REPETICIONES[s_elementos] = REPETICIONES[i]
            s_elementos = s_elementos + 1
    else:
        # El objetivo ahora es localizar el final de la candena, 
        # que estara compuesto por otras 20 C y que representaran el 
        # inicio de otra cadena
        # Al cerrar una cadena hay que procesarla
        S_POSTURA[s_elementos] = POSTURA[i]
        S_REPETICIONES[s_elementos] = REPETICIONES[i]
        s_elementos = s_elementos + 1
        
        # A partir de 5 elementos empezamos a mostrar el ejercicio que estimamos. Esto nos sirve para
        # crear la matriz de confusion
        if s_elementos >= 5:
            procesar_cadena_temporal(s_elementos, S_POSTURA, frame_actual, frame_inicio)
      
        if POSTURA[i] == "C" and REPETICIONES[i] >= 20:
            # En este caso hemos llegado al final de la cadena. La procesamos 
            # e iniciamos una nueva cadena
            procesar_cadena(s_elementos, S_POSTURA, S_REPETICIONES, frame_inicio)
            s_elementos = 0        
            S_POSTURA = [""] * 1000
            S_REPETICIONES = [int(0)] * 1000
            inicio_encontrado = True
            frame_inicio = frame_actual            
            c_frame_comienzo = frame_actual            
            S_POSTURA[s_elementos] = POSTURA[i]
            S_REPETICIONES[s_elementos] = REPETICIONES[i]
            s_elementos = s_elementos + 1            
    frame_actual = frame_actual + REPETICIONES[i]

   
N_FRAMES = 0
for i in range(0, elementos):
    N_FRAMES = N_FRAMES + REPETICIONES[i]
secuencia = [""] * (N_FRAMES)
posicion = 0
for i in range(0, elementos):
    for j in range(0, REPETICIONES[i]):
        secuencia[posicion] = POSTURA[i]
        posicion = posicion + 1
        
entrada = cv2.VideoCapture(VIDEO_ENTRADA)
# Generamos un video de salida
frame_width = int(entrada.get(3))
frame_height = int(entrada.get(4))  
salida = cv2.VideoWriter(VIDEO_SALIDA_EJERCICIOS, cv2.VideoWriter_fourcc('M','J','P','G'), FRECUENCIA, (frame_width,frame_height))
contador_frame = 0
contador_ejercicio = 0
contador_ejercicio_tmp = 0
texto_mostrar = ""
texto_mostrar_1 = ""
texto_mostrar_2 = ""
texto_velocidad = ""
texto_velocidad_1 = ""
texto_velocidad_2 = ""

ejercicio_real = -1
ejercicio_predicho = -1

while(True):
    # Capture frame-by-frame
    ret, frame = entrada.read()
    if ret == False or contador_frame >= N_FRAMES:
        break
    if contador_frame == t_frame_inicio[contador_ejercicio]:
            texto_mostrar_1 = str(t_tipo_ejercicio[contador_ejercicio])
            ejercicio_real = t_tipo_ejercicio[contador_ejercicio] - 1

            #############################################################################################            
            # Calculamos la velocidad de ejecucion
            #############################################################################################
            # El ejercicio desarrollado se debe realizar en los segundos que marca la tabla
            duracion_optima = float(0.0)
            for t in range(0, len(EJERCICIOS_POSIBLES[t_tipo_ejercicio[contador_ejercicio] - 1])):
                duracion_optima = duracion_optima + EJERCICIOS_POSIBLES[t_tipo_ejercicio[contador_ejercicio] - 1][t][1]
            # Duracion del ejercicio realizado:  (FRAME_FINAL - FRAME_INICIAL) / FRECUENCIA
            FRAME_INICIO = t_frame_inicio[contador_ejercicio]
            FRAME_FINAL = t_frame_final[contador_ejercicio]
            # Puede ser que el ultimo ejercicio no tenga numero de frame final
            if FRAME_FINAL == 0:
                FRAME_FINAL = N_FRAMES
            duracion_real = (float(FRAME_FINAL) - float(FRAME_INICIO)) / FRECUENCIA
            texto_velocidad_1 = ""
            if duracion_real > 1.1 * duracion_optima:
                texto_velocidad_1 = "Very quickly"
            else:
                if duracion_optima > 1.1 * duracion_real:
                    texto_velocidad_1 = "Very slowly"
                else:
                    texto_velocidad_1 = "Very well"
            #############################################################################################            
            
            contador_ejercicio = contador_ejercicio + 1
    if contador_frame == t_frame_final[contador_ejercicio]:
            texto_mostrar_1 = ""
            
    # Esta parte es para el control temporal del ejercicio            
    if contador_frame == c_frame_inicio[contador_ejercicio_tmp]:
            texto_mostrar_2 = str(c_tipo_ejercicio[contador_ejercicio_tmp])
            ejercicio_predicho = c_tipo_ejercicio[contador_ejercicio_tmp] - 1
            MATRIZ_CONFUSION[ejercicio_real][ejercicio_predicho] = MATRIZ_CONFUSION[ejercicio_real][ejercicio_predicho] + 1
            
            #############################################################################################            
            # Calculamos la velocidad de ejecucion para el caso temporal
            #############################################################################################
            # El ejercicio desarrollado se debe realizar en los segundos que marca la tabla
            duracion_optima = float(0.0)
            for t in range(0, len(EJERCICIOS_POSIBLES[c_tipo_ejercicio[contador_ejercicio_tmp] - 1])):
                if t > c_elementos[contador_ejercicio_tmp] - 1:
                    break
                duracion_optima = duracion_optima + EJERCICIOS_POSIBLES[c_tipo_ejercicio[contador_ejercicio_tmp] - 1][t][1]
            # Duracion del ejercicio realizado:  (FRAME_FINAL - FRAME_INICIAL) / FRECUENCIA
            FRAME_INICIO = c_frame_origen[contador_ejercicio_tmp]
            FRAME_FINAL = c_frame_final[contador_ejercicio_tmp]
            # Puede ser que el ultimo ejercicio no tenga numero de frame final
            duracion_real = (float(FRAME_FINAL) - float(FRAME_INICIO)) / FRECUENCIA
            texto_velocidad_2 = ""
            if duracion_real > 1.1 * duracion_optima:
                texto_velocidad_2 = "Very quickly"
            else:
                if duracion_optima > 1.1 * duracion_real:
                    texto_velocidad_2 = "Very slowly"
                else:
                    texto_velocidad_2 = "Very well"
            #############################################################################################            
            
    if contador_frame == c_frame_final[contador_ejercicio_tmp]:
            texto_mostrar_2 = ""
            texto_velocidad_2 = ""
            contador_ejercicio_tmp = contador_ejercicio_tmp + 1
            
    texto_postura = ("Frame: " + str(contador_frame) + ",  Posture: " + secuencia[contador_frame] +
        " - " + POSTURAS[valor(secuencia[contador_frame])] + ".")
    cv2.putText(frame, texto_postura, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)            
    
    texto_mostrar = "Performing exercise (final/temporal): " + texto_mostrar_1 + " / " + texto_mostrar_2
    texto_velocidad = "Velocity (final/temporal): " + texto_velocidad_1 + " / " + texto_velocidad_2
    
    cv2.putText(frame, texto_mostrar, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, texto_velocidad, (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.putText(frame, texto_mostrar_tmp, (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)        
    #cv2.putText(frame, texto_velocidad_tmp, (5, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)        
    
    contador_frame = contador_frame + 1    
    salida.write(frame)

# Normalizamos la matriz de confusion
for i in range(0, n_ejercicios):
    suma = float(0.0)
    for j in range(0, n_ejercicios):
        suma = suma + float(MATRIZ_CONFUSION[i][j])
    for j in range(0, n_ejercicios):
        MATRIZ_CONFUSION[i][j] = round(MATRIZ_CONFUSION[i][j] /suma, 2)
    
# Mostramos la matriz de confusion
cad = "["
n = n_ejercicios - 1
for i in range(0, n):
    cad = cad + "["
    for j in range(0, n):
        cad = cad + str(MATRIZ_CONFUSION[i][j]) + ", "
    cad = cad + str(MATRIZ_CONFUSION[i][n]) + "], \n"
cad = cad + "["
for j in range(0, n):
    cad = cad + str(MATRIZ_CONFUSION[n][j]) + ", "
cad = cad + str(MATRIZ_CONFUSION[n][n]) + "]] \n"
# Escribimos la matriz
print("Matriz de confusion")
print(cad)
