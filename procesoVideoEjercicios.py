#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Este proceso recibe un parametro de entrada: Video origen
# Por ejemplo: python procesoVideoEjercicios.py videoEntrada.mp4

# El proceso realiza distintas acciones:
#    1. Escribe un fichero videoEntrada_salidaRedNeuronal.dat con la salida de la red neuronal de
#       deteccion de posturas
#    2. Escribe un fichero videoEntrada_salidaMarkov.dat con la salida del modelo de Markov 
#       para filtrar las posiciones y obtener una secuencia ajustada
#    3. Escribe un video videoEntrada_EJERCICIOS_MARKOV_LEVENSHTEIN.avi con el resultado del proceso
#       indicando la postura, ejercicios detectados y evaluacion. Para ello utiliza la distancia
#       modificada de Levenshtein
#    4. Escribe un video intermedio videoEntrada_OPENPOSE.avi con el resultado del proceso OpenPose

# Es necesario tener OpenCV, Keras y Tensorflow
from numpy import argmax
import tensorflow as tf
import sys
import cv2
import numpy as np
import pandas as pd
import os
import math
from angulosLongitudes import angulosLongitudes
import traceback
from time import time
import textdistance
import csv

# Verificar la entrada
VIDEO_ENTRADA = ""
if len(sys.argv) == 2:
    VIDEO_ENTRADA = sys.argv[1]
else:
    print("ERROR: El proceso necesita especificar el video de entrada:")
    print("Ejemplo: python procesoVideoEjercicios.py videoEntrada.mp4.")
    print("SALIDA: El proceso realiza distintas acciones:")
    print("1. Escribe un fichero videoEntrada_salidaRedNeuronal.dat con la salida de la red neuronal de deteccion de posturas.")
    print("2. Escribe un fichero videoEntrada_salidaMarkov.dat con la salida del modelo de Markov para filtrar las posiciones y obtener una secuencia ajustada.")
    print("3. Escribe un video videoEntrada_EJERCICIOS_MARKOV_LEVENSHTEIN.avi con el resultado del proceso indicando la postura, ejercicios detectados y evaluacion. Para ello utiliza la distancia modificada de Levenshtein.")
    print("4. Escribe un video intermedio videoEntrada_OPENPOSE.avi con el resultado del proceso OpenPose.")

filename = os.path.splitext(VIDEO_ENTRADA)[0]
SALIDA_RED_NEURONAL = filename + "_salidaRedNeuronal.dat"
SALIDA_MARKOV = filename + "_salidaMarkov.dat"
VIDEO_SALIDA_EJERCICIOS = filename + "_EJERCICIOS_MARKOV_LEVENSHTEIN.avi"

# Este es un video temporal de deteccion de posturas incluyendo la salida de OpenPose. 
ESCRIBIR_VIDEO_SALIDA = True
VIDEO_SALIDA = filename + "_OPENPOSE.avi"

# Luego corregimos la frecuencia con la real
FRECUENCIA = float(29.88)
""" 
19 posiciones: Las posturas se encuentran en el paper            
"""
n_posturas = 19

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

# Con esto indicamos que debemos repartir el uso de la memoria de Tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
#K.set_session(sess)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config, ...)

# model = 0
# define model
# with tf.device('/cpu:0'):
model = tf.keras.models.load_model('/home/disa/skeleton/ENTRENAMIENTO/modeloPosturas.h5')
# summarize model.
model.summary()

# Puntos devueltos OpenPose
keyPoints = 25     

# Funciones para calcular las longitures y ángulos
aL = angulosLongitudes()    

########################################################################################
########################################################################################
# Como los datos devueltos por OpenPose modifican el identificador de la persona,
# es necesario llevar una tabla interna que nos indique el identificador correcto de 
# la persona a partir de las coordenadas del cuello de la persona detectada
# Es un sistema de tracking por encima de OpenPose
########################################################################################
########################################################################################

# Tabla de asociacion de personas
# Nos permite asociar un esqueleto a una posicion independientemente 
# del valor devuelto por OpenPose
maxPers = 20
#persona = [int(0)] * maxPers
cuello_x = np.zeros(maxPers)
cuello_y = np.zeros(maxPers)
activo = [int(0)] * maxPers 
textoPers = [""] * maxPers
# La entrada son las coordenadas (x,y) del cuello de la persona segun OpenPose
def buscarPersona(c_x, c_y):
    global maxPers
    global cuello_x
    global cuello_y
    global activo
    # Persona seleccionada
    persona_seleccionada = -1
    # Aumentamos los activadores que permiten detectar actividad de una persona            
    for i in range(0, maxPers):
        activo[i] = activo[i] + 1
    # Localizar la persona mas proxima
    minimo = 9999999
    proximo = -1
    for i in range(0, maxPers):
        distancia = math.sqrt( (c_x - cuello_x[i])**2 + (c_y - cuello_y[i])**2 )
        if cuello_x[i] != 0 and cuello_y[i] != 0 and distancia < minimo:
            minimo = distancia
            proximo = i
    # Establecemos 100 pixeles de separacion entre frames como umbral de deteccion de persona
    if minimo < 100:
        persona_seleccionada = proximo
        cuello_x[proximo] = c_x
        cuello_y[proximo] = c_y
        activo[proximo] = 0
    # Si no hay ninguna persona seleccionada, buscamos la primera que tiene ceros en las coordenadas o la que mas tiempo inactiva lleva
    if persona_seleccionada == -1:
        for i in range(0, maxPers):
            if cuello_x[i] == 0 and cuello_y[i] == 0:
                persona_seleccionada = i
                cuello_x[i] = c_x
                cuello_y[i] = c_y
                activo[i] = 0                        
                break
    if persona_seleccionada == -1:
        maximo = 0
        proximo = -1
        for i in range(0, maxPers):
            if activo[i] > maximo:
                maximo = activo[i]
                proximo = i
        persona_seleccionada = proximo
        cuello_x[proximo] = c_x
        cuello_y[proximo] = c_y
        activo[proximo] = 0
    return persona_seleccionada
########################################################################################
########################################################################################

########################################################################################
########################################################################################
# Algoritmo de Viterbi
########################################################################################
########################################################################################
# Implementacion del algoritmo de Viterbi HMM
def viterbi(pi, a, d, T):
   
    start_time = time()   
   
    nStates = np.shape(d)[0]
    #print('Nstates\n',nStates)
    #T = np.shape(obs)[0]
    #print('T\n',T)
   
    # init blank path
    path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T), dtype=np.double)

    #print('delta\n',delta)
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T), dtype=np.double)
   
    # init delta and phi
    #delta[:, 0] = pi * b[:, obs[0]]
    delta[:, 0] = pi * d[:, 0]
    phi[:, 0] = 0

    #print('delta\n',delta)
    #print('Phi\n',phi)

    #print('\nStart Walk Forward\n')   
    # the forward algorithm extension
    for t in range(1, T):
        deltasum=0
        for s in range(nStates):
            #delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            delta[s, t] = np.double(np.max(delta[:, t-1] * a[:, s])) * d[s, t]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            #print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
            #print(str(phi[s, t]))
            #print(delta[s, t])
            deltasum=deltasum+delta[s,t]
        for s in range(nStates):
            delta[s,t]=delta[s,t]/deltasum

        # Control de error           
        """positivo = False           
        for s in range(nStates):           
            if phi[s, t] != 0.0:
                positivo = True
                break
        if positivo == False:
            print("Error en t " + str(t))       
            sys.exit(-1)       """
           
   
    # find optimal path
    #print('-'*50)
    #print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #print('path[{}] = {}'.format(t, path[t]))
        
    elapsed_time = time() - start_time
    #print("Tiempo")
    #print(elapsed_time)
       
    return path, delta, phi
    
def valorInv(dato):
    if dato == 0:
        return "A"
    if dato == 1:
        return "B"
    if dato == 2:
        return "C"
    if dato == 3:
        return "D"
    if dato == 4:
        return "E"
    if dato == 5:
        return "F"
    if dato == 6:
        return "G"
    if dato == 7:
        return "H"
    if dato == 8:
        return "I"
    if dato == 9:
        return "J"
    if dato == 10:
        return "K"
    if dato == 11:
        return "L"
    if dato == 12:
        return "M"
    if dato == 13:
        return "N"
    if dato == 14:
        return "O"
    if dato == 15:
        return "P"
    if dato == 16:
        return "Q"
    if dato == 17:
        return "R"
    if dato == 18:
        return "S"
    return ""
########################################################################################
########################################################################################

########################################################################################
# Obtener la secuencia de las posturas llevadas a cabo utilizando OpenPose y la red
# neuronal entrenada
########################################################################################
SEC_POSTURAS = [int(0)] * 1000000
SEC_FRAMES = [int(0)] * 1000000
TOTAL_SEC = 0
DESC_POSTURA = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S"]
n_posturas = len(POSTURAS)
# Guarda los datos de salida de la red para montar las observaciones de Viterbi
N_FRAMES = 0
SALIDA_RED = np.zeros((1000000, n_posturas))
########################################################################################
########################################################################################
########################################################################################
########################################################################################

print("Leyendo video y escribiendo salida red neuronal (persona 0)...")

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/usr/local/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="./izquierda.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    # args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/disa/skeleton/openpose/models/"
    params["face"] = False
    params["hand"] = False

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Arrancamos el video
    cap = cv2.VideoCapture(VIDEO_ENTRADA)
    FRECUENCIA = cap.get(cv2.CAP_PROP_FPS)    
    
    # Generamos un video de salida
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))    
    
    if ESCRIBIR_VIDEO_SALIDA == True:
        salida = cv2.VideoWriter(VIDEO_SALIDA, cv2.VideoWriter_fourcc('M','J','P','G'), FRECUENCIA, (frame_width,frame_height))
        # cv2.VideoWriter('videoSalidaTransiciones.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
    
    postura_anterior = -1 
    
    # La variable repeticiones nos permite tener en cuenta cuantas repeticiones de una misma postura deben ser tenidas en cuenta para
    # registrar dos movimientos en la matriz de transicion
    repeticiones = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break        
        height, width, channels = frame.shape 
    
        #print(height)
        #print(width)        
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Process Image
        datum = op.Datum()
        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])
        
        #cv2.imshow('entrada', frame)        
        
        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #print("Face keypoints: \n" + str(datum.faceKeypoints))
        #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        #print("Right hand keypoints: \n", str(datum.handKeypoints[1][0][0]))
        
        # 3 posiciones:
        #   OK: A
        #   Abierta: B
        #   Cerrada: C
        
        #posicion = "C"
        numeroPersonas = 0
        if datum:
            if datum.poseKeypoints != []:
                numeroPersonas = len(datum.poseKeypoints)
        # print("Numero personas = " + str(numeroPersonas))

        # Display the resulting frame
        imagenSalida = datum.cvOutputData

        texto_y = int(15)

        # Limpiamos las variables de texto de cada persona
        for p in range(0, maxPers):
            textoPers[p] = ""

        for p in range(0, numeroPersonas):
            
            # Para cada persona detectada, deben existir las coordenadas del cuello
            if ((datum.poseKeypoints[p][1][0] != 0) and (datum.poseKeypoints[p][1][1] != 0)):
                
                persona_seleccionada = buscarPersona(datum.poseKeypoints[p][1][0], datum.poseKeypoints[p][1][1])

                pos_x = datum.poseKeypoints[p][0][0]
                pos_y = datum.poseKeypoints[p][0][1]
                pos_x = pos_x - 40
                if (pos_x < 0):
                    pos_x = 0
                
                pos_y = pos_y - 50
                if (pos_y < 0):
                    pos_y = 0
           
                cv2.putText(imagenSalida, "Person " + str(persona_seleccionada), (int(pos_x), int(pos_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Obtenemos el vector con las longitudes y los angulos
                datos = aL.obtenerAngulosLongitudes(datum.poseKeypoints, p)
                
                if datos != []:
                    yhat = model.predict([[datos]])
                    textoPers[persona_seleccionada] = "Person " + str(persona_seleccionada) + " - " + POSTURAS[argmax(yhat)]
                    
                    # Actualizamos la matriz de transicion
                    if p == 0:
                        # Guardamos datos para Viterbi
                        ##################################
                        SALIDA_RED[N_FRAMES] = yhat
                        # N_FRAMES = N_FRAMES + 1
                        ##################################
                if p == 0:
                        N_FRAMES = N_FRAMES + 1

        # Escribimos los textos de cada persona una vez que han sido obtenidos
        # No escribimos este texto ya que sera usado como entrada del apartado de evaluacion
        # del ejercicio
        """for p in range(0, maxPers):
            if textoPers[p] != "":
                cv2.putText(imagenSalida, textoPers[p], (5, texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                texto_y = texto_y + 20"""
        
        # Mostramos la imagen de salida
        # cv2.imshow('Brazos', imagenSalida)
        if ESCRIBIR_VIDEO_SALIDA == True:
            salida.write(imagenSalida)
        
        # Hand Pose        
        # https://medium.com/@prasad.pai/classification-of-hand-gesture-pose-using-tensorflow-30e83064e0ed        
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        #cv2.imwrite("salida.jpg", datum.cvOutputData)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    if ESCRIBIR_VIDEO_SALIDA == True:
        salida.release()
    
    # Escribiendo la salida de la red neuronal a un fichero de salida. (Persona 0)
    with open(SALIDA_RED_NEURONAL, 'wt') as f:
        cadena = ""
        for i in range(0, N_FRAMES):
            for j in range(0, n_posturas - 1):
                cadena = cadena + str(SALIDA_RED[i][j]) + ", "
            cadena = cadena + str(SALIDA_RED[i][n_posturas - 1]) + '\n'
        f.write(cadena)
    
    ########################################################################################
    # Obtener la secuencia de Viterbi a partir de los datos de la red neuronal obtenidos
    ########################################################################################    
    print("Procesando Viterbi sobre los datos (persona 0)...")
    
    # Definimos la matriz de transicion entre estados
    states = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']
    hidden_states = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S']
    pi = [np.float128(1.0/n_posturas)] * n_posturas
    state_space = pd.Series(pi, index=hidden_states, name='states')
    a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
    
    # Definimos la matriz de transicion
    M_T = [[0.4,0.1,0,0,0,0.1,0,0,0,0.1,0,0.1,0.1,0,0,0,0.1,0,0],
    [0.03,0.49,0.03,0.03,0.03,0.03,0.03,0.03,0.01,0.2,0.01,0.02,0.02,0.01,0,0,0.03,0,0],
    [0,0.2,0.5,0,0,0.1,0.05,0.05,0,0,0,0,0,0.1,0,0,0,0,0],
    [0,0.1,0,0.5,0,0,0.2,0,0,0,0.1,0,0.1,0,0,0,0,0,0],
    [0,0.1,0,0,0.5,0,0,0.2,0.1,0,0,0.1,0,0,0,0,0,0,0],
    [0.15,0.15,0.15,0,0,0.55,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0.15,0.15,0.15,0,0,0.35,0,0,0,0.15,0,0.05,0,0,0,0,0,0],
    [0,0.15,0.15,0,0.15,0,0,0.35,0.15,0,0,0.05,0,0,0,0,0,0,0],
    [0,0.1,0,0,0.1,0,0,0.1,0.6,0,0,0.1,0,0,0,0,0,0,0],
    [0.1,0.1,0,0,0,0.1,0,0,0,0.4,0,0.1,0.1,0,0,0,0.1,0,0],
    [0,0.1,0,0.1,0,0,0.1,0,0,0,0.6,0,0.1,0,0,0,0,0,0],
    [0.1,0.1,0.1,0,0.1,0,0,0.1,0.1,0.1,0,0.2,0,0,0,0,0.1,0,0],
    [0.1,0.1,0.1,0.1,0,0,0.1,0,0,0.1,0.1,0,0.2,0,0,0,0.1,0,0],
    [0,0.08,0.12,0,0,0.1,0.05,0.05,0,0,0,0,0,0.3,0.15,0.15,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0.6,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0,0.6,0,0,0],
    [0.08,0.08,0,0,0,0,0,0,0,0.18,0,0.08,0.08,0,0,0,0.3,0.1,0.1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0.6,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0,0.6]]
    
    for i in range(0, n_posturas):
        a_df.loc[hidden_states[i]] = M_T[i]
        
    print("\n HMM matrix:\n", a_df)
    a = a_df.values
    a = a.astype('float128')
        
    # Montamos las observaciones
    obs_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':18 }
    
    #######################################################################################################
    # En este caso vamos a procesar el video por partes, creando 10 frames por atras y 10 por delante
    # para ir calculando a partir del frame 11
    #######################################################################################################
    OBSERVADOS_ANTES = 10
    OBSERVADOS_DESPUES = 10
    TOTAL_OBSERVADOS = OBSERVADOS_ANTES + 1 + OBSERVADOS_DESPUES
    
    # En path total escribiremos la salida completa
    path_total = np.zeros(N_FRAMES, dtype=int)
    
    # Ventana de trabajo con las observaciones anteriores, la actual y las posteriores
    observaciones_ventana = np.zeros((TOTAL_OBSERVADOS, n_posturas))
    
    for x in range(0, N_FRAMES):
        # desplazamos todas las observaciones hacia la izquierda para añadir la leida al final
        for y in range(1, TOTAL_OBSERVADOS):
            observaciones_ventana[y-1] = observaciones_ventana[y]
        observaciones_ventana[TOTAL_OBSERVADOS - 1] = SALIDA_RED[x]
        
        # cuando x >= TOTAL_OBSERVADOS - 1 podemos empezar a llamar a Viterbi
        if x >= TOTAL_OBSERVADOS - 1:
    
            # Creamos la secuencia de observaciones a partir de la misma salida de la RN.
            obs = [int(0)] * TOTAL_OBSERVADOS
            for z in range(0, TOTAL_OBSERVADOS):
                obs[z] = argmax(observaciones_ventana[z])
        
            inv_obs_map = dict((v,k) for k, v in obs_map.items())
            obs_seq = [inv_obs_map[v] for v in list(obs)]
            
            observable_states = states        
            
            # Vector pi inicial
            pi = [np.float128(1.0/n_posturas)] * n_posturas
        
            # Creo la matriz de observaciones. 
            T = np.shape(obs)[0]
            nStates = n_posturas
        
            d = np.zeros((nStates, T), dtype=np.float128)
            for t in range(0, T):
                for s in range(nStates):
                    d[s,t] = observaciones_ventana[t][s]
        
            path, delta, phi = viterbi(pi, a, d, T)
            
            # Añadimos el path al total. Si x = TOTAL_OBSERVADOS - 1 entonces añadimos todas las medidas hasta
            # OBSERVADOS_ANTES + 1. Si x = N_FRAMES - 1, entonces añadimos el resto de Markov a la salida 
            if x == TOTAL_OBSERVADOS - 1:
                for z in range(0, OBSERVADOS_ANTES + 1):
                    path_total[z] = path[z]
            else:
                if x == N_FRAMES - 1:
                    for z in range(0, OBSERVADOS_DESPUES + 1):
                        path_total[x - OBSERVADOS_DESPUES + z] = path[OBSERVADOS_ANTES + z]
                else:
                    path_total[x - OBSERVADOS_DESPUES] = path[OBSERVADOS_ANTES]
    
    # Escribiendo la salida de la red neuronal a un fichero de salida. (Persona 0)
    with open(SALIDA_MARKOV, 'wt') as f:
        cadena = ""
        for i in range(0, N_FRAMES):
            cadena = cadena + valorInv(path_total[i]) + '\n'
        f.write(cadena)
        
    print("Fin del proceso de obtencion de Markov. Total de Frames = " + str(N_FRAMES))
    
except Exception as e:
    traceback.print_exc(file=sys.stdout)    
    print(e)
    raise e
    sys.exit(-1)

###############################################################################################
###############################################################################################
# Este segundo proceso permite aplicar la distancia de Levenshtein al problema de Viterbi
###############################################################################################
###############################################################################################
# Leemos el fichero de Markov y lo transformamos en pares [[postura, repeticiones]*]
ENTRADA_MARKOV = SALIDA_MARKOV
VIDEO_ENTRADA = VIDEO_SALIDA

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
if ESCRIBIR_VIDEO_SALIDA == True:
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
    if ESCRIBIR_VIDEO_SALIDA == True:    
        salida.write(frame)

if ESCRIBIR_VIDEO_SALIDA == True:
    salida.release()

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
print("Proceso finalizado")
