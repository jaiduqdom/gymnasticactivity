#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Este primer proceso lee completamente un video y genera un fichero llamado SALIDA_MARKOV con las posiciones 
# detectadas y el numero de frames. Este video es utilizado posteriormente por testCompleto_parte2 para calcular
# las distancias de Levenshtein

# Procesamos el video y utilizamos la red neuronal entrenada para obtener las posiciones y Markov para filtrar las
# posiciones y obtener una secuencia ajustada

# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# mlp for binary classification
# mlp for multiclass classification
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import numpy as np
import pandas as pd
import os
from sys import platform
import math
from angulosLongitudes import angulosLongitudes
from keras import backend as K
import traceback
from time import time

VIDEO_ENTRADA = "VIDEO_4_PASO_0_INICIAL.mp4"
VIDEO_SALIDA = "VIDEO_4_PASO_1_POSTURAS_OPENPOSE_MLP.avi"
SALIDA_RED_NEURONAL = "salidaRedNeuronal_Gimnasia_Jaime_4.dat"
SALIDA_MARKOV = "salidaMarkov_Gimnasia_Jaime_4.dat"
ESCRIBIR_VIDEO_SALIDA = True

""" 

19 posiciones: Las posturas se encuentran en el paper            
"""

# Relacion de posturas
POSTURAS = [ "Brazos arriba",
             "Brazos medio",
             "Brazos abajo",
             "Brazo izquierdo arriba",
             "Brazo derecho arriba",             
             "Brazos hacia delante",
             "Brazo izquierdo en el medio",
             "Brazo derecho en el medio",
             "Brazo derecho medio arriba (i)",             
             "Brazos medio arriba",
             "Brazo izquierdo medio arriba (k)",
             "Brazo derecho medio arriba (l)",
             "Brazo izquierdo medio arriba (m)",
             "Brazos en jarras",
             "Brazos en jarras hacia la derecha",
             "Brazos en jarras hacia la izquierda",             
             "Brazos cruzados detras cabeza",
             "Brazos cruzados detras cabeza hacia la derecha",
             "Brazos cruzados detras cabeza hacia la izquierda"]

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
    print("Tiempo")
    print(elapsed_time)
       
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
    
    # Generamos un video de salida
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))    
    
    if ESCRIBIR_VIDEO_SALIDA == True:
        salida = cv2.VideoWriter(VIDEO_SALIDA, cv2.VideoWriter_fourcc('M','J','P','G'), 29.88, (frame_width,frame_height))
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
           
                cv2.putText(imagenSalida, "Persona " + str(persona_seleccionada), (int(pos_x), int(pos_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Obtenemos el vector con las longitudes y los angulos
                datos = aL.obtenerAngulosLongitudes(datum.poseKeypoints, p)
                
                if datos != []:
                    yhat = model.predict([[datos]])
                    textoPers[persona_seleccionada] = "Persona " + str(persona_seleccionada) + " - " + POSTURAS[argmax(yhat)]
                    
                # Actualizamos la matriz de transicion
                if p == 0:
                    # Guardamos datos para Viterbi
                    ##################################
                    SALIDA_RED[N_FRAMES] = yhat
                    N_FRAMES = N_FRAMES + 1
                    ##################################                    

        # Escribimos los textos de cada persona una vez que han sido obtenidos
        for p in range(0, maxPers):
            if textoPers[p] != "":
                cv2.putText(imagenSalida, textoPers[p], (5, texto_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                texto_y = texto_y + 20
        
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
    
    """[[0.4,0.1,0,0.1,0.1,0.00,0,0,0,0.10,0,0,0.1,0.1,0,0,0,0,0],
    [0.03,0.49,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.2,0.01,0.01,0.01,0.01,0.03,0,0,0,0],
    [0,0.2,0.5,0,0,0,0.1,0.05,0.05,0,0,0,0,0,0.1,0,0,0,0],
    [0,0.1,0,0.5,0,0,0.1,0.1,0,0,0,0.1,0,0.1,0,0,0,0,0],
    [0,0.1,0,0,0.5,0,0.1,0,0.1,0,0.1,0,0.1,0,0,0,0,0,0],
    [0.1,0.0,0,0,0,0.3,0,0,0,0.2,0,0,0.1,0.1,0,0,0,0.1,0.1],
    [0.1,0.1,0.1,0,0,0,0.6,0,0,0.1,0,0,0,0,0,0,0,0,0],
    [0,0.15,0.15,0.15,0,0,0,0.35,0,0,0,0.1,0,0.1,0,0,0,0,0],
    [0,0.15,0.15,0,0.15,0,0,0,0.35,0,0.1,0,0.1,0,0,0,0,0,0],
    [0.1,0.1,0,0,0,0.1,0.1,0,0,0.4,0,0,0.1,0.1,0,0,0,0,0],
    [0,0.1,0,0,0.1,0,0,0,0.1,0,0.6,0,0.1,0,0,0,0,0,0],
    [0,0.1,0,0.1,0,0,0,0.1,0,0,0,0.6,0,0.1,0,0,0,0,0],
    [0.1,0.1,0.1,0,0.1,0.1,0,0,0.1,0.1,0.1,0,0.2,0,0,0,0,0,0],
    [0.1,0.1,0.1,0.1,0,0.1,0,0.1,0,0.1,0,0.1,0,0.2,0,0,0,0,0],
    [0,0.1,0.1,0,0,0,0.1,0.1,0.1,0,0,0,0,0,0.3,0.1,0.1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0.6,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4,0,0.6,0,0],
    [0,0,0,0,0,0.4,0,0,0,0,0,0,0,0,0,0,0,0.6,0],
    [0,0,0,0,0,0.4,0,0,0,0,0,0,0,0,0,0,0,0,0.6]]"""
    
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
        
    """# Agrupamos para otro array temporal
    P = [""] * 100000
    R = [int(0)] * 100000    
    e = -1
    for i in range(0, N_FRAMES):
        pos = valorInv(path_total[i])
        if e == -1:
            e = 0
            P[e] = valorInv(path_total[i])
            R[e] = 1
            e = e + 1
        else:
            if P[e-1] == valorInv(path_total[i]):
                R[e-1] = R[e-1] + 1
            else:
                P[e] = valorInv(path_total[i])
                R[e] = 1
                e = e + 1
    cadena = ""
    for j in range(0, e):
        cadena = cadena + "['" + P[j] + "', " + str(R[j]) + "], "
    print (cadena)"""
        
        
    print("Fin del proceso. Total de Frames = " + str(N_FRAMES))
    
except Exception as e:
    traceback.print_exc(file=sys.stdout)    
    print(e)
    raise e
    sys.exit(-1)
    
"""    
    # Calculamos los resultados
    DATOS = [["C", 253, "C"],["H", 62, "C"],["B", 5, "B"],["H", 1, "B"],["B", 3, "B"],["H", 1, "B"],["B", 70, "B"],["M", 1, "B"],
    ["B", 1, "B"],["M", 2, "B"],["B", 6, "B"],["M", 1, "B"],["B", 94, "B"],["M", 1, "B"],["B", 5, "B"],["M", 2, "B"],
    ["B", 46, "B"],["N", 7, "B"],["J", 3, "J"],["M", 2, "J"],["J", 159, "J"],["A", 338, "A"],["E", 260, "E"],["K", 1, "K"],
    ["E", 1, "K"],["K", 72, "K"],["I", 91, "I"],["C", 94, "C"],["O", 12, "G"],["G", 142, "G"],["B", 155, "B"],["A", 1, "A"],
    ["B", 4, "A"],["A", 134, "A"],["Q", 1, "A"],["A", 1, "A"],["Q", 1, "A"],["A", 1, "A"],["Q", 1, "A"],["A", 1, "A"],
    ["Q", 1, "A"],["A", 3, "A"],["Q", 2, "A"],["A", 1, "A"],["Q", 4, "A"],["A", 36, "A"],["Q", 1, "A"],["A", 1, "A"],
    ["Q", 1, "A"],["A", 3, "A"],["F", 9, "A"],["J", 112, "J"],["F", 137, "F"],["R", 133, "R"],["F", 22, "F"],["R", 1, "F"],
    ["F", 30, "F"],["S", 1, "S"],["F", 1, "S"],["S", 1, "S"],["F", 3, "S"],["S", 135, "S"],["F", 69, "F"],["R", 106, "R"],
    ["F", 70, "F"],["S", 1, "F"],["F", 5, "F"],["S", 128, "S"],["F", 11, "F"],["B", 3, "F"],["J", 23, "J"],["M", 2, "B"],
    ["B", 2, "B"],["M", 10, "B"],["B", 89, "B"],["I", 16, "B"],["C", 2, "C"],["H", 1, "C"],["C", 43, "C"],["O", 136, "O"],
    ["P", 13, "P"],["O", 1, "P"],["P", 83, "P"],["O", 7, "O"],["P", 3, "O"],["O", 1, "O"],["P", 7, "O"],["O", 83, "O"],
    ["Q", 2, "Q"],["O", 1, "Q"],["Q", 1, "Q"],["O", 1, "Q"],["Q", 116, "Q"],["O", 7, "O"],["Q", 1, "O"],["O", 77, "O"],
    ["P", 8, "P"],["O", 1, "P"],["P", 1, "P"],["O", 1, "P"],["P", 52, "P"],["O", 4, "O"],["P", 1, "O"],["O", 7, "O"],
    ["P", 1, "O"],["O", 1, "O"],["P", 5, "O"],["O", 78, "O"],["C", 92, "C"],["I", 6, "I"],["C", 1, "I"],["I", 44, "I"],
    ["K", 82, "K"],["M", 78, "M"],["J", 98, "J"],["F", 18, "A"],["A", 1, "A"],["F", 38, "A"],["Q", 4, "A"],["S", 1, "A"],
    ["A", 3, "A"],["Q", 1, "A"],["A", 4, "A"],["Q", 1, "A"],["A", 34, "A"],["Q", 1, "A"],["A", 1, "A"],["Q", 1, "A"],
    ["A", 46, "A"],["Q", 1, "A"],["A", 10, "A"],["Q", 1, "A"],["A", 1, "A"],["Q", 3, "A"],["A", 7, "A"],["Q", 1, "A"],
    ["A", 19, "A"],["Q", 1, "A"],["A", 3, "A"],["F", 2, "A"],["S", 1, "A"],["F", 8, "A"],["L", 2, "A"],["F", 4, "A"],
    ["L", 1, "A"],["S", 1, "A"],["F", 1, "A"],["L", 3, "A"],["F", 3, "A"],["L", 1, "A"],["F", 1, "A"],["L", 1, "A"],
    ["F", 2, "A"],["L", 3, "A"],["F", 1, "A"],["L", 1, "A"],["F", 14, "A"],["L", 1, "A"],["F", 11, "A"],["N", 93, "N"],
    ["L", 78, "L"],["H", 58, "H"],["C", 62, "C"],["H", 20, "B"],["B", 49, "B"],["N", 5, "B"],["J", 83, "J"],["F", 1, "A"],
    ["J", 1, "A"],["F", 1, "A"],["A", 39, "A"],["F", 5, "A"],["A", 40, "A"],["F", 12, "A"],["A", 1, "A"],["F", 10, "A"],
    ["J", 1, "A"],["F", 1, "A"],["J", 16, "A"],["N", 12, "A"],["D", 2, "D"],["N", 1, "D"],["D", 69, "D"],["L", 1, "D"],
    ["D", 4, "D"],["L", 15, "N"],["N", 4, "N"],["I", 1, "I"],["H", 1, "I"],["I", 25, "I"],["K", 41, "K"],["E", 106, "E"],
    ["B", 1, "B"],["E", 1, "B"],["B", 155, "B"],["I", 17, "C"],["C", 3, "C"],["I", 1, "C"],["C", 44, "C"],["O", 9, "G"],
    ["G", 143, "G"],["A", 203, "A"],["G", 132, "G"],["O", 11, "C"],["C", 51, "C"],["H", 20, "B"],["B", 73, "B"],["A", 147, "A"],
    ["Q", 1, "A"],["A", 25, "A"],["F", 17, "J"],["J", 63, "J"],["M", 1, "J"],["J", 1, "J"],["M", 1, "B"],["B", 6, "B"],
    ["M", 1, "B"],["B", 3, "B"],["M", 1, "B"],["B", 75, "B"],["I", 18, "C"],["C", 51, "C"],["O", 113, "O"],["P", 1, "P"],
    ["O", 1, "P"],["P", 6, "P"],["O", 3, "P"],["P", 3, "P"],["O", 1, "P"],["P", 72, "P"],["O", 3, "O"],["P", 5, "O"],
    ["O", 53, "O"],["Q", 86, "Q"],["O", 104, "O"],["G", 9, "J"],["J", 55, "J"],["F", 2, "F"],["J", 1, "F"],["F", 86, "F"],
    ["R", 81, "R"],["F", 47, "F"],["S", 87, "S"],["F", 58, "F"],["A", 147, "A"],["P", 2, "A"],["A", 33, "A"],["Q", 3, "A"],
    ["A", 1, "A"],["Q", 1, "A"],["A", 39, "A"],["Q", 4, "A"],["A", 73, "A"],["B", 1, "B"],["J", 3, "B"],["B", 5, "B"],
    ["M", 6, "B"],["B", 49, "B"],["H", 13, "B"],["C", 57, "C"],["D", 2, "C"],["C", 4, "C"],["D", 1, "C"]]
   
    
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
    
    totales = float(0)
    correctos = float(0)
    
    for i in range(0, len(DATOS)):
        totales = totales + float(DATOS[i][1])
        if DATOS[i][0] == DATOS[i][2]:
            correctos = correctos + 1 * float(DATOS[i][1])
    resultado = round(correctos / totales, 2)
    print ("Resultado = " + str(resultado))

    # Calculo totales viterbi
    correctos_viterbi = float(0)
    totales_viterbi = float(0)
    secuencia = [int(0)] * N_FRAMES
    posicion = 0
    for i in range(0, len(DATOS)):
        for j in range(0, DATOS[i][1]):
            secuencia[posicion] = valor(DATOS[i][2])
            posicion = posicion + 1
    totales_viterbi = float(N_FRAMES)

    for i in range(0, N_FRAMES):
        if secuencia[i] == path[i]:
            correctos_viterbi = correctos_viterbi + 1
    resultado = round(correctos_viterbi / totales_viterbi, 2)
    print ("Resultado Viterbi= " + str(resultado))

    # cv2.waitKey(0)
"""
