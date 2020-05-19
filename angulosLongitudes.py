#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:32:37 2019

@author: Jaime Duque Domingo (UVA)

Esta clase se encarga de implementar los métodos de extracción de los ángulos de los vectores que devuelve 
OpenPose así como de obtener las longitudes de los vectores necesarios. Dicha lista de valores calculados
representa la entrada de la red neuronal de detección de posturas.

"""
import numpy as np
import math

class angulosLongitudes:

  def __init__(self):
    # Número de ángulos que calculamos (incluimos un indicador booleano para decir si existe o no el angulo)
    self.keyAngles = 4 * 2
    # Número de huesos que calculamos (incluimos un indicador booleano para decir si existe o no el hueso)
    self.keyHuesos = 6 * 2

  ###################################################################################
  # Apartado: Cálculo de ángulos
  ###################################################################################
  # En este modelo utilizamos los ángulos del cuerpo en vez de las coordenadas de los puntos de OpenPose
  def calcularPendiente(self, x1, y1, x2, y2):
    pdt = 0
    if (x2 - x1) != 0:
        pdt = float( (y2 - y1) / (x2 - x1) ) 
    return pdt
    
  def calcularAnguloPendiente(self, m1, m2):
    aP = 9999999
    if m1 != 0 and m2 != 0:
        aP = math.atan2( (m2 - m1), (1 + m1 * m2) )
    return aP
    
  def normalizarAngulo(self, ang):
    # Si no es posible calcular el ángulo devolvemos 0
    if ang == 9999999:
        return 0
    # Normalizamos el ángulo de 0 a 1
    # El primer paso es añadir 2*pi para que no haya ángulos negativos y los ángulos
    # - pi se normalicen a pi. Iremos de pi a 3*pi asegurándonos que todos tienen valor
    # cuando han podido ser calculados
    a = 2 * math.pi + ang
    # Regla de 3: 3pi es a 1 como a es a x
    return float( a / (3 * math.pi) )
    
  # Calcular el ángulo de dos rectas r y s a partir de sus coordenadas
  def calcularAngulo(self, r_x1, r_y1, r_x2, r_y2, s_x1, s_y1, s_x2, s_y2):
    m1 = self.calcularPendiente(r_x1, r_y1, r_x2, r_y2)
    m2 = self.calcularPendiente(s_x1, s_y1, s_x2, s_y2)
    a1 = self.calcularAnguloPendiente(m1, m2)
    a2 = self.normalizarAngulo(a1)
    return a2
   
  def anguloRectas(self,poseKeyPoints, persona, r1, r2, s1, s2):
    return self.calcularAngulo(  poseKeyPoints[persona][r1][0],
                                 poseKeyPoints[persona][r1][1],
                                 poseKeyPoints[persona][r2][0],
                                 poseKeyPoints[persona][r2][1],
                                 poseKeyPoints[persona][s1][0],
                                 poseKeyPoints[persona][s1][1],
                                 poseKeyPoints[persona][s2][0],
                                 poseKeyPoints[persona][s2][1] )
    
  # Calcular el ángulo de dos vectores a partir de sus coordenadas
  # (p_x2, p_y2 ) es el vertice del angulo
  def calcularAnguloVector(self, p_x1, p_y1, p_x2, p_y2, p_x3, p_y3):

    if p_x1 == 0 or p_y1 == 0 or p_x2 == 0 or p_y2 == 0 or p_x3 == 0 or p_y3 == 0:
        return 99999
      
    v1_x = p_x1 - p_x2
    v1_y = p_y1 - p_y2
    v2_x = p_x3 - p_x2
    v2_y = p_y3 - p_y2    
    
    ang = math.atan2(v1_x * v2_y - v1_y * v2_x, v1_x * v2_x + v1_y * v2_y)    
    
    return float((math.pi + ang) / (2 * math.pi))
    
    """mod_v1 = math.sqrt(v1_x**2 + v1_y**2)
    mod_v2 = math.sqrt(v2_x**2 + v2_y**2)
    esc_v1_v2 = v1_x * v2_x + v1_y * v2_y
    
    p_mod = mod_v1 * mod_v2
    ang = 99999
    if p_mod != 0:
        # ang = math.acos(esc_v1_v2/p_mod)
        v = esc_v1_v2/p_mod
        if v > 1:
            v = 1
        if v < -1:
            v = -1
        ang = math.acos(v)
        
        # Calculamos el seno del angulo para ver el signo (Lagrange)
        v_s = mod_v1**2 * mod_v2**2 - esc_v1_v2**2
        if v_s < 0:
            v_s = 0
        vect_v1_v2 = math.sqrt(v_s)
        sinA = vect_v1_v2 / p_mod
        if sinA < 0:
            ang = 2 * math.pi - ang
    ang = ang - math.pi
    # en este punto ang se encuentra entre -pi y pi
    # dividimos entre pi para devolver datos entre -1 y 1
    return float( ang / math.pi )"""
    
  def anguloVectores(self, poseKeyPoints, persona, p1, p2, p3):
    return self.calcularAnguloVector(poseKeyPoints[persona][p1][0],
                                     poseKeyPoints[persona][p1][1],
                                     poseKeyPoints[persona][p2][0],
                                     poseKeyPoints[persona][p2][1],
                                     poseKeyPoints[persona][p3][0],
                                     poseKeyPoints[persona][p3][1] )
    
  def obtenerAngulosInteres(self, poseKeyPoints, persona):
    angulo = np.zeros(self.keyAngles)

    a = self.anguloVectores(poseKeyPoints, persona, 1, 2, 3)
    if a == 99999:
        angulo[0] = 0
        angulo[1] = 0
    else:
        angulo[0] = 1
        angulo[1] = a
        
    a = self.anguloVectores(poseKeyPoints, persona, 1, 5, 6)
    if a == 99999:
        angulo[2] = 0
        angulo[3] = 0
    else:
        angulo[2] = 1
        angulo[3] = a

    a = self.anguloVectores(poseKeyPoints, persona, 2, 3, 4)
    if a == 99999:
        angulo[4] = 0
        angulo[5] = 0
    else:
        angulo[4] = 1
        angulo[5] = a
        
    a = self.anguloVectores(poseKeyPoints, persona, 5, 6, 7)
    if a == 99999:
        angulo[6] = 0
        angulo[7] = 0
    else:
        angulo[6] = 1
        angulo[7] = a
    
    """angulo[0] =  self.anguloVectores(poseKeyPoints, persona, 0, 1, 2, math.pi / 2)
    angulo[1] =  self.anguloVectores(poseKeyPoints, persona, 0, 1, 5, math.pi / 2)    
    angulo[2] =  self.anguloVectores(poseKeyPoints, persona, 2, 1, 5, math.pi)    
    angulo[3] =  self.anguloVectores(poseKeyPoints, persona, 1, 2, 3, math.pi / 2)
    angulo[4] =  self.anguloVectores(poseKeyPoints, persona, 1, 5, 6, math.pi / 2)
    angulo[5] =  self.anguloVectores(poseKeyPoints, persona, 2, 3, 4, math.pi)
    angulo[6] =  self.anguloVectores(poseKeyPoints, persona, 5, 6, 7, math.pi)
    angulo[7] =  self.anguloVectores(poseKeyPoints, persona, 0, 1, 8, math.pi)
    angulo[8] =  self.anguloVectores(poseKeyPoints, persona, 1, 8, 9, math.pi / 2)
    angulo[9] =  self.anguloVectores(poseKeyPoints, persona, 1, 8, 12, math.pi / 2)
    angulo[10] = self.anguloVectores(poseKeyPoints, persona, 8, 9, 10, math.pi / 2)
    angulo[11] = self.anguloVectores(poseKeyPoints, persona, 8, 12, 13, math.pi / 2)
    angulo[12] = self.anguloVectores(poseKeyPoints, persona, 9, 10, 11, math.pi)
    angulo[13] = self.anguloVectores(poseKeyPoints, persona, 12, 13, 14, math.pi)"""
    return angulo

  ###################################################################################
  # Apartado: Cálculo de longitudes
  ###################################################################################

  def calcularLongitudHueso(self, x1, y1, x2, y2):
    return math.sqrt( (float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)    

  def longitudHueso(self, poseKeyPoints, persona, p1, p2):
    if (poseKeyPoints[persona][p1][0] == 0 or
        poseKeyPoints[persona][p1][1] == 0 or    
        poseKeyPoints[persona][p2][0] == 0 or        
        poseKeyPoints[persona][p2][1] == 0):
            return 0    
    return self.calcularLongitudHueso( poseKeyPoints[persona][p1][0],
                                       poseKeyPoints[persona][p1][1],
                                       poseKeyPoints[persona][p2][0],
                                       poseKeyPoints[persona][p2][1])

  def obtenerLongitudHuesosInteres(self, poseKeyPoints, persona):
    hueso = np.zeros(self.keyHuesos)
    
    norm = self.longitudHueso(poseKeyPoints, persona, 1, 8)
        
    if norm == 0:
        return []        
    
    hueso[1] =  self.longitudHueso(poseKeyPoints, persona, 1, 2) / norm
    if hueso[1] == 0:
        hueso[0] = 0
    else:
        hueso[0] = 1
    
    hueso[3] =  self.longitudHueso(poseKeyPoints, persona, 1, 5) / norm
    if hueso[3] == 0:
        hueso[2] = 0
    else:
        hueso[2] = 1
    
    hueso[5] =  self.longitudHueso(poseKeyPoints, persona, 2, 3) / norm
    if hueso[5] == 0:
        hueso[4] = 0
    else:
        hueso[4] = 1
    
    hueso[7] =  self.longitudHueso(poseKeyPoints, persona, 3, 4) / norm
    if hueso[7] == 0:
        hueso[6] = 0
    else:
        hueso[6] = 1
    
    hueso[9] =  self.longitudHueso(poseKeyPoints, persona, 5, 6) / norm
    if hueso[9] == 0:
        hueso[8] = 0
    else:
        hueso[8] = 1

    hueso[11] =  self.longitudHueso(poseKeyPoints, persona, 6, 7) / norm    
    if hueso[11] == 0:
        hueso[10] = 0
    else:
        hueso[10] = 1    
        
    """hueso[0] =  self.longitudHueso(poseKeyPoints, persona, 0, 1)
    hueso[1] =  self.longitudHueso(poseKeyPoints, persona, 1, 8)
    hueso[2] =  self.longitudHueso(poseKeyPoints, persona, 1, 2)
    hueso[3] =  self.longitudHueso(poseKeyPoints, persona, 1, 5)
    hueso[4] =  self.longitudHueso(poseKeyPoints, persona, 2, 3)
    hueso[5] =  self.longitudHueso(poseKeyPoints, persona, 3, 4)
    hueso[6] =  self.longitudHueso(poseKeyPoints, persona, 5, 6)
    hueso[7] =  self.longitudHueso(poseKeyPoints, persona, 6, 7)
    hueso[8] =  self.longitudHueso(poseKeyPoints, persona, 8, 9)
    hueso[9] =  self.longitudHueso(poseKeyPoints, persona, 8, 12)
    hueso[10] = self.longitudHueso(poseKeyPoints, persona, 9, 10)
    hueso[11] = self.longitudHueso(poseKeyPoints, persona, 10, 11)
    hueso[12] = self.longitudHueso(poseKeyPoints, persona, 12, 13)
    hueso[13] = self.longitudHueso(poseKeyPoints, persona, 13, 14)"""



    # este hueso deben existir para normalizar
    """if hueso[0] == 0 or hueso[1] == 0:
        return []
        
    # Valores por defecto de los huesos. Se han calculado a partir de un esqueleto en
    # posición de reposo.
    vd = [0.26, 0.74, 0.17, 0.18, 0.40, 0.18, 0.30, 0.28, 0.15, 0.12, 0.60, 0.19, 0.53, 0.32]
    norm = hueso[0] + hueso[1]
    for i in range(0, self.keyHuesos):
        hueso[i] = hueso[i] / norm
        if hueso[i] == 0:
            hueso[i] = vd[i] * norm / (vd[0] + vd[1])"""
    
    """HUESOS : [0.26264843 0.73735157 0.17215858 0.18744442 0.394499   0.18224717
    0.30276555 0.28197195 0.14656761 0.1192859  0.60779611 0.19460296
    0.5348607  0.31696131]
    
    HUESOS : [0.2647081  0.7352919  0.1336484  0.16146355 0.39979975 0.19015053
    0.26775835 0.2185452  0.11540793 0.1208523  0.51959449 0.
    0.47246433 0.30441959]"""
    
    # print("HUESOS : " + str(hueso))
    return hueso

  ###################################################################################
  # Apartado: Método principal y concatenación de datos
  # Si no puede devolver un array correcto, devuelve []
  ###################################################################################
  def obtenerAngulosLongitudes(self, poseKeypoints, persona):
    # Aplicamos una comprobación inicial para ver si es factible el cálculo.

    # Debe existir la coordenada del cuello y del cuerpo
    resultado = []
    if (poseKeypoints[persona][0][0] == 0 or poseKeypoints[persona][0][1] == 0 or
        poseKeypoints[persona][1][0] == 0 or poseKeypoints[persona][1][1] == 0): 
	return resultado

    # Obtenemos las longitudes de los huesos
    longitudHuesosInteres = self.obtenerLongitudHuesosInteres(poseKeypoints, persona)
    if longitudHuesosInteres == []:
	return resultado

    # Obtenemos los ángulos de los vectores devueltos por OpenPose
    angulosInteres = self.obtenerAngulosInteres(poseKeypoints, persona)

    resultado = np.concatenate((angulosInteres, longitudHuesosInteres), axis=0)
    return resultado
  ###################################################################################
  ###################################################################################

