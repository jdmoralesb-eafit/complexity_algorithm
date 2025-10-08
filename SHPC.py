
"""
Implementación del SHPC de forma serial en python
"""

#Carga de librerias
import cv2
import numpy as np
from numpy import asarray
import scipy as sp
import math as mt
import time
from PIL import Image

#Funcion de reconstrucción
def crear_mascara_circular(shape, centro, radio):
    # Crear una imagen en blanco (negra) del tamaño especificado
    mascara = np.zeros(shape, dtype=np.uint8)
    
    # Dibujar el círculo en la máscara
    cv2.circle(mascara, centro, radio, (255, 255, 255), -1)
    
    return mascara

def amplitud(matriz):
    amplitud = np.abs(matriz)
    return (amplitud)

# Función para el guardado de la imagen
def guardado(name_out, matriz):
    
    resultado = Image.fromarray(matriz)
    resultado = resultado.convert('RGB')
    resultado.save("./"+str(name_out))

def lectura(name_file):
    replica = Image.open(str(name_file)).convert('L')
    return (replica)
def tiro(holo,fx_0,fy_0,fx_tmp, fy_tmp,lamb,M,N,dx,dy,k,m,n):
    
    #Calculo de los angulos de inclinación

    theta_x=mt.asin((fx_0 - fx_tmp) * lamb /(M*dx))
    theta_y=mt.asin((fy_0 - fy_tmp) * lamb /(N*dy))

    #Creación de la fase asociada

    fase= np.exp(1j*k* ((mt.sin(theta_x) * m * dx)+ ((mt.sin(theta_y) * n * dy))))
    fase1=fase
    holo=holo*fase
    
    fase = np.angle(holo, deg=False)
    min_val = np.min(fase)
    max_val = np.max(fase)
    fase = (fase - min_val) / (max_val - min_val)
    threshold_value = 0.2
    fase = np.where(fase > threshold_value, 1, 0)
    value=np.sum(fase)
    return value, fase1

def shpc_reconstruction(
    archivo,
    dx=3.75,
    dy=3.75,
    lamb=0.633,
    G=3,
    radio_mascara=200,
    paso=0.2
):
    """
    Implementación serial del SHPC como función en Python.
    Detecta automáticamente el cuadrante más fuerte en Fourier.
    """
    U = archivo
    N, M = U.shape

    k = 2 * np.pi / lamb
    Fox = M / 2
    Foy = N / 2

    # malla
    x = np.arange(0, M, 1)
    y = np.arange(0, N, 1)
    m, n = np.meshgrid(x - (M / 2), y - (N / 2))

    # --- Fourier ---
    fourier = np.fft.fftshift(sp.fft.fft2(np.fft.fftshift(U)))
    b = amplitud(fourier)

    # --- Paso 1: buscar máximo en la franja superior (30%) ---
    franja_alta = int(N * 0.3)   # 30% superior
    submatriz = b[0:franja_alta, :]
    pos_local = np.unravel_index(np.argmax(submatriz, axis=None), submatriz.shape)
    pos_max = (pos_local[0], pos_local[1])  # coordenadas dentro de la franja
    pos_max = (pos_max[0], pos_max[1])      # (fila, col)

    # --- Paso 2: determinar cuadrante según centro ---
    if pos_max[0] < N/2 and pos_max[1] > M/2:
        cuadrante = 1
    elif pos_max[0] < N/2 and pos_max[1] < M/2:
        cuadrante = 2
    elif pos_max[0] > N/2 and pos_max[1] < M/2:
        cuadrante = 3 
    else:
        cuadrante = 4

    print(f"[INFO] Cuadrante detectado automáticamente: {cuadrante}")

    # --- Paso 3: aplicar máscara de cuadrante ---
    primer_cuadrante = np.zeros((N, M))
    primer_cuadrante[0:round(N / 2 - (N * 0.1)), round(M / 2 + (M * 0.1)):M] = 1

    segundo_cuadrante = np.zeros((N, M))
    segundo_cuadrante[0:round(N / 2 - (N * 0.1)), 0:round(M / 2 - (M * 0.1))] = 1

    tercer_cuadrante = np.zeros((N, M))
    tercer_cuadrante[round(N / 2 + (N * 0.1)):N, 0:round(M / 2 - (M * 0.1))] = 1

    cuarto_cuadrante = np.zeros((N, M))
    cuarto_cuadrante[round(N / 2 + (N * 0.1)):N, round(M / 2 + (M * 0.1)):M] = 1

    if cuadrante == 1:
        fourier = primer_cuadrante * fourier
    if cuadrante == 2:
        fourier = segundo_cuadrante * fourier
    if cuadrante == 3:
        fourier = tercer_cuadrante * fourier
    if cuadrante == 4:
        fourier = cuarto_cuadrante * fourier

    # --- Paso 4: continuar igual ---
    a = amplitud(fourier)
    pos_max = np.unravel_index(np.argmax(a, axis=None), a.shape)
    mascara = crear_mascara_circular(U.shape, (pos_max[1], pos_max[0]), radio_mascara)

    fourier = fourier * mascara
    fourier = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fourier)))

    # búsqueda de frecuencias óptimas
    fx = pos_max[1]
    fy = pos_max[0]
    suma_maxima = 0
    x_max_out, y_max_out = fx, fy

    frec_esp_x = np.arange(fx - paso * G, fx + paso * G, paso)
    frec_esp_y = np.arange(fy - paso * G, fy + paso * G, paso)

    for fy_temp in frec_esp_y:
        for fx_temp in frec_esp_x:
            temp, _ = tiro(fourier, Fox, Foy, fx_temp, fy_temp, lamb, M, N, dx, dy, k, m, n)
            if temp > suma_maxima:
                x_max_out, y_max_out = fx_temp, fy_temp
                suma_maxima = temp

    # corrección de fase
    theta_x = mt.asin((Fox - x_max_out) * lamb / (M * dx))
    theta_y = mt.asin((Foy - y_max_out) * lamb / (N * dy))

    fase = np.exp(1j * k * ((mt.sin(theta_x) * m * dx) + ((mt.sin(theta_y) * n * dy))))
    holo = fourier * fase

    fase = np.angle(holo, deg=False)
    min_val, max_val = np.min(fase), np.max(fase)
    fase_norm = 255 * (fase - min_val) / (max_val - min_val)

    return fase_norm, holo
