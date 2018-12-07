#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:06:18 2018

@author: Lucas de Sousa Fernandes
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def negativo(img):
    '''
    Filtro negativo
    '''
    sz = img.shape
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
    for x in range(sz[0]):
        for y in range(sz[1]):
            img_f[x, y] = 255 - img[x, y]
    return img_f

def logaritmo(img, c):
    '''
    Filtro logaritmo
    '''
    sz = img.shape
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
    for x in range(sz[0]):
        for y in range(sz[1]):
            img_f[x, y] = c*math.log(1 + img[x, y])
    return img_f

def potencia(img, c, gama, epsilon=0):
    '''
    Filtro logaritmo
    '''
    if epsilon==None:
        epsilon = 0
    sz = img.shape
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
    for x in range(sz[0]):
        for y in range(sz[1]):
            img_f[x, y] = min(255*(c*((img[x, y]/255)**gama) + epsilon), 255)
    return img_f

def linear(img):
    '''
    Linear por partes
    '''
    x_pts = [0, 255]
    y_pts = [0, 255]
    
    fig, ax = plt.subplots()
    
    line, = ax.plot(x_pts, y_pts, marker="o")
    
    
    def onpick(event):
        if len(x_pts)>=4:
            return None
        m_x, m_y = event.x, event.y
        x, y = ax.transData.inverted().transform([m_x, m_y])
        x_pts[-1] = x
        y_pts[-1] = y
        x_pts.append(255)
        y_pts.append(255)
        line.set_xdata(x_pts)
        line.set_ydata(y_pts)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', onpick)
    
    plt.title("Entrada (aperte qualquer tecla para terminar)")
    plt.show()
    
    happy = False
    while not happy:
        happy = plt.waitforbuttonpress()
    plt.close()
    
    sz = img.shape
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
    for i in range(sz[0]):
        for j in range(sz[1]):
            x = img[i, j]
            r = 0
            while x>x_pts[r]:
                r += 1
            coef = (y_pts[r] - y_pts[r-1])/(x_pts[r] - x_pts[r-1])
            img_f[i, j] = coef*x + y_pts[r-1]
    return img_f
 
def camadas_bits_plot(img):
    '''
    Camadas de Bits
    '''
    sz = img.shape
    camadas = []
    for i in range(8):
        img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
        for x in range(sz[0]):
            for y in range(sz[1]):
                img_f[x, y] = np.bitwise_and(img[x, y], np.uint64(math.pow(2, i)))
        camadas.append(img_f)
    
    fig, axes = plt.subplots(3, 3, squeeze=False)
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    axes[0,1].imshow(cv2.cvtColor(camadas[0], cv2.COLOR_GRAY2RGB))
    axes[0,2].imshow(cv2.cvtColor(camadas[1], cv2.COLOR_GRAY2RGB))
    axes[1,0].imshow(cv2.cvtColor(camadas[2], cv2.COLOR_GRAY2RGB))
    axes[1,1].imshow(cv2.cvtColor(camadas[3], cv2.COLOR_GRAY2RGB))
    axes[1,2].imshow(cv2.cvtColor(camadas[4], cv2.COLOR_GRAY2RGB))
    axes[2,0].imshow(cv2.cvtColor(camadas[5], cv2.COLOR_GRAY2RGB))
    axes[2,1].imshow(cv2.cvtColor(camadas[6], cv2.COLOR_GRAY2RGB))
    axes[2,2].imshow(cv2.cvtColor(camadas[7], cv2.COLOR_GRAY2RGB))
    
    plt.plot()

def histograma(img):
    '''
    Exibição de histograma
    '''
        
    img_flat = img.flatten()    
    plt.hist(img_flat, bins='auto')
    plt.title("Histograma")
    plt.show()

def equalizar(img):
    '''
    Equalização
    '''
    N = img.shape[0]
    M = img.shape[1]
    n = []
    for i in range(256):
        n.append(0)
    for i in range(N):
        for j in range(M):
            n[int(img[i, j])] += 1
    img_f = np.zeros([N, M], dtype=np.uint8)
    for i in range(N):
        for j in range(M):
            soma_n = 0
            for k in range(int(img[i,j])+1):
                soma_n += n[k]
            img_f[i,j] = max(int((255*soma_n)/N*M), 255)
    return img_f

### Main

img = cv2.imread("img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()

#### Negativo
img_neg = negativo(img)
plt.imshow(cv2.cvtColor(img_neg, cv2.COLOR_GRAY2RGB))
plt.title("Negativo")
plt.show()

#### Logaritmo
img_log = logaritmo(img, 30)
plt.imshow(cv2.cvtColor(img_log, cv2.COLOR_GRAY2RGB))
plt.title("Logaritmo")
plt.show()

#### Potência
img_pot = potencia(img, 1, 3.0)
plt.imshow(cv2.cvtColor(img_pot, cv2.COLOR_GRAY2RGB))
plt.title("Potência")
plt.show()

#### Linear por partes
%matplotlib auto
img_lin = linear(img)
%matplotlib inline
plt.imshow(cv2.cvtColor(img_lin, cv2.COLOR_GRAY2RGB))
plt.title("Linear por partes")
plt.show()

#### Camadas de Bits
%matplotlib auto
camadas_bits_plot(img)

%matplotlib inline

#### Exibição de Histograma
histograma(img)

### Equalização de Histograma
img_eq = equalizar(img)
plt.imshow(cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB))
plt.title("Imagem equalizada")
plt.show()
histograma(img_eq)