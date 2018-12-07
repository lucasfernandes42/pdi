#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 23:22:11 2018

@author: lucas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def convolucao(img, mascara, coef=1):
    '''
    Filtro de convolução
    '''
    sz = img.shape
    sz_m = mascara.shape
    pad_x = int(sz_m[0]/2)
    pad_y = int(sz_m[1]/2)
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)
    img2 = np.zeros([sz[0]+pad_x, sz[1]+pad_y], dtype=np.uint8)
    img2[pad_x-1:sz[0], pad_y-1:sz[1]] = img
    
    for x in range(pad_x-1, sz[0]):
        for y in range(pad_y-1, sz[1]):
            soma = 0
            for xm in range(sz_m[0]):
                for ym in range(sz_m[1]):
                    i = xm - pad_x
                    j = ym - pad_y
                    soma += img2[x+i, y+j]*mascara[xm, ym]
            img_f[x, y] = max(min(soma*coef, 255), 0)
    return img_f[pad_x-1:sz[0], pad_y-1:sz[1]]

def filtro_media(img):
    mascara = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]])
    return convolucao(img, mascara, 1/9)

def filtro_media_ponderada(img):
    mascara = np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])
    return convolucao(img, mascara, 1/16)

def filtro_laplaciano(img):
    mascara = np.array([[0,-1,0],
                        [-1,4,-1],
                        [0,-1,0]])
    return convolucao(img, mascara) 

def gaussiana(img, delta):
    def h(x, y):
        return math.e**(-((x**2 + y**2))/(2*delta**2))
    mascara = np.array([[h(-1,-1),h(-1,0),h(-1,1)],
                        [h(0,-1), h(0,0), h(0,1)],
                        [h(1,-1), h(1,0), h(1,1)]])
    return convolucao(img, mascara)

def highboost(img, coef):
    #img_borrada = gaussiana(img, 2)
    mascara = filtro_laplaciano(img)
    #img_borrada = filtro_media_ponderada(img)
    #mascara = img_borrada - img
    sz = img.shape    
    img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)

    for i in range(sz[0]):
        for j in range(sz[1]):
            img_f[i,j] = max(0, min(img[i,j] + coef*mascara[i,j], 255))
    return img_f

def sobel(img, orient=None):
    mascara_gx = np.array([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]])
    mascara_gy = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    if orient=='h':
        return convolucao(img, mascara_gx)
    elif orient=='v':
        return convolucao(img, mascara_gy)
    else:
        gx = convolucao(img, mascara_gx)
        gy = convolucao(img, mascara_gy)
        sz = img.shape    
        img_f = np.zeros([sz[0], sz[1]], dtype=np.uint8)

        for i in range(sz[0]):
            for j in range(sz[1]):
                gx[i,j] = gx[i,j]/255
                gy[i,j] = gy[i,j]/255
                img_f[i,j] = max(0, min(255, 255*math.sqrt(gx[i,j]**2 + gy[i,j]**2)))
                img_f[i, j] = max(0,  min(255, 255*gx[i, j]))
        return img_f

def filtro_mediana(img):
    sz = img.shape    
    img_f = img.copy()
    valores = np.zeros(9)

    for x in range(1,sz[0]-1):
        for y in range(1,sz[1]-1):
            i = 0
            for fx in range(3):
                for fy in range(3):
                    valores[i] = img[x + fx - 1][y + fy - 1]
                    i = i + 1
            valores.sort()
            img_f[x, y] = valores[4]
    return img_f

    
### Main
img = cv2.imread("img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()

img2 = cv2.imread("DIP3E_Original_Images_CH03/Fig0338(a)(blurry_moon).tif")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()

img_nois = cv2.imread("img_noise.bmp")
img_nois = cv2.cvtColor(img_nois, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img_nois, cv2.COLOR_GRAY2RGB))
plt.title("Original with noise")
plt.show()


### Convolução
mascara = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])
img_con = convolucao(img2, mascara)
plt.imshow(cv2.cvtColor(img_con, cv2.COLOR_GRAY2RGB))
plt.title("Convolução")
plt.show()

### Filtro média
img_media = filtro_media(img_nois)
plt.imshow(cv2.cvtColor(img_media, cv2.COLOR_GRAY2RGB))
plt.title("Filtro média")
plt.show()

### Média ponderada
img_mp = filtro_media_ponderada(img_nois)
plt.imshow(cv2.cvtColor(img_mp, cv2.COLOR_GRAY2RGB))
plt.title("Filtro média ponderada")
plt.show()


### Laplaciano
img_l = filtro_laplaciano(img2)
plt.imshow(cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB))
plt.title("Filtro laplaciano")
plt.show()

### Sobel
img_s = sobel(img2, orient='h')
plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_GRAY2RGB))
plt.title("Sobel")
plt.show()

### Nitidez High-Boost
img_hb = highboost(img, 4)
plt.imshow(cv2.cvtColor(img_hb, cv2.COLOR_GRAY2RGB))
plt.title("Nitidez High Boost")
plt.show()

### Filtro da mediana
img_mn = filtro_mediana(img_nois)
plt.imshow(cv2.cvtColor(img_mn, cv2.COLOR_GRAY2RGB))
plt.title("Filtro mediana")
plt.show()
