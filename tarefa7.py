#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:25:54 2018

@author: lucas
"""

import math
import numpy as np
from matplotlib import pyplot as plt
import cv2

def elemento_estruturante(raio):
    sz = [raio, raio]
    H = np.zeros([2*sz[0]+1, 2*sz[1]+1], dtype=np.uint8)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]+1):
        for j in range(2*sz[1]+1):
            if D(i,j)<=raio:
                H[i,j] = 255
    return H

def erosão(img, es):
    sz = img.shape
    sz_m = es.shape
    pad_x = int(sz_m[0]/2)
    pad_y = int(sz_m[1]/2)
    img_f = np.zeros([sz[0]+2*pad_x, sz[1]+2*pad_y], dtype=np.uint8)
    img2 = np.pad(img, ((pad_x, pad_y),(pad_x, pad_y)), mode="constant")
    
    for x in range(pad_x, sz[0]+pad_x):
        for y in range(pad_y, sz[1]+pad_y):
            values = []
            for xm in range(sz_m[0]):
                for ym in range(sz_m[1]):
                    i = xm - pad_x
                    j = ym - pad_y
                    if es[xm, ym]>0: values.append(es[xm, ym]-img2[x+i, y+j])
            img_f[x, y] = 255 - max(values)
    return img_f[pad_x-1:sz[0]+pad_x-1, pad_y-1:sz[1]+pad_y-1]

def dilatação(img, es):
    sz = img.shape
    sz_m = es.shape
    pad_x = int(sz_m[0]/2)
    pad_y = int(sz_m[1]/2)
    img_f = np.zeros([sz[0]+2*pad_x, sz[1]+2*pad_y], dtype=np.uint8)
    img2 = np.pad(img, ((pad_x, pad_y),(pad_x, pad_y)), mode="constant")
    
    for x in range(pad_x, sz[0]+pad_x):
        for y in range(pad_y, sz[1]+pad_y):
            values = []
            for xm in range(sz_m[0]):
                for ym in range(sz_m[1]):
                    i = xm - pad_x
                    j = ym - pad_y
                    if es[xm, ym]>0: values.append(es[xm, ym]-img2[x+i, y+j])
            img_f[x, y] = 255 - min(values)
    return img_f[pad_x-1:sz[0]+pad_x-1, pad_y-1:sz[1]+pad_y-1]

def gradiente_morf(img, es):
    return dilatação(img,es) - erosão(img,es)

## Imagem original
img = cv2.imread("DIP3E_Original_Images_CH09/Fig0935(a)(ckt_board_section).tif", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.show()

## Definião do elemento estruturante
es = elemento_estruturante(raio=8)
plt.imshow(es, cmap="gray")
plt.title("Elemento estruturante")
plt.show()
## Erosão
img_e = erosão(img, es)
plt.imshow(img_e, cmap="gray")
plt.title("Erosão")
plt.show()

## Dilatação
img_d = dilatação(img, es)
plt.imshow(img_d, cmap="gray")
plt.title("Dilatação")
plt.show()

## Gradiente morfológico
img_g = gradiente_morf(img, es)
plt.imshow(img_g, cmap="gray")
plt.title("Gradiente morfológico")
plt.show()
