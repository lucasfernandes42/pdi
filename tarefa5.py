#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:27:36 2018

@author: lucas
"""

from matplotlib import pyplot as plt
import cv2
import numpy as np


def threshold(img, r, g, b, binary=True):
    sz = img.shape
    if binary:
        img_f = np.zeros([sz[0], sz[1], 3], dtype=np.uint8)
    else:
        img_f = img.copy()

    for x in range(0,sz[0]):
        for y in range(0,sz[1]):
            if img[x,y][0] > b: img_f[x, y][0] = 255
            if img[x,y][1] > g: img_f[x, y][1] = 255
            if img[x,y][2] > r: img_f[x, y][2] = 255
    return img_f

def thresholdHSV(img, H, S, V, binary=True):
    H = H/180.0
    S = S/255.0
    V = V/255.0
    if H == 1.0: H = 0.0
    if S == 0.0: R, G, B = V, V, V
    else:
        i = int(H*6.) 
        f = (H*6.)-i 
        p,q,t = V*(1.-S), V*(1.-S*f), V*(1.-S*(1.-f))
        i%=6
        if i == 0: R, G, B = V, t, p
        if i == 1: R, G, B = q, V, p
        if i == 2: R, G, B = p, V, t
        if i == 3: R, G, B = p, q, V
        if i == 4: R, G, B = t, p, V
        if i == 5: R, G, B = V, p, q   
        R = int(max(0, min(255, R*255)))
        G = int(max(0, min(255, G*255)))
        B = int(max(0, min(255, B*255)))
    print("RGB = ({},{},{})".format(R, G, B))
    return threshold(img, R, G, B, binary)

def brilho(img, brilho):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Imagem em HSV
    sz = img.shape    
    img_f = img.copy()

    for x in range(1,sz[0]-1):
        for y in range(1,sz[1]-1):
            img_f[x, y][2] = min(255, int(img[x, y][2]*brilho))        
    return cv2.cvtColor(img_f, cv2.COLOR_HSV2RGB)

def sub(img1, img2):
    sz = img1.shape
    img_f = np.zeros([sz[0], sz[1], 3], dtype=np.uint8)
    for x in range(0,sz[0]):
        for y in range(0,sz[1]):
            img_f[x, y][0] = max(0, int(img1[x, y][0]) - int(img2[x, y][0]))    
            img_f[x, y][1] = max(0, int(img1[x, y][1]) - int(img2[x, y][1]))
            ### RuntimeWarning: overflow encountered in ubyte_scalars
            ### quando se retira o min()
            img_f[x, y][2] = max(0, int(img1[x, y][2]) - int(img2[x, y][2]))
            
    return img_f

def chromakey(img, value=235, bk=None):
    img_f = threshold(img, r=255, g=value, b=255)
    img_f = sub(img, img_f)
    if bk==None:
        return img_f
    sz = bk.shape
    img_f = cv2.resize(img_f, (sz[1], sz[0]))
    for i in range(0,sz[0]):
        for j in range(0,sz[1]):
            if img_f[i, j][1]==0:
                img_f[i, j]= bk[i,j]
    return img_f

def sepia(img):
    sz = img.shape
    img_f = np.zeros([sz[0], sz[1], 3], dtype=np.uint8)
    for x in range(sz[0]):
        for y in range(sz[1]):
            img_f[x, y][2] = int(min(255, 0.393*img[x, y][2] + 0.769*img[x, y][1] + 0.189*img[x, y][0]))
            img_f[x, y][1] = int(min(255, 0.349*img[x, y][2] + 0.686*img[x, y][1] + 0.168*img[x, y][0]))
            img_f[x, y][0] = int(min(255, 0.272*img[x, y][2] + 0.543*img[x, y][1] + 0.131*img[x, y][0]))
    return img_f


### Imagens usadas
img_ck = cv2.imread("chromakey.jpg")
plt.imshow(cv2.cvtColor(img_ck, cv2.COLOR_BGR2RGB))
plt.title("Original Chroma Key")
plt.show()

img_1 = cv2.imread("chihiro1.png")
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.title("Frame 1")
plt.show()

img_2 = cv2.imread("chihiro2.png")
plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
plt.title("Frame 2")
plt.show()

## Threshold
img_t = threshold(img_ck, r=255, g=235, b=255)
plt.imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
plt.title("Threshold RGB")
plt.show() 

img_t = thresholdHSV(img_ck, H=150, S=20, V=255)
plt.imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
plt.title("Threshold HSV")
plt.show()

## Subtração
img_s = sub(img_ck, img_t)
plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
plt.title("Subtração")
plt.show()

## Chroma Key
img_ck2 = chromakey(img_ck, value=220, bk=img_1)
plt.imshow(cv2.cvtColor(img_ck2, cv2.COLOR_BGR2RGB))
plt.title("Chroma key")
plt.show()

## Subtração (frame)
img_s = sub(img_2, img_1)
plt.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
plt.title("Subtração")
plt.show()

## Brilho
img_b = brilho(img_1, 1.8)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.title("Brilho")
plt.show()

## Sepia
img_sepia = sepia(img_1)
plt.imshow(cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB))
plt.title("Sépia")
plt.show()