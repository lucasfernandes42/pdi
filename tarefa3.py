
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 19:26:58 2018

@author: lucas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def process(img, H):
    sz = img.shape
    '''img_p = np.zeros([2*sz[0], 2*sz[1]], dtype=np.uint8)
    img_p[0:sz[0], 0:sz[1]] = img
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_p[i,j] = img_p[i,j]*((-1)**(i+j))
    img_F = np.fft.fft2(img_p) '''
    img_F = np.fft.fft2(img, s=(2*sz[0], 2*sz[1])) #Compute the  FFT of a real array
    img_F = np.fft.fftshift(img_F)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_F[i,j] = img_F[i,j]*H[i, j]
    img_result_g = np.fft.ifft2(img_F)
    img_result = np.zeros([2*sz[0], 2*sz[1]], dtype=np.uint8)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_result[i,j] = max(0, min(255, (img_result_g[i,j].real)*((-1)**(i+j))))
    return img_result[0:sz[0], 0:sz[1]]

def ILPF(img, f):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            if D(i,j)<=f:
                H[i,j] = 1
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)

def butterworth_lowpass(img, f, n):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            H[i,j] = 1/(1+(D(i,j)/f)**(2*n))
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)

def gaussian_lowpass(img, f):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            H[i,j] = math.exp(-(D(i,j)*D(i,j))/(2*f*f))
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)

def IHPF(img, f):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            if D(i,j)>f:
                H[i,j] = 1
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)

def butterworth_highpass(img, f, n):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            H[i,j] = 1 - (1/(1+(D(i,j)/f)**(2*n)))
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)

def gaussian_highpass(img, f):
    sz = img.shape
    H = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float128)
    def D(u, v):
        return math.sqrt((u-sz[0])**2+ (v - sz[1])**2)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            H[i,j] = 1 - math.exp(-(D(i,j)*D(i,j))/(2*f*f))
    #plt.imshow(cv2.cvtColor(np.uint8(255*H),cv2.COLOR_GRAY2RGB))
    #plt.show()
    return process(img, H)
'''
# Lowpass filters
img = cv2.imread("DIP3E_Original_Images_CH04/Fig0441(a)(characters_test_pattern).tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()

img_ideal = ILPF(img, 60)
plt.imshow(cv2.cvtColor(img_ideal, cv2.COLOR_GRAY2RGB))
plt.title("Ideal lowpass filter")
plt.show()

img_bw = butterworth_lowpass(img, 60, 2)
plt.imshow(cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB))
plt.title("Butterworth lowpass filter")
plt.show()

img_g = gaussian_lowpass(img, 60)
plt.imshow(cv2.cvtColor(img_g, cv2.COLOR_GRAY2RGB))
plt.title("Gaussian lowpass filter")
plt.show()

# Highpass filters
img_ideal = IHPF(img, 30)
plt.imshow(cv2.cvtColor(img_ideal, cv2.COLOR_GRAY2RGB))
plt.title("Ideal highpass filter")
plt.show()

img_bw = butterworth_highpass(img, 30, 2)
plt.imshow(cv2.cvtColor(img_bw, cv2.COLOR_GRAY2RGB))
plt.title("Butterworth highpass filter")
plt.show()

img_g = gaussian_highpass(img, 30)
plt.imshow(cv2.cvtColor(img_g, cv2.COLOR_GRAY2RGB))
plt.title("Gaussian highpass filter")
plt.show()
'''
###########################################
########## FILTROS ESPACIAIS ##############
###########################################

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

def media_geometrica(img):
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
            prod = 1
            for i in range(len(valores)):
                prod = prod*valores[i]
            img_f[x, y] = prod**(1/9)
    return img_f

def media_harmonica(img):
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
            sum = 0
            for i in range(len(valores)):
                sum += 1/valores[i]
            img_f[x, y] = 9/sum
    return img_f

def media_contra_harmonica(img, Q):
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
            sum_1 = 0
            for i in range(len(valores)):
                sum_1 += valores[i]**(Q+1)
            sum_2 = 0
            for i in range(len(valores)):
                sum_2 += valores[i]**Q
            img_f[x, y] = int(sum_1/sum_2)
    return img_f

def mediana(img):
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

def max_filter(img):
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
            img_f[x, y] = max(valores)
    return img_f

def min_filter(img):
    sz = img.shape    
    img_f = img.copy()
    valores = []

    for x in range(1,sz[0]-1):
        for y in range(1,sz[1]-1):
            for fx in range(3):
                for fy in range(3):
                    valores.append(img[x + fx - 1, y + fy - 1])
            img_f[x, y] = min(valores)
            valores = []
    return img_f

def ponto_medio(img):
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
            img_f[x, y] = (min(valores)+max(valores))/2
    return img_f
'''
# Filtros espaciais
img = cv2.imread("img_noise.jpg")
img = cv2.resize(img, (300,300))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()

# Média geométrica
img_mg = media_geometrica(img)
plt.imshow(cv2.cvtColor(img_mg, cv2.COLOR_GRAY2RGB))
plt.title("Filtro da média geométrica")
plt.show()

# Média harmônica
img_mh = media_harmonica(img)
plt.imshow(cv2.cvtColor(img_mh, cv2.COLOR_GRAY2RGB))
plt.title("Filtro da média harmônica")
plt.show()

# Média contra harmônica
img_mch = media_contra_harmonica(img, 2)
plt.imshow(cv2.cvtColor(img_mch, cv2.COLOR_GRAY2RGB))
plt.title("Filtro da média contra harmônica")
plt.show()

# Mediana
img_mn = mediana(img)
plt.imshow(cv2.cvtColor(img_mn, cv2.COLOR_GRAY2RGB))
plt.title("Filtro mediana")
plt.show()

# Ponto máximo
img_m = cv2.imread("max.png")
img_m = cv2.resize(img_m, (300,300))
img_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img_m, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()
img_max = max_filter(img_m)
plt.imshow(cv2.cvtColor(img_max, cv2.COLOR_GRAY2RGB))
plt.title("Filtro do ponto máximo")
plt.show()

# Ponto mínimo
img_m = cv2.imread("min.png")
img_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2GRAY)
img_m = cv2.resize(img_m, (300,300))
plt.imshow(cv2.cvtColor(img_m, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show()
img_min = min_filter(img_m)
plt.imshow(cv2.cvtColor(img_min, cv2.COLOR_GRAY2RGB))
plt.title("Filtro do ponto mínimo")
plt.show()

# Ponto médio
img_pm = ponto_medio(img)
plt.imshow(cv2.cvtColor(img_pm, cv2.COLOR_GRAY2RGB))
plt.title("Filtro do ponto médio")
plt.show()
'''

