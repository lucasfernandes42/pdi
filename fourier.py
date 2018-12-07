#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:48:02 2018

@author: lucas
"""

import cmath
import numpy as np
import cv2
from matplotlib import pyplot as plt

#------------------------------------------------
def even(x):
    i = 0
    y = []
    while i<len(x):
        y.append(x[i])
        i+=2
    return y   
 
def odd(x):
    i = 1
    y = []
    while i<len(x):
        y.append(x[i])
        i+=2
    return y

def matrix_complex2real(x):
    y = []
    for row in x:
        row_y = []
        for num in row:
            row_y.append(num.real)
        y.append(row_y)
    return np.array(y)

def array_complex2real(x):
    y = []
    for num in x:
        y.append(num.real)
    return y

def shift(img):
    sz = img.shape
    img_p = np.zeros([2*sz[0], 2*sz[1]]) + 0j
    img_p[0:sz[0], 0:sz[1]] = img
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_p[i,j] = img_p[i,j]*((-1)**(i+j))
    return img_p

def ishift(y):
    sz = [int(a/2) for a in y.shape]
    img_result = np.zeros([2*sz[0], 2*sz[1]], dtype=np.uint8)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_result[i,j] = min(255, max(0, y[i,j]*((-1)**(i+j))))
    img_result = np.uint8(img_result[0:sz[0], 0:sz[1]])
    return img_result

def applyMask(x, H):
    sz = x.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            x[i,j] = x[i,j]*H[i, j]
    return x
    
def spectrum(x):
    x1 = x.copy()
    sz = x1.shape
    vmax = np.log(max(x1.flat))
    for i in range(sz[0]):
        for j in range(sz[1]):
            x1[i,j] = min(max(0, 255*np.log(abs(x1[i,j]))/vmax), 255)
    return np.uint8(matrix_complex2real(x1))
#---------------------------------------------------
def fft_radix(x, N, inverse=False):
    if N>1:
        x_even = even(x)
        x_odd = odd(x)
        x[:N//2] = fft_radix(x_even, N//2, inverse)             #DFT of (x0, x2s, x4s, ...)
        x[N//2:] = fft_radix(x_odd, N//2, inverse)           #DFT of (xs, xs+2s, xs+4s, ...)
        coef = (2j if inverse else -2j)*cmath.pi/float(N)                           
        exps = [cmath.exp(coef*k) for k in range(N//2)]
        for k in range(N//2):                           #combine DFTs of two halves into full DFT:
            t = x[k]
            r = x[k+N//2]
            x[k] = t + exps[k]*r
            x[k+N//2] = t - exps[k]*r
    return x

def fft(x):
    x = [x[i] + 0j for i in range(len(x))]
    return fft_radix(x, len(x))

def fft2d(x):
    x = [fft(row) for row in x]
    x = list(map(list, zip(*x))) #transpose
    x = [fft(row) for row in x]
    x = list(map(list, zip(*x))) #transpose
    return x

#-----------------------------------------------------
def ifft(x):
    return np.array(fft_radix(x, len(x), inverse=True))/len(x)

def ifft2d(x):
    x = [ifft(row) for row in x]
    x = list(map(list, zip(*x))) #transpose
    x = [ifft(row) for row in x]
    x = list(map(list, zip(*x))) #transpose
    return x

#-----------------------------------------------------