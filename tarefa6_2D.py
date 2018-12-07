#!/usr/2*bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov  2 15:20:05 2018

@author: lucas
"""

import numpy as np 
import math
from matplotlib import pyplot as plt
import cv2

# Funções do banco de filtros
h_B = [1/math.sqrt(2), 1/math.sqrt(2)] 

h_A = [-1/math.sqrt(2), 1/math.sqrt(2)]

# Função recursiva que executa a FWT em 2D
def fwt2D(f, scl): 
    
    if scl==0:
        #vmax = max(f.flatten())
        #vmin = abs(min(f.flatten()))
        #f = np.float32([(f[i, :]+vmin)/(vmax+vmin) for i in range(f.shape[0])])
        #f = np.uint8(255*f)
        return f
    # Convoluções A e B em torno das colunas
    f_A = np.float32(np.transpose([np.convolve(f[:, i], h_A, 'same') for i in range(f.shape[1])]))
    for j in range(int(f_A.shape[1]/2)):
        f_A[:, j] = f_A[:, 2*j+1]
    f_A = f_A[:, :int(f_A.shape[1]/2)]
    
    f_B = np.float32(np.transpose([np.convolve(f[:, i], h_B, 'same') for i in range(f.shape[1])]))
    for i in range(int(f_B.shape[1]/2)):
        f_B[:, i] = f_B[:, 2*i+1]
    f_B = f_B[:, :int(f_B.shape[1]/2)]
    
    # Convoluções A e B em torno das linhas, filhos de A
    f_A_A = np.float32([np.convolve(f_A[i, :], h_A, 'same') for i in range(f_A.shape[0])])
    for j in range(int(f_A_A.shape[0]/2)):
        f_A_A[j, :] = f_A_A[2*j+1,:]
    f_A_A = f_A_A[:int(f_A_A.shape[0]/2), :]
    #vmax = max(f_A_A.flatten())
    #vmin = abs(min(0, min(f_A_A.flatten())))
    #f_A_A = np.float32([(f_A_A[i, :]+vmin)/(vmax+vmin) for i in range(f_A_A.shape[0])])
    #f_A_A = np.uint8(255*f_A_A)
    
    f_A_B = np.float32([np.convolve(f_A[i, :], h_B, 'same') for i in range(f_A.shape[0])])
    for j in range(int(f_A_B.shape[0]/2)):
        f_A_B[j, :] = f_A_B[2*j+1, :]
    f_A_B = f_A_B[:int(f_A_B.shape[0]/2), :]
    #vmax = max(f_A_B.flatten())
    #vmin = abs(min(0, min(f_A_B.flatten())))
    #f_A_B = np.float32([(f_A_B[i, :]+vmin)/(vmax+vmin) for i in range(f_A_B.shape[0])])
    #f_A_B = np.uint8(255*f_A_B)
    
    # Convoluções A e B em torno das linhas, filhos de B
    f_B_A = np.float32([np.convolve(f_B[i, :], h_A, 'same') for i in range(f_B.shape[0])])
    for j in range(int(f_B_A.shape[0]/2)):
        f_B_A[j, :] = f_B_A[2*j+1, :]
    f_B_A = f_B_A[:int(f_B_A.shape[0]/2), :]
    #vmax = max(f_B_A.flatten())
    #vmin = abs(min(0, min(f_B_A.flatten())))
    #f_B_A = np.float32([(f_B_A[i, :]+vmin)/(vmax+vmin) for i in range(f_B_A.shape[0])])
    #f_B_A = np.uint8(255*f_B_A)
    
    f_B_B = np.float32([np.convolve(f_B[i, :], h_B, 'same') for i in range(f_B.shape[0])])
    for j in range(int(f_B_B.shape[0]/2)):
        f_B_B[j, :] = f_B_B[2*j+1, :]
    f_B_B = f_B_B[:int(f_B_B.shape[0]/2), :]
    im = fwt2D(f_B_B, scl-1)
    img = np.concatenate((np.concatenate((im, f_B_A), axis=0), 
                                               np.concatenate((f_A_B, f_A_A), 
                                                               axis=0)), axis=1)
    #vmax = max(img.flatten())
    #vmin = abs(min(img.flatten()))
    #img_f = np.float32([(img[i, :]+vmin)/(vmax+vmin) for i in range(img.shape[0])])
    #img_f = np.uint8(255*img_f)
    return (img).round().astype(np.uint8)

'''Função da transformada inversa.'''
def ifwt2D(img, k):    
    if k==0:
        return img
    sz = img.shape
    W_o = ifwt2D(img[0:int(sz[0]/2), 0:int(sz[1]/2)], k-1)
    W_V = img[int(sz[0]/2):sz[0], 0:int(sz[1]/2)]
    W_H = img[0:int(sz[0]/2), int(sz[1]/2):sz[1]]
    W_D = img[int(sz[0]/2):sz[0], int(sz[1]/2):sz[1]]
    sz = W_V.shape
    #print(sz, W_D.shape, W_V.shape,W_H.shape)
    # Convoluções emm W_D e W_V
    f_D = np.zeros([2*sz[0], sz[1]], dtype=np.float32)
    for j in range(2*sz[0]):
        f_D[j,:] = W_D[int(j/2), :]        
    f_D = np.float32([np.convolve(f_D[i,:], h_A, 'same') for i in range(f_D.shape[0])])
    
    
    f_V = np.zeros([2*sz[0], sz[1]], dtype=np.float32)
    for j in range(2*sz[0]):
        f_V[j,:] = W_V[ int(j/2), :]        
    f_V = np.float32([np.convolve(f_V[i,:], h_B, 'same') for i in range(f_V.shape[0])])
    
    # Convoluções em W_H e W_o
    f_H = np.zeros([2*sz[0], sz[1]], dtype=np.float32)
    for j in range(2*sz[0]):
        f_H[j,:] = W_H[ int(j/2), :]        
    f_H = np.float32([np.convolve(f_H[i,:], h_A, 'same') for i in range(f_H.shape[0])])
    
    
    f_o = np.zeros([2*sz[0], sz[1]], dtype=np.float32)
    for j in range(2*sz[0]):
        f_o[j,:] = W_o[ int(j/2), :]        
    f_o = np.float32([np.convolve(f_o[i,:], h_B, 'same') for i in range(f_o.shape[0])])
    
    # Convoluções em W_D+W_V e W_H+W_o
    W_D_V = f_D + f_V
    #vmax = max(255,max(W_D_V.flatten()))
    #vmin = abs(min(0, min(W_D_V.flatten())))
    #W_D_V = np.uint8(255*np.float32([(W_D_V[i, :]+vmin)/(vmax+vmin) for i in range(W_D_V.shape[0])]))
    f_D_V = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float32)
    for j in range(2*sz[1]):
        f_D_V[:, j] = W_D_V[:, int(j/2)]        
    f_D_V = np.float32(np.transpose([np.convolve(f_D_V[:, i], h_A, 'same') for i in range(f_D_V.shape[1])]))
    
    
    W_H_o = f_H + f_o
    #vmax = max(255, max(W_H_o.flatten()))
    #vmin = abs(min(0, min(W_H_o.flatten())))
    #W_H_o = np.uint8(255*np.float32([(vmin+W_H_o[i, :])/(vmax+vmin) for i in range(W_H_o.shape[0])]))
    f_H_o = np.zeros([2*sz[0], 2*sz[1]], dtype=np.float32)
    for j in range(2*sz[1]):
        f_H_o[:, j] = W_H_o[:, int(j/2)]        
    f_H_o = np.float32(np.transpose([np.convolve(f_H_o[:, i], h_B, 'same') for i in range(f_H_o.shape[1])]))
    
    # Reconstruindo imagem
    f = f_D_V + f_H_o
    #vmax = max(255, max(f.flatten()))
    #vmin = abs(min(0, min(f.flatten())))
    #img_f = np.float32([(f[i, :]+vmin)/(vmax+vmin) for i in range(f.shape[0])])
    #img_f = np.uint8(255*img_f)
    return (f).round().astype(np.uint8)
####################

# Imagem para teste
img = cv2.imread("octagon.png", cv2.IMREAD_GRAYSCALE)
#plt.imshow(img, cmap='gray')
#plt.title("Original")
#plt.show() 

# Executar FWT 2D, otimizar e visualizar a árvore. 
im = fwt2D(img, 2)
plt.imshow(im, cmap='gray')
plt.title("Decomposição")
plt.show() 


# Reconstruir imagem de transformada
img_new = ifwt2D(im, 2)
plt.imshow(img_new, cmap='gray')
plt.title("Reconstruída")
plt.show()
print(img_new.shape) 