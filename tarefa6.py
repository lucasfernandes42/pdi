#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov  2 15:20:05 2018

@author: lucas
"""

import cv2
import numpy as np 
import math
from anytree import NodeMixin, RenderTree
from matplotlib import pyplot as plt

# Imagem para teste
img = cv2.imread("lena.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.title("Original")
plt.show() 


# Funções do banco de filtros
h_B = [1/math.sqrt(2), 1/math.sqrt(2)] 

h_A = [-1/math.sqrt(2), 1/math.sqrt(2)]

# Classes dos pacotes wavelet e nós da árvore, contando con a energia de cada pacote

class WaveletPack(object):
    def __init__(self, f): 
        self.f = f
    def E(self):
        energy = 0
        sz = self.f.shape
        for i in range(sz[0]):
            for j in range(sz[1]):
                energy += abs(self.f[i,j])# - 127 ?
        return energy  
    
class WaveletPackNode(WaveletPack, NodeMixin):
    def __init__(self, f, name, parent=None):
        self.f = f
        self.name = name
        self.parent = parent

# Função recursiva que executa a FWT em 2D
def fwt2D(f, scl, W=None): 
    
    if W==None:
        W = WaveletPackNode(f, "W_o "+str(scl), parent=None)
    if scl==0:
        return 0
    # Convoluções A e B em torno das colunas
    f_A = np.float32([np.convolve(f[i, :], h_A) for i in range(f.shape[0])])
    for j in range(int(f_A.shape[1]/2)):
        f_A[:, j] = f_A[:, 2*j+1]
    f_A = f_A[:, :int(f_A.shape[1]/2)]
    
    f_B = np.float32([np.convolve(f[i, :], h_B) for i in range(f.shape[0])])
    for i in range(int(f_B.shape[1]/2)):
        f_B[:, i] = f_B[:, 2*i+1]
    f_B = f_B[:, :int(f_B.shape[1]/2)]
    
    # Convoluções A e B em torno das linhas, filhos de A
    f_A_A = np.float32([np.convolve(f_A[i, :], h_A) for i in range(f_A.shape[0])])
    for j in range(int(f_A_A.shape[0]/2)):
        f_A_A[j, :] = f_A_A[2*j+1,:]
    f_A_A = f_A_A[:int(f_A_A.shape[0]/2), :]
    name = "W_D {}".format(scl-1)
    W_D = WaveletPackNode(f_A_A, name, parent=W)
    fwt2D(f_A_A, scl-1, W=W_D)
    
    f_A_B = np.float32([np.convolve(f_A[i, :], h_B) for i in range(f_A.shape[0])])
    for j in range(int(f_A_B.shape[0]/2)):
        f_A_B[j, :] = f_A_B[2*j+1, :]
    f_A_B = f_A_B[:int(f_A_B.shape[0]/2), :]
    name = "W_V {}".format(scl-1)
    W_V = WaveletPackNode(f_A_B, name, parent=W)
    fwt2D(f_A_B, scl-1, W=W_V)
    
    # Convoluções A e B em torno das linhas, filhos de B
    f_B_A = np.float32([np.convolve(f_B[i, :], h_A) for i in range(f_B.shape[0])])
    for j in range(int(f_B_A.shape[0]/2)):
        f_B_A[j, :] = f_B_A[2*j+1, :]
    f_B_A = f_B_A[:int(f_B_A.shape[0]/2), :]
    name = "W_H {}".format(scl-1)
    W_H = WaveletPackNode(f_B_A, name, parent=W)
    fwt2D(f_B_A, scl-1, W=W_H)
    
    f_B_B = np.float32([np.convolve(f_B[i, :], h_B) for i in range(f_B.shape[0])])
    for j in range(int(f_B_B.shape[0]/2)):
        f_B_B[j, :] = f_B_B[2*j+1, :]
    f_B_B = f_B_B[:int(f_B_B.shape[0]/2), :]
    name = "W_o {}".format(scl-1)
    W_o = WaveletPackNode(f_B_B, name, parent=W)
    fwt2D(f_B_B, scl-1, W=W_o)
    
    return W

# Análise de energia, otimizando a árvore de pacotes
def otimizeFWT(Tree):
    if Tree.is_root:
        for child in Tree.children: otimizeFWT(child)
        return 0
    children_Energy = sum([child.E() for child in Tree.children])
    parent_Energy = Tree.E()
    #print("current: {}, children: {}, parent: {}".format(Tree.name, children_Energy, parent_Energy))
    if children_Energy >= parent_Energy:
        #print("deleted " + Tree.name + " children")
        Tree.children = []
    for child in Tree.children: otimizeFWT(child) 

## Visualizar estrutura da árvore
def printTree(Tree):
    for pre, fill, node in RenderTree(Tree):
        print("{}{} --------- E = {}".format(pre, node.name, node.E()))
    
# Mostrar imagens    
def showImgs(Tree):
    for pre, fill,node in RenderTree(Tree):
        if node.is_leaf:
            vmax = max(node.f.flatten())
            vmin = abs(min(node.f.flatten()))
            img_h = np.float32([(node.f[i, :]+vmin)/(vmax+vmin) for i in range(node.f.shape[0])])
            plt.imshow(cv2.cvtColor(img_h, cv2.COLOR_GRAY2RGB))
            plt.title(node.name)
            plt.show()
'''    
def combineImg(Tree, img):
    if not Tree.is_leaf:
        D = Tree.children[0]
        V = Tree.children[1]
        H = Tree.children[2]
        O = Tree.children[3]
        img = np.concatenate((np.concatenate((combineImg(O,O.f), combineImg(H, H.f)), axis=1),np.concatenate((combineImg(V, V.f), combineImg(D, D.f)), axis=1)), axis=0)
    return img'''
            
# Executar FWT 2D, otimizar e visualizar a árvore. 
Tree = fwt2D(img,3)
otimizeFWT(Tree)
printTree(Tree)

# Mostrar imagens
showImgs(Tree)

'''
im = combineImg(Tree, Tree.f.copy())

vmax = max(im.flatten())
vmin = abs(min(im.flatten()))
img_h = np.float32([(im[i, :]+vmin)/(vmax+vmin) for i in range(im.shape[0])])
plt.imshow(cv2.cvtColor(img_h, cv2.COLOR_GRAY2RGB))
plt.title("test")
plt.show()'''