#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 22:24:00 2018

@author: lucas
"""

import math
import numpy as np

h_B = [1/math.sqrt(2), 1/math.sqrt(2)] 

h_A = [-1/math.sqrt(2), 1/math.sqrt(2)]

from anytree import Node, RenderTree

cont = {}
def fwt(f, j):
    W = Node("W")
    cont[W.name] = f
    f_A = np.convolve(h_A, f)
    for i in range(int(len(f_A)/2)):
        f_A[i] = f_A[2*i+1]
    f_A = f_A[:int(len(f_A)/2)]
    f_B = np.convolve(h_B, f)
    for i in range(int(len(f_B)/2)):
        f_B[i] = f_B[2*i+1]
    f_B = f_B[:int(len(f_B)/2)]
    W_A(f_A, j-1, W)
    W_B(f_B, j-1, W)
    return W

def W_B(f, j, W_p):
    W = Node("W_B " + str(j), parent=W_p)
    if j==0:
        cont[W.name] = f
        return 0
    cont[W.name] = f
    f_A = np.convolve(h_A, f)
    for i in range(int(len(f_A)/2)):
        f_A[i] = f_A[2*i+1]
    f_A = f_A[:int(len(f_A)/2)]
    f_B = np.convolve(h_B, f)
    for i in range(int(len(f_B)/2)):
        f_B[i] = f_B[2*i+1]
    f_B = f_B[:int(len(f_B)/2)]
    W_A(f_A, j-1, W)
    W_B(f_B, j-1, W)

def W_A(f, j, W_p):
    W = Node("W_A " + str(j), parent=W_p)
    cont[W.name] = f
    if j==0:
        return 0
    
    
f = [1, 4, -3, 0]

W = fwt(f, 2)

for pre, fill, node in RenderTree(W):
     print("{}{}: {}".format(pre, node.name, cont[node.name]))
     