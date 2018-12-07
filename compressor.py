#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:58:44 2018

@author: lucas
"""

from matplotlib import pyplot as plt
import cv2
import numpy as np
from heapq import heappush, heappop
from anytree import Node, NodeMixin, RenderTree
import tarefa6_2D as haar

def compress_rle(img):
    img_comp = []
    sz = img.shape
    for i in range(sz[0]):
        linha = []
        j = 0
        while j < sz[1]:
            if j==0 and i==0: # indicar o caractere especial
               for c in range(0, 256):
                   indicador = 256
                   if not c in img.flatten():
                       indicador = c
                       linha.append(indicador)
                       break
            cont = 1
            k = 1
            while(j+k<sz[1] and img[i, j+k]==img[i, j]):
                cont += 1
                k+=1
            if cont<3:
                linha.append(img[i,j])
                j += 1
            else:
                linha.append(indicador)
                linha.append(cont)
                linha.append(img[i,j])
                j += cont
        img_comp.append(linha)
    return img_comp

def decompress_rle(imgc):
    line = imgc[0]
    indicador = int(line[0])
    linef = []
    img = []
    i = 1
    while i < len(line):
        if int(line[i])==indicador:
            cont = int(line[i+1])
            val = int(line[i+2])
            i += 3
            for j in range(cont):
                linef.append(val) 
        else:
            linef.append(int(line[i]))
            i += 1
    img.append(linef)
    first = True            
    for line in imgc:
        if first:
            first = False
            continue
        linef = []
        i = 0
        while i < len(line):
            if int(line[i])==indicador:
                cont = int(line[i+1])
                val = int(line[i+2])
                i += 3
                for j in range(cont):
                    linef.append(val) 
            else:
                linef.append(int(line[i]))
                i += 1
        img.append(linef)
    return np.uint8(img)

def compress_lzw(img):
    dic = {}
    img_f = []
    for i in range(256): dic[i]=str(i)
    k = 256
    for lin in img:
        line = []
        I = ""
        for element in lin:
            cur = I+str(element)
            if cur in dic.keys():
                I = cur
            else:
                if I != '': line.append(I)
                dic[k] = cur
                k = k + 1
                I = str(element)
        line.append(I)
        img_f.append(line)
    return img_f
            
def decompress_lzw(img):
    return np.uint8(compress_lzw(img))

def compress_huffman(img):
    sz = img.shape
    contador = {}
    for i in range(sz[0]):
        for j in range(sz[1]):
            if str(img[i,j]) in contador.keys():    
                contador[str(img[i,j])]+=1
            else:
                contador[str(img[i,j])]=1
    heap = []
    for item in contador.keys():
        heappush(heap, (contador[item], item))
    root = None
    tree = {}
    while len(contador) > 1:
        a = heappop(heap)
        b = heappop(heap)
        heappush(heap, (a[0]+b[0], a[1]+'+'+b[1]))
        contador.pop(a[1])
        contador.pop(b[1])
        contador[a[1]+'+'+b[1]] = a[0]+b[0]
        root = tree[a[1]+'+'+b[1]] = Node(a[1]+'+'+b[1])
        if a[1] in tree.keys(): tree[a[1]].parent=tree[a[1]+'+'+b[1]]    
        else:
            tree[a[1]] = Node(a[1], parent=root)
        if b[1] in tree.keys(): tree[b[1]].parent=tree[a[1]+'+'+b[1]]    
        else:
            tree[b[1]] = Node(b[1], parent=root)
    code = {}
    code[root.name] = ''
    def make_code(node):
        if node.is_leaf:
            return 0
        code[node.children[0].name] = code[node.name] + '0'
        code[node.children[1].name] = code[node.name] + '1'
        make_code(node.children[0])
        make_code(node.children[1])
    make_code(root)
    for node in tree.values():
        if not node.is_leaf:
            code.pop(node.name)
    ####
    for i in range(sz[0]):
        for j in range(sz[1]):
            img[i,j]=code[str(img[i,j])]
    return img, code
    "necessário agora salvar o arquivo codificado e implementar a decodificação"

def save_code(code, filename="code.huf"):
    file = open(filename, "w")
    for key in code.keys():
        file.write(key)
        file.write(code[key])
    file.close()

def load_code(filename="code.huf"):
    file = open(filename, "r")
    lines = []
    for line in file:
        lines.append(line.rstrip())
    code = {}
    i=0
    while(i<len(lines)):
        code[lines[i]] = lines[i+1]
        i+=2
    return code
def decompress_huffman(img, code):
    code_inv = {v: k for k, v in code.items()}
    sz = img.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            img[i,j]=code_inv[str(img[i,j])]
    return img

def compress_haar_lzw(img, filename="compressed.hlf"):
    if(len(img.shape)<3):
        Tree = haar.fwt2D(img, 1)
        imgs = haar.getImgs(Tree)
        file = open(filename, "wb")
        for im in imgs:
            compressed_img = compress_lzw(im)
            for line in compressed_img:
                file.write(bytearray(np.uint8(line)))
    else:
        file = open(filename, "wb")
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        for color in [B,G,R]:
            Tree = haar.fwt2D(color, 1)
            imgs = haar.getImgs(Tree)
            for im in imgs:
                compressed_img = compress_lzw(im)
                for line in compressed_img:
                    file.write(bytearray(np.uint8(line)))
def compress_haar_rle(img, filename="compressed.hrf"):
    if(len(img.shape)<3):
        Tree = haar.fwt2D(img, 1)
        imgs = haar.getImgs(Tree)
        file = open(filename, "wb")
        for im in imgs:
            compressed_img = compress_rle(im)
            for line in compressed_img:
                file.write(bytearray(np.uint8(line)))
    else:
        file = open(filename, "wb")
        B = img[:,:,0]
        G = img[:,:,1]
        R = img[:,:,2]
        for color in [B,G,R]:
            sz = color.shape
            i = 0
            while(i<sz[0]-8):
                j=0
                while(j<sz[1]-8):
                    Tree = haar.fwt2D(color[i:i+8,j:j+8], 1)
                    imgs = haar.getImgs(Tree)
                    for im in imgs:
                        compressed_img = compress_rle(im)
                        for line in compressed_img:
                            #file.write(int.to_bytes(len(line), length=1, byteorder='little'))
                            file.write(bytearray(np.uint8(line)))
                    j=j+8
                i=i+8
    '''falta implementar a transformada inversa de wavelet de haar'''
    '''decodificação:
            1. ler linha a linha, 
            2. formar blocos de 8x8,
            3. aplicar a descompressão RLE nas células 4x4 dos blocos
            4. aplicar a transformada inversa de haar nos blocos
                    '''

def decompress_haar_rle(filename):
    file = open(filename, "rb")
    ### Carregar blocos 8x8 de inteiros na memória
    blocos=[]    
    while(True):
        bloco = []
        for i in range(4):
            line = []
            line_size = int.from_bytes(file.read(1), byteorder='little')
            if(line_size==0):
                break
            for j in range(line_size):
                line.append(int.from_bytes(file.read(1), byteorder='little'))
            bloco.append(line)
        if(line_size==0):
            break
        blocos.append(bloco)
    ### Descomprimir bloco por bloco
    dec_blocos = []
    print(len(blocos))
    for bloco in blocos:
        #print(bloco)
        dec_bloco = decompress_rle(bloco)
        dec_bloco = np.uint8(dec_bloco)
        dec_bloco = haar.ifwt2D([dec_bloco[0:]])
        dec_blocos.append(dec_bloco)
    del(blocos)
    print(len(dec_blocos))
    im = np.concatenate(dec_blocos)
    plt.imshow(im, cmap='gray')
    plt.show()
    
def compress_fourier_rle(img, filename="compressed_fourier.frf"):
    file = open(filename, "wb")
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    for color in [B,G,R]:
        sz = color.shape
        i,j = 0,0
        while(i<sz[0]-8):
            while(j<sz[1]-8):
                img_F = np.fft.fft2(color[i:i+8,j:j+8])
                img_F = np.fft.fftshift(img_F)
                img_F = np.uint8(255*img_F)
                compressed_img = compress_rle(img_F)
                for line in compressed_img:
                    file.write(int.to_bytes(len(line), length=1, byteorder='little'))
                    for element in line:
                        file.write(int.to_bytes(int(element), length=1, byteorder='little'))
                i,j=i+8,j+8
    file.close()

def decompress_fourier_rle(filename="compressed_fourier.frf"):
    file = open(filename, "rb")
    ### Carregar blocos 8x8 de inteiros na memória
    blocos=[]    
    while(True):
        bloco = []
        for i in range(8):
            line = []
            line_size = int.from_bytes(file.read(1), byteorder='little')
            if(line_size==0):
                break
            for j in range(line_size):
                line.append(int.from_bytes(file.read(1), byteorder='little'))
            bloco.append(line)
        if(line_size==0):
            break
        blocos.append(bloco)
    ### Descomprimir bloco por bloco
    dec_blocos = []
    for bloco in blocos:
        #print(bloco)
        dec_bloco = decompress_rle(bloco)
        dec_bloco = np.uint8(255*np.fft.ifft2(np.float32(dec_bloco/255)))
        dec_blocos.append(dec_bloco)
    del(blocos)
    print(len(dec_blocos))
    im = np.concatenate(dec_blocos)
    plt.imshow(im, cmap='gray')
    plt.show()
img = cv2.imread("lena.bmp")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#compress_haar_lzw(img,filename="compressed_color.hlf")
#compress_haar_rle(img,filename="compressed_color.hrf")
#compress_fourier_rle(img)
#decompress_haar_rle("compressed_color.hrf")
'''
imgc = compress_rle(img)

img = decompress_rle(imgc)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()
'''
'''
img_lzw = compress_lzw(img)
print(img_lzw)
img = decompress_lzw(img_lzw)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()
'''
a = [[0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,1,1,1,2,2,2,2],[2,2,3,3,3,3,3,3,4,4,4,4,4]]
a_new, code = compress_huffman(img)
save_code(code)
code