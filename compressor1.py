#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2
import numpy as np
from heapq import heappush, heappop
from anytree import Node, NodeMixin, RenderTree
import tarefa6_2D as haar
from bitstring import BitArray

def get_huffman_code(img):
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
    return code

def compress_huffman(img, code):
    sz = img.shape
    img_f = []
    for i in range(sz[0]):
        line = []
        for j in range(sz[1]):
            line.append(code[str(img[i,j])])
        img_f.append(line)
    return img_f

def save_code(code, filename="code.huf"):
    file = open(filename, "w")
    for key in code.keys():
        file.write(key+"\n")
        file.write(code[key]+"\n")
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
    file.close()
    return code
def decompress_huffman(img, code):
    code_inv = {v: k for k, v in code.items()}
    img_f = []
    for line in img:
        i = []
        for element in line:
            i.append(code_inv[element])
        img_f.append(i)
    return np.uint8(img_f)
import tarefa3 as filt
def save_WHG_gray(img, filename="compressed.whg", codefile="code.huf", level=1):
    _, haar_img = haar.fwt2D(img, level)
    haar_img = filt.mediana(haar_img)
    code = get_huffman_code(haar_img)
    save_code(code, codefile)
    haar_img_Comp = compress_huffman(haar_img, code)
    bit_string = BitArray("0b"+ "".join(np.array(haar_img_Comp).flat))
    file = open(filename, "wb")
    file.write(bit_string.tobytes())
    file.close

def load_WHG_gray(filename="compressed.whg", code_filename="code.huf", level=1):
    code = load_code(code_filename)
    code_inv = {v: k for k, v in code.items()}
    file = open(filename, "rb")
    bit_string = str(BitArray(file.read()).bin)
    data = []
    i=0
    while(i<len(bit_string)-1):
        buffer = bit_string[i]
        while(i<len(bit_string)-1 and not buffer in code_inv.keys()):
            i+=1
            buffer = "".join([buffer, bit_string[i]])
        if i<len(bit_string)-1:
            data.append(code_inv[buffer])
            i+=1
    file.close()
    while len(data)>512*512:
        data = data[:-1]
    while len(data)<512*512:
        data.append(0)
    data = np.uint8(data).reshape(512,512)
    img_rec = haar.ifwt2D(data, level)
    return img_rec        

def save_WHG(img, file="compressed", level=1):
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    lev = level
    save_WHG_gray(B, file+".b.whg", file+".b.whg.huf", lev)
    save_WHG_gray(G, file+".g.whg", file+".g.whg.huf", lev)
    save_WHG_gray(R, file+".r.whg", file+".r.whg.huf", lev)
    
def load_WHG(file="compressed", level=1):
    lev = level
    img_B = load_WHG_gray(file+".b.whg", file+".b.whg.huf", lev)
    img_G = load_WHG_gray(file+".g.whg", file+".g.whg.huf", lev)
    img_R = load_WHG_gray(file+".r.whg", file+".r.whg.huf", lev)
    sz = img_B.shape
    img = np.zeros([sz[0], sz[1], 3], dtype=np.uint8)
    img[:,:,0] = img_B
    img[:,:,1] = img_G
    img[:,:,2] = img_R
    return img
img = cv2.imread("lena.bmp")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

save_WHG(img, file="lenna/Lenna", level=8)

img_rec = load_WHG(file="lenna/Lenna", level=8)
cv2.imshow("Lenna", img_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
img = cv2.imread("img_noise.bmp")
save_WHG(img, file="Noise")

img_rec = load_WHG(file="Noise")
cv2.imshow("Noise", img_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Level 2
#save_WHG(img, file="Lenna2", level=2)
#img_rec = load_WHG("Lenna2", level=2)
#cv2.imshow("Lenna", img_rec)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
