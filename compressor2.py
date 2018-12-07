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

def save_FHG_gray(img, filename="compressed.wfg", codefile="code.hff", level=1):
    sz = img.shape
    img_F = np.fft.fft2(img, s=(2*sz[0], 2*sz[1]))
    img_F = np.fft.fftshift(img_F)
    img_F = np.uint8(255*img_F)
    code = get_huffman_code(img_F)
    save_code(code, codefile)
    img_Comp = compress_huffman(img_F, code)
    bit_string = BitArray("0b"+ "".join(np.array(img_Comp).flat))
    file = open(filename, "wb")
    file.write(bit_string.tobytes())
    file.close

def load_FHG_gray(filename="compressed.wfg", code_filename="code.hff", level=1):
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
    img_rec = np.fft.ifft2(np.float32(data/255))
    sz = [512, 512]
    img_g = np.zeros([2*sz[0], 2*sz[1]], dtype=np.uint8)
    for i in range(2*sz[0]):
        for j in range(2*sz[1]):
            img_g[i,j] = max(0, min(255, (img_rec[i,j].real)*((-1)**(i+j))))
    img_g = img_g[0:sz[0], 0:sz[1]]
    img_rec = np.uint8(255*img_g)
    return img_rec        

def save_FHG(img, file="compressed"):
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    save_FHG_gray(B, file+".b.fhg", file+".b.fhg.huf")
    save_FHG_gray(G, file+".g.fhg", file+".g.fhg.huf")
    save_FHG_gray(R, file+".r.fhg", file+".r.fhg.huf")
    
def load_FHG(file="compressed"):
    img_B = load_FHG_gray(file+".b.fhg", file+".b.fhg.huf")
    img_G = load_FHG_gray(file+".g.fhg", file+".g.fhg.huf")
    img_R = load_FHG_gray(file+".r.fhg", file+".r.fhg.huf")
    sz = img_B.shape
    img = np.zeros([sz[0], sz[1], 3], dtype=np.uint8)
    img[:,:,0] = img_B
    img[:,:,1] = img_G
    img[:,:,2] = img_R
    return img
img = cv2.imread("lena.bmp")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

save_FHG(img, file="Lenna")
img_rec = load_FHG(file="Lenna")
cv2.imshow("Lenna", img_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()

