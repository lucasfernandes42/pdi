import numpy as np 
import math
from matplotlib import pyplot as plt
import cv2

def fwt(f, j):
    if j==0:
        return f
    trans = []
    i=0
    while i<len(f)-1:
        trans.append(min(255, max(0, int((float(f[i])+float(f[i+1]))/math.sqrt(2)))))
        i+=2
    i=0
    while i<len(f)-1:
        trans.append(min(255, max(0, int((float(f[i])-float(f[i+1]))/math.sqrt(2)))))
        i+=2
    return np.concatenate((fwt(trans[:int(len(f)/2)], j-1), trans[int(len(f)/2):]), axis=0)

def ifwt(f, j):
    if j==0:
        return f.copy()
    mid = ifwt(f[:int(len(f)/2)], j-1)/2 *j
    side= f[int(len(f)/2):]/2 *j
    out = np.zeros(len(f), dtype=float)
    out[0::2] = mid + side
    out[1::2] = mid - side
    return out.astype(np.uint8)    
def fwt2D(f, scl): 
    
    if scl==0:
        return f
    B = np.uint8(np.transpose([fwt(f[:, i], scl) for i in range(f.shape[1])]))
    A = np.uint8([fwt(B[i, :], scl) for i in range(B.shape[0])])
    return A

def ifwt2D(f, scl):   #com perda 
    if scl==0:
        return f
    A = np.uint8([ifwt(f[i, :], scl) for i in range(f.shape[0])])
    B = np.uint8(np.transpose([ifwt(A[:, i], scl) for i in range(A.shape[1])]))
    return B

print(ifwt(fwt([1,2,3,4], 2), 2))
img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
#plt.imshow(img, cmap='gray')
#plt.title("Original")
#plt.show() 

# Executar FWT 2D, otimizar e visualizar a árvore. 
im = fwt2D(img, 1)
plt.imshow(im, cmap='gray')
plt.title("Decomposição")
plt.show() 

img_new = ifwt2D(im, 2)
plt.imshow(img_new, cmap='gray')
plt.title("Reconstruída")
plt.show()
print(img_new.shape) 