#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:25:40 2018

@author: lucas
"""

import fourier
import cv2
import numpy as np
 
erasing = False
ray = 0

def mouse_drawing(event, x, y, flags, params):
    global erasing, ray
    size=2*ray+1
    if event == cv2.EVENT_MOUSEWHEEL:
        ray=ray+1
        print("size of eraser: {}, press right mouse button or 'm' to reset.".format(ray*2 + 1))
    if event == cv2.EVENT_RBUTTONDOWN:
        ray = 0
        print("size of eraser: 1")
    if event == cv2.EVENT_LBUTTONDOWN:
        erasing = True
        print("erased around point ({}, {}) with size {}, press 'p' to process or 'r' to reset.".format(x,y,size))
        cv2.rectangle(img, (x-ray,y-ray), (x+ray,y+ray),(0,0,0),-1)
        if ray==0: H[y,x]=0 
        else: H[max(0, y-ray):min(H.shape[0], y+ray),max(0, x-ray):min(H.shape[1], x+ray)]=0
    elif event == cv2.EVENT_MOUSEMOVE and erasing:
        print("erased around point ({}, {}) with size {}, press 'p' to process or 'r' to reset.".format(x,y,size))
        cv2.rectangle(img, (x-ray,y-ray), (x+ray,y+ray),(0,0,0),-1)
        if ray==0: H[y,x]=0
        else: H[max(0, y-ray):min(H.shape[0], y+ray),max(0, x-ray):min(H.shape[1], x+ray)]=0
    elif event == cv2.EVENT_LBUTTONUP:
        erasing = False

#img_original = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE) 

img_original = cv2.imread("DIP3E_Original_Images_CH04/Fig0441(a)(characters_test_pattern).tif", cv2.IMREAD_GRAYSCALE) 
img_original = cv2.resize(img_original, (128, 128))
cv2.imshow("Original", img_original)
print("press any key to continue")
cv2.waitKey(0)
print("processing..")

cv2.namedWindow("Frequency")
cv2.setMouseCallback("Frequency", mouse_drawing)

x = np.array(fourier.fft2d(fourier.shift(img_original)))
sz = x.shape
print("done!")
H = np.zeros([sz[0], sz[1]]) + 1      
img = fourier.spectrum(x)
points = []
while True:
 
    cv2.imshow("Frequency", img)
 
    key = cv2.waitKey(1)
    #print(key)
    if key == 27 or key == ord("p"):
        break
    elif key== ord("+"):
        ray = ray + 1
        print("size of eraser: {}, press right mouse button or 'm' to reset.".format(ray*2 + 1))
    elif key== ord("-"):
        ray = max(0, ray - 1)
        print("size of eraser: {}, press right mouse button or 'm' to reset.".format(ray*2 + 1))
    elif key == ord("m"):
        ray=0
        print("size of eraser: 1")
    elif key == ord("r"):
        print("reset!")
        H = np.zeros([sz[0], sz[1]]) + 1   
        img = fourier.spectrum(x)
 
 
cv2.destroyWindow("Frequency")

#####

cv2.imshow("Mask", H)
print("press any key to continue")
cv2.waitKey(0)
cv2.destroyWindow("Mask")

#####
print("processing...")
x = fourier.applyMask(x, H)
        
y = np.array(fourier.ifft2d(x))
img_result = fourier.ishift(fourier.matrix_complex2real(y))
print("done!")
cv2.imshow("Final image", img_result)
print("press any key to exit")
cv2.waitKey(0)
cv2.destroyAllWindows()