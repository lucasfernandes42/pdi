#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:15:50 2018

@author: lucas
"""

from matplotlib import pyplot as plt
import cv2

def RGBtoHSV(img):
    img = img.copy()
    sz = img.shape
    for i in range(sz[0]):
        for j in range(sz[1]):         
            pixel = img[i,j]
            R = pixel[0]/255
            G = pixel[1]/255
            B = pixel[2]/255
            hue = None
            M = max(R, G, B)
            m = min(R, G, B)
            if M==m:
                return 0 
            if M==R and G>=B:
                hue = 60*(G-B)/(M-m)
            if M==R and G<B:
                hue = 60*(G-B)/(M-m) + 360
            if M==G:
                hue = 60*(B-R)/(M-m) + 120
            if M==B:
                hue = 60*(R-G)/(M-m) + 240
            if M != 0:
                sat = (M-m)/M
            else:
                sat = 0
            val = M
            pixel[0] = hue/2
            pixel[1] = sat*255
            pixel[2] = val*255
    return img

def HSVtoRGB(img):
    img2 = img.copy()
    sz = img2.shape
    for x in range(sz[0]):
        for y in range(sz[1]):         
            pixel = img2[x,y]
            H = pixel[0]/180.
            S = pixel[1]/255.
            V = pixel[2]/255.
            if H == 1.0: H = 0.0
            if S == 0.0: R, G, B = V, V, V
            else:
                i = int(H*6.) 
                f = (H*6.)-i 
                p,q,t = V*(1.-S), V*(1.-S*f), V*(1.-S*(1.-f))
                i%=6
                if i == 0: R, G, B = V, t, p
                if i == 1: R, G, B = q, V, p
                if i == 2: R, G, B = p, V, t
                if i == 3: R, G, B = p, q, V
                if i == 4: R, G, B = t, p, V
                if i == 5: R, G, B = V, p, q   
                pixel[0] = max(0, min(255, R*255))
                pixel[1] = max(0, min(255, G*255))
                pixel[2] = max(0, min(255, B*255))
    return img2
 
def RGB2CMY(img):     
    img = img.copy()
    sz = img.shape
    for i in range(sz[0]):
        for j in range(sz[1]):         
            pixel = img[i,j]
            pixel[0] = 255 - pixel[0] # C = 255 - R
            pixel[1] = 255 - pixel[1] # M = 255 - G
            pixel[2] = 255 - pixel[2] # Y = 255 - B
    return img            

img = cv2.imread("img.png")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original in RGB")
plt.show()

img_hsv = RGBtoHSV(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV from RGB")
plt.show()

img_rgb = HSVtoRGB(img_hsv)
plt.imshow(img_rgb)
plt.title("RGB from HSV")
plt.show()

img = cv2.imread("rgb.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original in RGB")
plt.show()

img_cmy = RGB2CMY(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(img_cmy)
plt.title("CMY from RGB")
plt.show()

img_rgb2 = RGB2CMY(img_cmy)
plt.imshow(img_rgb2)
plt.title("RGB from CMY")
plt.show()