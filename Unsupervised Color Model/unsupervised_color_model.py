import os
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from skimage.color import rgb2hsv,hsv2rgb

def calEqualisation(img, ratio):
    array = img * ratio
    array = np.clip(array, 0, 255)
    return array

def RGBequalisation(img):
    img = np.float32(img)
    avgRGB = []
    for i in range(3):
        avg = np.mean(img[:,:,i])
        avgRGB.append(avg)
    avgR = avgRGB[0] / avgRGB[2]
    avgG =  avgRGB[0] / avgRGB[1]
    ratio = [0,avgG,avgR]
    for i in range(1,3):
        img[:,:,i] = calEqualisation(img[:,:,i], ratio[i])
    return img

def histogramR(rArray, height, width):
    length = height * width
    r_Array = []
    for i in range(height):
        for j in range(width):
            r_Array.append(rArray[i][j])
    r_Array.sort()
    img_min = int(r_Array[int(length / 500)])
    img_max = int(r_Array[-int(length / 500)])
    arrayGlobalStretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if rArray[i][j] < img_min:
                arrayGlobalStretching[i][j] = img_min
            elif (rArray[i][j] > img_max):
                pOut = rArray[i][j]
                arrayGlobalStretching[i][j] = 255
            else:
                pOut = int((rArray[i][j] - img_min) * ((255 - img_min) / (img_max - img_min))) + img_min
                arrayGlobalStretching[i][j] = pOut
    return (arrayGlobalStretching)

def histogramG(rArray, height, width):
    length = height * width
    r_Array = []
    for i in range(height):
        for j in range(width):
            r_Array.append(rArray[i][j])
    r_Array.sort()
    img_min = int(r_Array[int(length / 500)])
    img_max = int(r_Array[-int(length / 500)])
    arrayGlobalStretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if rArray[i][j] < img_min:
                pOut = rArray[i][j]
                arrayGlobalStretching[i][j] = 0
            elif (rArray[i][j] > img_max):
                pOut = rArray[i][j]
                arrayGlobalStretching[i][j] = 255
            else:
                pOut = int((rArray[i][j] - img_min) * ((255) / (img_max - img_min)) )
                arrayGlobalStretching[i][j] = pOut
    return (arrayGlobalStretching)

def histogramB(rArray, height, width):
    length = height * width
    r_Array = []
    for i in range(height):
        for j in range(width):
            r_Array.append(rArray[i][j])
    r_Array.sort()
    img_min = int(r_Array[int(length / 500)])
    img_max = int(r_Array[-int(length / 500)])
    arrayGlobalStretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if rArray[i][j] < img_min:
                arrayGlobalStretching[i][j] = 0
            elif (rArray[i][j] > img_max):
                arrayGlobalStretching[i][j] = img_max
            else:
                pOut = int((rArray[i][j] - img_min) * ((img_max) / (img_max - img_min)))
                arrayGlobalStretching[i][j] = pOut
    return (arrayGlobalStretching)

def stretching(img):
    height = len(img)
    width = len(img[0])
    img[:, :, 2] = histogramR(img[:, :, 2], height, width)
    img[:, :, 1] = histogramG(img[:, :, 1], height, width)
    img[:, :, 0] = histogramB(img[:, :, 0], height, width)
    return img

def globalStretching(img_L, height, width):
    img_min = np.min(img_L)
    img_max = np.max(img_L)
    arrayGlobalStrecthingL = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            pOut = (img_L[i][j] - img_min) * ((1) / (img_max - img_min))
            arrayGlobalStrecthingL[i][j] = pOut
    return arrayGlobalStrecthingL

def HSVStretching(sceneUCM):
    sceneUCM = np.uint8(sceneUCM)
    height = len(sceneUCM)
    width = len(sceneUCM[0])
    img_hsv = rgb2hsv(sceneUCM)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = globalStretching(s, height, width)
    img_v_stretching = globalStretching(v, height, width)
    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255
    return img_rgb

def sceneRadianceRGB(sceneUCM):
    sceneUCM = np.clip(sceneUCM, 0, 255)
    sceneUCM = np.uint8(sceneUCM)
    return sceneUCM

img = cv2.imread("naziv_slike.ekstenzija")
cv2_imshow(img)
sceneUCM = RGBequalisation(img)
sceneUCM = stretching(sceneUCM)
sceneUCM = HSVStretching(sceneUCM)
sceneUCM = sceneRadianceRGB(sceneUCM)
cv2_imshow(sceneUCM)

