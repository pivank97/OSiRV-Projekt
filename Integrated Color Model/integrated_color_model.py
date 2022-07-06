import cv2
import numpy as np
from skimage.color import rgb2hsv,hsv2rgb
from google.colab.patches import cv2_imshow

def sceneRadianceRGB(sceneICM):
    sceneICM = np.clip(sceneICM, 0, 255)
    sceneICM = np.uint8(sceneICM)
    return sceneICM

def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        max_channel = np.max(img[:,:,k])
        min_channel = np.min(img[:,:,k])
        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - min_channel) * (255 - 0) / (max_channel - min_channel) + 0
    return img

def globalStretching(img_L, height, width):
    img_min = np.min(img_L)
    img_max = np.max(img_L)
    img_mean = np.mean(img_L)
    arrayGlobalStretching = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            pOut = (img_L[i][j] - img_min) * ((1) / (img_max - img_min))
            arrayGlobalStretching[i][j] = pOut
    return arrayGlobalStretching

def HSVstretching(sceneICM):
    height = len(sceneICM)
    width = len(sceneICM[0])
    img_hsv = rgb2hsv(sceneICM)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = globalStretching(s, height, width)
    img_v_stretching = globalStretching(v, height, width)
    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255
    return img_rgb

img = cv2.imread("naziv_slike.ekstenzija")
cv2_imshow(img)
img = stretching(img)
sceneICM = sceneRadianceRGB(img)
sceneICM = HSVstretching(sceneICM)
sceneICM = sceneRadianceRGB(sceneICM)
cv2_imshow(sceneICM)
