import os
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def recoverGC(sceneGC):
    sceneGC = sceneGC / 255.0
    for i in range(3):
        sceneGC[:, :, i] =  np.power(sceneGC[:, :, i] / float(np.max(sceneGC[:, :, i])), 0.65)
    sceneGC = np.clip(sceneGC * 255, 0, 255)
    sceneGC = np.uint8(sceneGC)
    return sceneGC

img = cv2.imread("naziv_slike.ekstenzija")
cv2_imshow(img)
sceneGC = recoverGC(img)
cv2_imshow(sceneGC)
