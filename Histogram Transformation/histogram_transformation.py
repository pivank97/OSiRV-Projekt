import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def recoverHE(sceneHE):
    for i in range(3):
        sceneHE[:, :, i] =  cv2.equalizeHist(sceneHE[:, :, i])
    return sceneHE

img = cv2.imread("naziv_slike.ekstenzija")
cv2_imshow(img)
sceneHE = recoverHE(img)
cv2_imshow(sceneHE)
