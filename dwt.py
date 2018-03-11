import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

class DWT:
    def __init__(self, img):
        self.image = img
        self.resized_image = self.resizeImage(self.image, width=256, height=256)
        self.cA, self.cH, self.cV, self.cD = self.dwt2(self.resized_image)

    def showImage(self, image, window='window'):
        cv2.imshow(window, image)
        cv2.waitKey()

    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return cA, cH, cV, cD

    def resizeImage(self, image, width=None, height=None, inter=cv2.INTER_LINEAR):
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def compareImages(self):
        images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]
        result = []
        for path in images_path:
            image = cv2.imread('images/dataset/%s'%path, cv2.IMREAD_GRAYSCALE) / 255
            image_resized = self.resizeImage(image, width=256, height=256)
            cA, cH, cV, cD = self.dwt2(image_resized)
            obj = {'path': path, 'value': np.sum((self.cA - cA)**2)}
            result.append(obj)
        return result

img_path = [arg for arg in sys.argv]
img = cv2.imread(img_path[1], cv2.IMREAD_GRAYSCALE) / 255
dwt = DWT(img)

[print(pic) for pic in sorted(dwt.compareImages(), key=lambda k: k['value'])[0:15]]
