import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from pprint import pprint

HEIGHT=512
WIDTH=512
MAX_LEVEL=4

def showImage(image, window='window'):
    cv2.imshow(window, image)
    cv2.waitKey()

def resizeImage(image, inter=cv2.INTER_LINEAR):
    dim = (WIDTH, HEIGHT)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

class DWTSubTree():
    def __init__(self, img, level, maxLevel):

        self.cA, self.cV, self.cD, self.cH = None, None, None, None
        self.level = level + 1
        self.maxLevel = maxLevel
        self.image = img
        self.run()

    def run(self):
        self.cA, self.cV, self.cD, self.cH = self.dwt2(self.image)

    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        if (self.level == self.maxLevel):
            return cA, cH, cV, cD

        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel), cV, cD, cH


img_path = [arg for arg in sys.argv]
images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]

if __name__ == "__main__":
    result = []

    #Read as greyscale
    img = cv2.imread(img_path[1], 0) / 255

    QueryImageClass = DWTSubTree(resizeImage(img), level=0, maxLevel=MAX_LEVEL)
    QueryImageClass.run()

    def getLeafNodes(tree):
        if (tree.maxLevel == tree.level):
            return tree.cA, tree.cH, tree.cV, tree.cD
        return getLeafNodes(tree.cA)

    cA_q, cH_q, cV_q, cD_q = getLeafNodes(QueryImageClass.cA)

    for path in images_path:
        # Read as grey-scale
        dataset_img = cv2.imread('./images/dataset/%s'%path, 0) / 255
        DatasetImageClass = DWTSubTree(resizeImage(dataset_img), level=0, maxLevel=MAX_LEVEL)
        DatasetImageClass.run()
        cA_d, cH_d, cV_d, cD_d = getLeafNodes(DatasetImageClass.cA)

        distance =  LA.norm(cA_q - cA_d, ord=2) + LA.norm(cH_q - cH_d, ord=2) + LA.norm(cV_q - cV_d, ord=2) + LA.norm(cD_q - cD_d, ord=2)

        result.append({ 'path': path, 'distance': distance})

[print(pic) for pic in sorted(result, key=lambda k: k['distance'])[0:15]]
