import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from pprint import pprint

HEIGHT=256
WIDTH=256

class DWTRootTree():
    def __init__(self, img, maxLevel):
        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = None, None, None, None
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = None, None, None, None
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = None, None, None, None
        self.image = self.resizeImage(img)
        self.level = 0
        self.maxLevel = maxLevel

        # Add color components keys
        self.C1 = np.empty(shape=(HEIGHT, WIDTH))
        self.C2 = np.empty(shape=(HEIGHT, WIDTH))
        self.C3 = np.empty(shape=(HEIGHT, WIDTH))

    def run(self):
        self.getRGBvalues()
        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.C1)
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.C2)
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.C3)

    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cH, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cV, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cD, level=self.level, maxLevel=self.maxLevel)

    def showImage(self, image, window='window'):
        cv2.imshow('window', image)
        cv2.waitKey()

    # Resize
    def resizeImage(self, image, inter=cv2.INTER_LINEAR):
        dim = (HEIGHT, WIDTH)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def getColorSpace(self, R, G, B):
        C1 = ( R + B + G ) / 3
        C2 = ( R + (max(R, G, B) - B) ) / 2
        C3 = ( R + 2 * (( max(R, G, B) - G) + B) ) / 4
        return C1, C2, C3

    def getRGBvalues(self):
        # Iterate all pixels
        for row in range(0, HEIGHT):
            for col in range(0, WIDTH):
                C1_pixel, C2_pixel, C3_pixel = self.getColorSpace(self.image[row, col, 0], self.image[row, col, 1], self.image[row, col, 2])
                self.C1[row][col] = C1_pixel
                self.C2[row][col] = C2_pixel
                self.C3[row][col] = C3_pixel


class DWTSubTree():
    def __init__(self, img, level, maxLevel):

        self.cA, self.cV, self.cD, self.cH = None, None, None, None
        self.level = level + 1
        self.maxLevel = maxLevel
        self.image = img
        self.run()

    def run(self):
        if (self.level == self.maxLevel):
            return self.image

        self.cA, self.cV, self.cD, self.cH = self.dwt2(self.image)

    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cV, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cD, level=self.level, maxLevel=self.maxLevel),\
                DWTSubTree(cH, level=self.level, maxLevel=self.maxLevel)


img_path = [arg for arg in sys.argv]
images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]

if __name__ == "__main__":
    img = cv2.imread(img_path[1]) / 255
    root = DWTRootTree(img, maxLevel=3)
    root.run()

    def getLeaf(tree):
        if (tree.maxLevel == tree.level):
            return root.showImage(tree.image)
        return getLeaf(tree.cA)

    getLeaf(root.C1_cA)
    getLeaf(root.C1_cV)
    getLeaf(root.C1_cD)

    # print(LA.norm(root.C1_cA.cA.image - root2.C1_cA.cA.image, ord=2))
