import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from pprint import pprint
from sklearn import decomposition
from matplotlib import pyplot as plt

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = "none"
mpl.rcParams['font.size'] = 16

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def plot(data):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xlabel("\$\$")
    ax.set_ylabel("\$\$")
    ax.plot(data, '-o', color="black")
    plt.show()

HEIGHT=512
WIDTH=512
MAX_LEVEL=1

class DWTRootTree():
    def __init__(self, img, maxLevel):
        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = None, None, None, None
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = None, None, None, None
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = None, None, None, None
        self.image = self.resizeImage(img)
        self.level = 0
        self.maxLevel = maxLevel
        self.pca = decomposition.PCA(n_components=10)

        # Add color components keys
        self.C1 = np.empty(shape=(HEIGHT, WIDTH))
        self.C2 = np.empty(shape=(HEIGHT, WIDTH))
        self.C3 = np.empty(shape=(HEIGHT, WIDTH))

    def run(self):
        self.getRGBvalues()
        self.components_C1 = self.pca.fit_transform(self.C1)
        self.components_C2 = self.pca.fit_transform(self.C2)
        self.components_C3 = self.pca.fit_transform(self.C3)

        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.components_C1)
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.components_C2)
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.components_C3)


    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel), cH, cV, cD

    def showImage(self, image, window='window'):
        cv2.imshow('window', image)
        cv2.waitKey()

    # Resize
    def resizeImage(self, image, inter=cv2.INTER_LINEAR):
        dim = (HEIGHT, WIDTH)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def getRGBvalues(self):
        # Iterate all pixels
        for row in range(0, HEIGHT):
            for col in range(0, WIDTH):
                C1_pixel, C2_pixel, C3_pixel = self.image[row, col, 0], self.image[row, col, 1], self.image[row, col, 2]
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
    img = np.uint8(cv2.imread(img_path[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    QueryImageClass = DWTRootTree(img, maxLevel=MAX_LEVEL)
    QueryImageClass.run()

    def getLeafs(tree):
        if (tree.maxLevel == tree.level):
            return tree.cA, tree.cH, tree.cV, tree.cD
        return getLeafs(tree.cA)

    C1_cA_q, C1_cH_q, C1_cV_q, C1_cD_q = getLeafs(QueryImageClass.C1_cA)
    C2_cA_q, C2_cH_q, C2_cV_q, C2_cD_q = getLeafs(QueryImageClass.C2_cA)
    C3_cA_q, C3_cH_q, C3_cV_q, C3_cD_q = getLeafs(QueryImageClass.C3_cA)

    for path in images_path:
        dataset_img = np.uint8(cv2.imread('./images/dataset/%s'%path))
        dataset_img = cv2.cvtColor(dataset_img, cv2.COLOR_BGR2HSV)
        DatasetImageClass = DWTRootTree(dataset_img, maxLevel=MAX_LEVEL)
        DatasetImageClass.run()
        C1_cA_d, C1_cH_d, C1_cV_d, C1_cD_d = getLeafs(DatasetImageClass.C1_cA)
        C2_cA_d, C2_cH_d, C2_cV_d, C2_cD_d = getLeafs(DatasetImageClass.C2_cA)
        C3_cA_d, C3_cH_d, C3_cV_d, C3_cD_d = getLeafs(DatasetImageClass.C3_cA)

        distance =  LA.norm(C1_cA_q - C1_cA_d, ord=2) + LA.norm(C1_cH_q - C1_cH_d, ord=2) + LA.norm(C1_cV_q - C1_cV_d, ord=2) + LA.norm(C1_cD_q - C1_cD_d, ord=2)+\
                    LA.norm(C2_cA_q - C2_cA_d, ord=2) + LA.norm(C2_cH_q - C2_cH_d, ord=2) + LA.norm(C2_cV_q - C2_cV_d, ord=2) + LA.norm(C2_cD_q - C2_cD_d, ord=2)+\
                    LA.norm(C3_cA_q - C3_cA_d, ord=2) + LA.norm(C3_cH_q - C3_cH_d, ord=2) + LA.norm(C3_cV_q - C3_cV_d, ord=2) + LA.norm(C3_cD_q - C3_cD_d, ord=2)

        result.append({ 'path': path, 'distance': distance})

[print(pic) for pic in sorted(result, key=lambda k: k['distance'])[0:15]]
