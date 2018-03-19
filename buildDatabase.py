import pywt
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn import decomposition
from multiprocessing import Pool
import sys
import _pickle as cPickle

args = [arg for arg in sys.argv]
images_path = [f for f in listdir(args[1]) if isfile(join(args[1], f))]

HEIGHT=64
WIDTH=64
MAX_LEVEL=2

class DWTRootTree():
    def __init__(self, img, maxLevel):
        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = None, None, None, None
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = None, None, None, None
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = None, None, None, None
        self.image = self.resizeImage(img)
        self.level = 0
        self.maxLevel = maxLevel
        self.pca = decomposition.PCA()

        # Add color components keys
        self.C1 = np.empty(shape=(HEIGHT, WIDTH))
        self.C2 = np.empty(shape=(HEIGHT, WIDTH))
        self.C3 = np.empty(shape=(HEIGHT, WIDTH))

    def run(self):
        self.getRGBvalues()

        if ('--pca' in args):
            self.components_C1 = self.pca.fit_transform(self.C1)[0:10]
            self.components_C2 = self.pca.fit_transform(self.C2)[0:10]
            self.components_C3 = self.pca.fit_transform(self.C3)[0:10]

            self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.components_C1)
            self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.components_C2)
            self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.components_C3)

        else:
            self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.C1)
            self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.C2)
            self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.C3)

    def showImage(self, image, window='window'):
        cv2.imshow('window', image)
        cv2.waitKey()


    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel), cH, cV, cD

    # Resize
    def resizeImage(self, image, inter=cv2.INTER_LINEAR):
        dim = (HEIGHT, WIDTH)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def getRGBvalues(self):
        self.C1, self.C2, self.C3 = cv2.split(self.image)


class DWTSubTree():
    def __init__(self, img, level, maxLevel):
        self.cA, self.cV, self.cD, self.cH = None, None, None, None
        self.level = level + 1
        self.maxLevel = maxLevel
        self.image = img
        self.run()

    def run(self):
        self.cA, self.cH, self.cV, self.cD = self.dwt2(self.image)

    # DWT Algorithm
    def dwt2(self, img, wavelet='db1',):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs

        if (self.level == self.maxLevel):
            return cA, cH, cV, cD
        return  DWTSubTree(cA, level=self.level, maxLevel=self.maxLevel), cV, cD, cH


def getLeafNodes(tree):
    if (tree.maxLevel == tree.level):
        return tree.cA, tree.cH, tree.cV, tree.cD
    return getLeafNodes(tree.cA)

def main(path):
    dataset_img = np.uint8(cv2.imread(args[1]+'/%s'%path))
    dataset_img = cv2.cvtColor(dataset_img, cv2.COLOR_BGR2HSV) / 255
    DatasetImageClass = DWTRootTree(dataset_img, maxLevel=MAX_LEVEL)
    DatasetImageClass.run()

    C1_cA, C1_cH, C1_cV, C1_cD = getLeafNodes(DatasetImageClass.C1_cA)
    C2_cA, C2_cH, C2_cV, C2_cD = getLeafNodes(DatasetImageClass.C2_cA)
    C3_cA, C3_cH, C3_cV, C3_cD = getLeafNodes(DatasetImageClass.C3_cA)

    return list(zip(np.ravel(C1_cA), np.ravel(C1_cH), np.ravel(C1_cV),\
                    np.ravel(C1_cD), np.ravel(C2_cA), np.ravel(C2_cH),\
                    np.ravel(C2_cV), np.ravel(C2_cD), np.ravel(C3_cA),\
                    np.ravel(C3_cH), np.ravel(C3_cV),  np.ravel(C3_cD)))

if __name__ == "__main__":
    pool=Pool()
    m = []
    result = []
    for path in images_path:
        r = pool.apply_async(main, (path,))
        result.append(r)
    for r in result:
        m.append(r.get())

    t = [item for sublist in m for item in sublist]
    from sklearn.neighbors import KDTree
    tree = KDTree(t)

    cPickle.dump(tree, open('tree_pca.p' if'--pca' in args else 'tree.p','wb'))
