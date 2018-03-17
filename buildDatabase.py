import pywt
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
from sklearn import decomposition
from multiprocessing import Pool
import sys

images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]
args = [arg for arg in sys.argv]

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
            self.components_C1 = self.pca.fit_transform(self.C1)[0:20]
            self.components_C2 = self.pca.fit_transform(self.C2)[0:20]
            self.components_C3 = self.pca.fit_transform(self.C3)[0:20]

            self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.components_C1)
            self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.components_C2)
            self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.components_C3)

        else:
            self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.C1)
            self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.C2)
            self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.C3)

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
    dataset_img = np.uint8(cv2.imread('./images/dataset/%s'%path))
    dataset_img = cv2.cvtColor(dataset_img, cv2.COLOR_BGR2HSV)
    DatasetImageClass = DWTRootTree(dataset_img, maxLevel=MAX_LEVEL)
    DatasetImageClass.run()

    C1_cA, C1_cH, C1_cV, C1_cD = getLeafNodes(DatasetImageClass.C1_cA)
    C2_cA, C2_cH, C2_cV, C2_cD = getLeafNodes(DatasetImageClass.C2_cA)
    C3_cA, C3_cH, C3_cV, C3_cD = getLeafNodes(DatasetImageClass.C3_cA)

    return {
     'C1_cA' if '--pca' not in args else 'pca/C1_cA': C1_cA,\
     'C1_cH' if '--pca' not in args else 'pca/C1_cH': C1_cH,\
     'C1_cV' if '--pca' not in args else 'pca/C1_cV': C1_cV,\
     'C1_cD' if '--pca' not in args else 'pca/C1_cD': C1_cD,\
     'C2_cA' if '--pca' not in args else 'pca/C2_cA': C2_cA,\
     'C2_cH' if '--pca' not in args else 'pca/C2_cH': C2_cH,\
     'C2_cV' if '--pca' not in args else 'pca/C2_cV': C2_cV,\
     'C2_cD' if '--pca' not in args else 'pca/C2_cD': C2_cD,\
     'C3_cA' if '--pca' not in args else 'pca/C3_cA': C3_cA,\
     'C3_cH' if '--pca' not in args else 'pca/C3_cH': C3_cH,\
     'C3_cV' if '--pca' not in args else 'pca/C3_cV': C3_cV,\
     'C3_cD' if '--pca' not in args else 'pca/C3_cD': C3_cD,\
     'path': path }


def handle_output(result):
    dataset = h5py.File('dataset.hdf5', 'a')
    for key, value in result.items():
        if (key == 'path'):
            continue
        try:
            grp = dataset.create_dataset("%s/level%s/%s"%(result['path'], MAX_LEVEL, key), data=value, dtype=np.float)
        except Exception as exc:
            pass
            #print(str(exc))

if __name__ == "__main__":
    pool=Pool()
    dataset = h5py.File('dataset.hdf5', 'a')
    result = []
    for path in images_path:
        r = pool.apply_async(main, (path,), callback=handle_output)
        result.append(r)
    for r in result:
        r.wait()
