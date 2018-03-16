import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from pprint import pprint
import h5py

dataset = h5py.File("dataset.hdf5", "r+")

HEIGHT=512
WIDTH=512
MAX_LEVEL=2

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

    def run(self, path):
        self.getRGBvalues()
        self.C1_cA, self.C1_cV, self.C1_cD, self.C1_cH = self.dwt2(self.C1, path, id='C1')
        self.C2_cA, self.C2_cV, self.C2_cD, self.C2_cH = self.dwt2(self.C2, path, id='C2')
        self.C3_cA, self.C3_cV, self.C3_cD, self.C3_cH = self.dwt2(self.C3, path, id='C3')

    # DWT Algorithm
    def dwt2(self, img, path,id, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        return  DWTSubTree(cA, path, id=id, level=self.level, maxLevel=self.maxLevel), cH, cV, cD

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
    def __init__(self, img, path, level, maxLevel, id):

        self.cA, self.cV, self.cD, self.cH = None, None, None, None
        self.level = level + 1
        self.maxLevel = maxLevel
        self.image = img
        self.id = id
        self.path = path
        self.run(path)

    def run(self, path):
        self.cA, self.cH, self.cV, self.cD = self.dwt2(self.image, path)

    # DWT Algorithm
    def dwt2(self, img, path, wavelet='db1',):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs

        if (self.level == self.maxLevel):
            try:
                grp = dataset.create_dataset("%s/level%s/%s/cA/"%(path, self.level, self.id), data=cA, dtype=np.float)
                grp.attrs['std'] = np.std(cA)

                grp = dataset.create_dataset("%s/level%s/%s/cH/"%(path, self.level, self.id), data=cH, dtype=np.float)
                grp.attrs['std'] = np.std(cH)

                grp = dataset.create_dataset("%s/level%s/%s/cV/"%(path, self.level, self.id), data=cV, dtype=np.float)
                grp.attrs['std'] = np.std(cV)

                grp = dataset.create_dataset("%s/level%s/%s/cD/"%(path, self.level, self.id), data=cD, dtype=np.float)
                grp.attrs['std'] = np.std(cD)
            except Exception as exc:
                print(str(exc))
            return cA, cH, cV, cD
        return  DWTSubTree(cA, path=self.path, id=self.id, level=self.level, maxLevel=self.maxLevel), cV, cD, cH


img_path = [arg for arg in sys.argv]
images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]

# According to paper only std of cA is computed
def filterBySTD(datasetPath, queryPath, beta=0.5):
    sigma_one_marked = dataset[datasetPath]['level%s'%MAX_LEVEL]['C1']['cA'].attrs['std']
    sigma_two_marked = dataset[datasetPath]['level%s'%MAX_LEVEL]['C2']['cA'].attrs['std']
    sigma_three_marked = dataset[datasetPath]['level%s'%MAX_LEVEL]['C3']['cA'].attrs['std']

    sigma_one = dataset[queryPath]['level%s'%MAX_LEVEL]['C1']['cA'].attrs['std']
    sigma_two = dataset[queryPath]['level%s'%MAX_LEVEL]['C2']['cA'].attrs['std']
    sigma_three = dataset[queryPath]['level%s'%MAX_LEVEL]['C3']['cA'].attrs['std']

    if (
        sigma_one * beta < sigma_one_marked and sigma_one_marked < sigma_one / beta\
        or (sigma_two * beta < sigma_two_marked and sigma_two_marked < sigma_two / beta)\
        and (sigma_three * beta < sigma_three_marked and sigma_three_marked < sigma_three / beta)\
       ):
        return True
    return False


def main():
    result = []
    img = np.uint8(cv2.imread(img_path[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    QueryImageClass = DWTRootTree(img, maxLevel=MAX_LEVEL)
    QueryImageClass.run(images_path[1])

    def getLeafs(tree):
        if (tree.maxLevel == tree.level):
            return tree.cA, tree.cH, tree.cV, tree.cD
        return getLeafs(tree.cA)

    C1_cA_q, C1_cH_q, C1_cV_q, C1_cD_q = getLeafs(QueryImageClass.C1_cA)
    C2_cA_q, C2_cH_q, C2_cV_q, C2_cD_q = getLeafs(QueryImageClass.C2_cA)
    C3_cA_q, C3_cH_q, C3_cV_q, C3_cD_q = getLeafs(QueryImageClass.C3_cA)

    for path in images_path:

        if ("%s/level%s"%(path, MAX_LEVEL) in dataset):
            if(not filterBySTD(datasetPath=path, queryPath=images_path[1])):
                continue
            C1_cA_d, C1_cH_d, C1_cV_d, C1_cD_d = dataset[path]['level%s'%MAX_LEVEL]['C1']['cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C1']['cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C1']['cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C1']['cD'].value

            C2_cA_d, C2_cH_d, C2_cV_d, C2_cD_d = dataset[path]['level%s'%MAX_LEVEL]['C2']['cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C2']['cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C2']['cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C2']['cD'].value

            C3_cA_d, C3_cH_d, C3_cV_d, C3_cD_d = dataset[path]['level%s'%MAX_LEVEL]['C3']['cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C3']['cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C3']['cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['C3']['cD'].value

        else:
            dataset_img = np.uint8(cv2.imread('./images/dataset/%s'%path))
            dataset_img = cv2.cvtColor(dataset_img, cv2.COLOR_BGR2HSV)
            DatasetImageClass = DWTRootTree(dataset_img, maxLevel=MAX_LEVEL)
            DatasetImageClass.run(path)
            C1_cA_d, C1_cH_d, C1_cV_d, C1_cD_d = getLeafs(DatasetImageClass.C1_cA)
            C2_cA_d, C2_cH_d, C2_cV_d, C2_cD_d = getLeafs(DatasetImageClass.C2_cA)
            C3_cA_d, C3_cH_d, C3_cV_d, C3_cD_d = getLeafs(DatasetImageClass.C3_cA)

        distance =  LA.norm(C1_cA_q - C1_cA_d, ord=2) +2*LA.norm(C1_cH_q - C1_cH_d, ord=2) + 2*LA.norm(C1_cV_q - C1_cV_d, ord=2) + 2*LA.norm(C1_cD_q - C1_cD_d, ord=2)+\
                    LA.norm(C2_cA_q - C2_cA_d, ord=2) +2*LA.norm(C2_cH_q - C2_cH_d, ord=2) + 2*LA.norm(C2_cV_q - C2_cV_d, ord=2) + 2*LA.norm(C2_cD_q - C2_cD_d, ord=2)+\
                    LA.norm(C3_cA_q - C3_cA_d, ord=2) +2*LA.norm(C3_cH_q - C3_cH_d, ord=2) + 2*LA.norm(C3_cV_q - C3_cV_d, ord=2) + 2*LA.norm(C3_cD_q - C3_cD_d, ord=2)

        result.append({ 'path': path, 'distance': distance})
    [print('Image %s, distance: %s'%(pic['path'], pic['distance'])) for pic in sorted(result, key=lambda k: k['distance'])[0:15]]

if __name__ == "__main__":
    main()
    pass

# LA.norm(C1_cH_q - C1_cH_d, ord=2) + LA.norm(C1_cV_q - C1_cV_d, ord=2) + LA.norm(C1_cD_q - C1_cD_d, ord=2)+
# LA.norm(C2_cH_q - C2_cH_d, ord=2) + LA.norm(C2_cV_q - C2_cV_d, ord=2) + LA.norm(C2_cD_q - C2_cD_d, ord=2)+
# LA.norm(C3_cH_q - C3_cH_d, ord=2) + LA.norm(C3_cV_q - C3_cV_d, ord=2) + LA.norm(C3_cD_q - C3_cD_d, ord=2)
