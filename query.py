import buildDatabase as database
import numpy as np
import cv2
import h5py
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA

dataset = h5py.File('dataset.hdf5', 'r+')
images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]
args = [arg for arg in sys.argv]
MAX_LEVEL=2
result=[]

def getLeafNodes(tree):
    if (tree.maxLevel == tree.level):
        return tree.cA, tree.cH, tree.cV, tree.cD
    return getLeafNodes(tree.cA)

def main():
    query_img = np.uint8(cv2.imread(args[1]))
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)
    QueryImageClass = database.DWTRootTree(query_img, maxLevel=MAX_LEVEL)
    QueryImageClass.run()

    C1_cA_q, C1_cH_q, C1_cV_q, C1_cD_q = database.getLeafNodes(QueryImageClass.C1_cA)
    C2_cA_q, C2_cH_q, C2_cV_q, C2_cD_q = database.getLeafNodes(QueryImageClass.C2_cA)
    C3_cA_q, C3_cH_q, C3_cV_q, C3_cD_q = database.getLeafNodes(QueryImageClass.C3_cA)

    for path in images_path:
        try:
            C1_cA_d, C1_cH_d, C1_cV_d, C1_cD_d = dataset[path]['level%s'%MAX_LEVEL]['pca/C1_cA' if '--pca' in args else 'C1_cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C1_cH' if '--pca' in args else 'C1_cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C1_cV' if '--pca' in args else 'C1_cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C1_cD' if '--pca' in args else 'C1_cD'].value

            C2_cA_d, C2_cH_d, C2_cV_d, C2_cD_d = dataset[path]['level%s'%MAX_LEVEL]['pca/C2_cA' if '--pca' in args else 'C2_cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C2_cH' if '--pca' in args else 'C2_cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C2_cV' if '--pca' in args else 'C2_cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C2_cD' if '--pca' in args else 'C2_cD'].value

            C3_cA_d, C3_cH_d, C3_cV_d, C3_cD_d = dataset[path]['level%s'%MAX_LEVEL]['pca/C3_cA' if '--pca' in args else 'C3_cA'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C3_cH' if '--pca' in args else 'C3_cH'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C3_cV' if '--pca' in args else 'C3_cV'].value,\
                                                 dataset[path]['level%s'%MAX_LEVEL]['pca/C3_cD' if '--pca' in args else 'C3_cD'].value

            distance =  LA.norm(C1_cA_q - C1_cA_d, ord=2) +LA.norm(C1_cH_q - C1_cH_d, ord=2) + LA.norm(C1_cV_q - C1_cV_d, ord=2) + LA.norm(C1_cD_q - C1_cD_d, ord=2)+\
                        LA.norm(C2_cA_q - C2_cA_d, ord=2) +LA.norm(C2_cH_q - C2_cH_d, ord=2) + LA.norm(C2_cV_q - C2_cV_d, ord=2) + LA.norm(C2_cD_q - C2_cD_d, ord=2)+\
                        LA.norm(C3_cA_q - C3_cA_d, ord=2) +LA.norm(C3_cH_q - C3_cH_d, ord=2) + LA.norm(C3_cV_q - C3_cV_d, ord=2) + LA.norm(C3_cD_q - C3_cD_d, ord=2)

            result.append({ 'path': path, 'distance': distance})
        except Exception as exc:
            print(str(exc))
    [print('Image %s, distance: %s'%(pic['path'], pic['distance'])) for pic in sorted(result, key=lambda k: k['distance'])[0:15]]

if __name__ == "__main__":
    main()
