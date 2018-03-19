import buildDatabase as database
import numpy as np
import cv2
import h5py
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
import _pickle as cPickle

args = [arg for arg in sys.argv]
images_path = [f for f in listdir(args[1]) if isfile(join(args[1], f))]

with open('tree.p' if '--pca' not in args else 'tree_pca.p', 'rb') as f:
   tree = cPickle.load(f)

def main():
    query_img = np.uint8(cv2.imread(args[2]))
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV) / 255
    QueryImageClass = database.DWTRootTree(query_img, maxLevel=database.MAX_LEVEL)
    QueryImageClass.run()

    C1_cA_q, C1_cH_q, C1_cV_q, C1_cD_q = database.getLeafNodes(QueryImageClass.C1_cA)
    C2_cA_q, C2_cH_q, C2_cV_q, C2_cD_q = database.getLeafNodes(QueryImageClass.C2_cA)
    C3_cA_q, C3_cH_q, C3_cV_q, C3_cD_q = database.getLeafNodes(QueryImageClass.C3_cA)
    q = list(zip(   np.ravel(C1_cA_q), np.ravel(C1_cH_q), np.ravel(C1_cV_q),\
                    np.ravel(C1_cD_q), np.ravel(C2_cA_q), np.ravel(C2_cH_q),\
                    np.ravel(C2_cV_q), np.ravel(C2_cD_q), np.ravel(C3_cA_q),\
                    np.ravel(C3_cH_q), np.ravel(C3_cV_q),  np.ravel(C3_cD_q)))


    # QueryImageClass.showImage(C1_cD_q)

    ind = tree.query_radius(q, r=0)
    print(ind)
    t = [item for sublist in ind for item in sublist]
    #print(np.unique(np.floor(np.array(t)/(C1_cA_q.shape[:][1]**2)).astype(dtype=np.int), return_counts=True))
    print([images_path[pos] for pos in np.unique(np.floor((np.unique(t) / (C1_cA_q.shape[:][1]*C1_cA_q.shape[:][0]))).astype(dtype=np.int))])

if __name__ == "__main__":
    main()
