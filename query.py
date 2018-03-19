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

    feature_vector = []
    [feature_vector.append(np.ravel(value)) for index, value in enumerate([ database.getLeafNodes(QueryImageClass.C1_cA),\
                                                                            database.getLeafNodes(QueryImageClass.C2_cA),\
                                                                            database.getLeafNodes(QueryImageClass.C3_cA)])]

    ind = tree.query_radius(np.ravel(feature_vector).reshape(1,-1), r=args[3])
    t = [item for sublist in ind for item in sublist]
    [print(images_path[image]) for image in t]

if __name__ == "__main__":
    if (len(args) != 4):
        print("Syntax: filename.py ./database ./query-image.jpg distance(int) --pca(flag)")
        sys.exit(0)
    main()
