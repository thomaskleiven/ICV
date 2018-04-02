import time
start_time = time.time()
import buildDatabase as database
import numpy as np
import cv2
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
import _pickle as cPickle
import matplotlib.pyplot as plt

args = [arg for arg in sys.argv]

images_path = [f for f in listdir(args[1]) if isfile(join(args[1], f))]

with open('tree.p' if '--pca' not in args else 'tree_pca.p', 'rb') as f:
   tree = cPickle.load(f)

def main():
    query_img = np.uint8(cv2.imread(args[2]))
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2HSV)
    QueryImageClass = database.DWTRootTree(query_img, maxLevel=database.MAX_LEVEL)
    QueryImageClass.run()

    feature_vector = []
    [feature_vector.append(np.ravel(value)) for index, value in enumerate([ database.getLeafNodes(QueryImageClass.C1_cA),\
                                                                            database.getLeafNodes(QueryImageClass.C2_cA),\
                                                                            database.getLeafNodes(QueryImageClass.C3_cA)])]
    if ('--r' in args):
        ind = tree.query_radius(np.ravel(feature_vector).reshape(1,-1), r=args[3])
    else:
        dist, ind = tree.query(np.ravel(feature_vector).reshape(1,-1), int(args[3]))

    print("--- Query time: %.2f seconds ---"% (time.time() - start_time))

    t = [item for sublist in ind for item in sublist]

    images = []
    
    # Open and display returned images
    [images.append(QueryImageClass.resizeImage(cv2.cvtColor(cv2.imread(args[1] + images_path[image]), cv2.COLOR_BGR2RGB))) for image in t]
    show_images(images)

def show_images(images, cols = 5, titles = None):
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.axis('off')
        a.set_title(n)
        plt.imshow(image)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

if __name__ == "__main__":
    if (len(args) < 4):
        print("Syntax: filename.py ./database ./query-image.jpg distance(int) --pca(flag) --r(flag)")
        sys.exit(0)
    main()
