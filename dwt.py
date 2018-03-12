import pywt
import cv2
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from numpy import linalg as LA
from pprint import pprint


class DWT:
    def __init__(self, img, height, width):
        self.dwt_types = ['cA', 'cH', 'cV', 'cD']
        self.height, self.width = height, width
        self.resized_image = self.resizeImage(img)
        self.query_image = {}
        self.dataset_image = {}

        # Add color components keys
        self.query_image['C1'] = { 'colorcomponent' : np.empty(shape=(height, width)) }
        self.query_image['C2'] = { 'colorcomponent' : np.empty(shape=(height, width)) }
        self.query_image['C3'] = { 'colorcomponent' : np.empty(shape=(height, width)) }

        # Add color components keys
        self.dataset_image['C1'] = { 'colorcomponent' : np.empty(shape=(height, width)) }
        self.dataset_image['C2'] = { 'colorcomponent' : np.empty(shape=(height, width)) }
        self.dataset_image['C3'] = { 'colorcomponent' : np.empty(shape=(height, width)) }

    def getDWTQueryImage(self, level):
        for i in range(1,4):
            if ( level > 1):
                dwt = [cA, cH, cV, cD] = self.dwt2(self.query_image['C%i'%i]['cA'])
            else:
                dwt = [cA, cH, cV, cD] = self.dwt2(self.query_image['C%i'%i]['colorcomponent'])
            for index in range(0, len(self.dwt_types)):
                self.query_image['C%i'%i][self.dwt_types[index]] = dwt[index]

    def getDWTDatasetImage(self, level):
        for i in range(1,4):
            if (level > 1):
                dwt = [cA, cH, cV, cD] = self.dwt2(self.dataset_image['C%i'%i]['cA'])
            else:
                dwt = [cA, cH, cV, cD] = self.dwt2(self.dataset_image['C%i'%i]['colorcomponent'])
            print(dwt[0])
            for index in range(0, len(self.dwt_types)):
                self.dataset_image['C%i'%i][self.dwt_types[index]] = dwt[index]


    def getColorSpace(self, R, G, B):
        C1 = ( R + B + G ) / 3
        C2 = ( R + (max(R, G, B) - B) ) / 2
        C3 = ( R + 2 * (( max(R, G, B) - G) + B) ) / 4
        return C1, C2, C3

    def showImage(self, image, window='window'):
        cv2.imshow(window, image)
        cv2.waitKey(0)

    # DWT Algorithm
    def dwt2(self, img, wavelet='db1'):
        coeffs = pywt.dwt2(img, wavelet)
        cA, (cH, cV, cD) = coeffs
        print(img.shape[:])
        print(cA.shape[:])
        self.showImage(cA)
        return cA, cH, cV, cD

    # Resize
    def resizeImage(self, image, inter=cv2.INTER_LINEAR):
        dim = (self.width, self.height)
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def getRGBvalues(self, image, query):
        # Get new height and width
        height, width = image.shape[:2]

        # Iterate all pixels
        for row in range(0, height):
            for col in range(0, width):
                C1_pixel, C2_pixel, C3_pixel = self.getColorSpace(image[row, col, 0], image[row, col, 1], image[row, col, 2])

                if query:
                    self.query_image['C1']['colorcomponent'][row][col] = C1_pixel
                    self.query_image['C2']['colorcomponent'][row][col] = C2_pixel
                    self.query_image['C3']['colorcomponent'][row][col] = C3_pixel
                else:
                    self.dataset_image['C1']['colorcomponent'][row][col] = C1_pixel
                    self.dataset_image['C2']['colorcomponent'][row][col] = C2_pixel
                    self.dataset_image['C3']['colorcomponent'][row][col] = C3_pixel


    def computeDistance(self):
        res = 0
        for i in range(1,4):
            img_ca = self.query_image['C%i'%i]['cA']
            img_m_ca = self.dataset_image['C%i'%i]['cA']

            img_cv = self.query_image['C%i'%i]['cV']
            img_m_cv = self.dataset_image['C%i'%i]['cV']

            img_ch = self.query_image['C%i'%i]['cH']
            img_m_ch = self.dataset_image['C%i'%i]['cH']

            img_cd = self.query_image['C%i'%i]['cD']
            img_m_cd = self.dataset_image['C%i'%i]['cD']


            res +=  LA.norm(img_ca -img_m_ca, ord=2)+\
                    LA.norm(img_cv -img_m_cv, ord=2)+\
                    LA.norm(img_ch -img_m_ch, ord=2)+\
                    LA.norm(img_cd-img_m_cd, ord=2)
        return res


HEIGHT=256
WIDTH=256
images_path = [f for f in listdir('./images/dataset') if isfile(join('./images/dataset', f))]
result = []
levels = 2

if __name__ == "__main__":
    img_path = [arg for arg in sys.argv]
    # Normalize image
    img = cv2.imread(img_path[1]) / 255
    print(img.shape[:])

    #Counter
    index = 0

    # Init class
    dwt = DWT(img,height=HEIGHT, width=WIDTH)

    for path in images_path:
        sys.stdout.write('%s\r' % str(round(index/len(images_path), 2)))
        sys.stdout.flush()
        image = cv2.imread('images/dataset/%s'%path) / 255
        image_resized = dwt.resizeImage(image)


        # Get color space query image
        dwt.getRGBvalues(dwt.resized_image, query=True)
        dwt.getDWTQueryImage(1)

        # Get RGB values
        dwt.getRGBvalues(image_resized, query=False)
        dwt.getDWTDatasetImage(1)
        break
        for level in levels:
            dwt.getRGBvalues(dwt.query_image[''], query=True)
            dwt.getDWTQueryImage(1)

            # Get RGB values
            dwt.getRGBvalues(image_resized, query=False)
            dwt.getDWTDatasetImage(1)

        res = dwt.computeDistance()
        obj = {'path': path, 'value': res}
        result.append(obj)
        index += 1

[print(pic) for pic in sorted(result, key=lambda k: k['value'])[0:15]]
