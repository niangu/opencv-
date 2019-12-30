import cv2
import sys
import numpy as np

img = cv2.imread('/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (36).jpeg')

def fd(algorithm):
    if algorithm == "SIFT":
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == "SURF":
        return cv2.xfeatures2d.SURF_create(float(sys.argv[3])if len(sys.argv)==4 else 4000)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#fd_alg = fd("SIFT")
fd_alg = fd("SURF")
keypoints, descriptor = fd_alg.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=4, color=(51, 163, 236))

cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
cv2.imshow('keypoints', img)

cv2.waitKey()&0xff
