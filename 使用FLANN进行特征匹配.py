import numpy as np
import cv2
from matplotlib import pyplot as plt

queryImage = cv2.imread('/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (36).jpeg', 0)
trainingImage = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (27).jpeg', 0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(queryImage, None)
kp2, des2 = sift.detectAndCompute(trainingImage, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

for i,(m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1, 0]

drawParams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)

cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
cv2.imshow('keypoints', resultImage)

cv2.waitKey()&0xff
