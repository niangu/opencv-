import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (23).jpg", 0)
img2 = cv2.imread("/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (26).jpg", 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:10], None, flags=2)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img3)
cv2.waitKey(0)
