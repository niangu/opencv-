import cv2
import numpy as np
from matplotlib import pyplot as plt

roi = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg')#/home/niangu/桌面/答题卡识别/6.jpeg
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('/home/niangu/桌面/答题卡识别/6.jpeg')#/home/niangu/桌面/答题卡识别/1.jpeg
hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dst = cv2.filter2D(dst, -1, disc)

ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))

res = cv2.bitwise_and(target, thresh)
cv2.namedWindow('1', cv2.WINDOW_NORMAL)
res = np.hstack((target, thresh, res))
cv2.imshow('1', res)
cv2.waitKey(0)