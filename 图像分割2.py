import cv2
import numpy as np

filename = '/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (33).jpeg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst>0.01*dst.max()]=[0, 0, 255]
cv2.imshow('dst', img)


cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
cv2.imshow('edged', img)
cv2.waitKey(0)