# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from numba import jit
import bisect
import matplotlib.pyplot as plt
import scipy.signal as signal




img = cv.imread('/home/niangu/桌面/答题卡识别/text/111.jpg')
# https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python
# @jit
# def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
#     # src : input one-layer image (numpy array)
#     # tol : tolerance, from 0 to 100.
#     # vin  : src image bounds
#     # vout : dst image bounds
#     # return : output img
#
#     assert len(src.shape) == 2 ,'Input image should be 2-dims'
#
#     tol = max(0, min(100, tol))
#
#     if tol > 0:
#         # Compute in and out limits
#         # Histogram
#         hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
#
#         # Cumulative histogram
#         cum = hist.copy()
#         for i in range(1, 256): cum[i] = cum[i - 1] + hist[i]
#
#         # Compute bounds
#         total = src.shape[0] * src.shape[1]
#         low_bound = total * tol / 100
#         upp_bound = total * (100 - tol) / 100
#         vin[0] = bisect.bisect_left(cum, low_bound)
#         vin[1] = bisect.bisect_left(cum, upp_bound)
#
#     # Stretching
#     scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
#     vs = src-vin[0]
#     vs[src<vin[0]]=0
#     vd = vs*scale+0.5 + vout[0]
#     vd[vd>vout[1]] = vout[1]
#     dst = vd
#
#     return dst


#img = cv.imread('/home/niangu/桌面/答题卡识别/webwxgetmsgimg (16).jpeg')

"""
rows, cols, ch = img.shape

pts1 = np.float32([[120, 105], [940, 105], [120, 1340], [970, 1310]])#找到四个点
pts2 = np.float32([[0, 0], [1060, 0], [0, 1430], [1060, 1430]])#替换的四个点

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(img, M, (1060, 1430))

"""
#img = cv.imread('/home/niangu/桌面/答题卡识别/webwxgetmsgimg (17).jpeg')

blur4 = cv.bilateralFilter(img,9,75,75)
blur3 = cv.GaussianBlur(img, (5, 5), 0)
median = cv.medianBlur(img, 5)
blur2 = cv.blur(img,(5,5))
gray = cv.cvtColor(median, cv.COLOR_BGR2GRAY)
kernel = np.ones((5,5), np.uint8)
opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

# f = adapthisteq(f);
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cll = clahe.apply(gray)


# rotated = cv.rotate(erosion, 90)
# rotated_img = cv.rotate(img, 90)

edged = cv.Canny(opening, 50, 150, apertureSize=3)
"""
minLineLength = 10
maxLineGap = 20
lines = cv.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

"""
lines = cv.HoughLines(edged, 1, np.pi / 180, 200)
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        #cv.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)

# img = cv.rotate(rotated_img, -90)

# gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)
# edged = cv.Canny(gaussian_bulr, 50, 150)
#cv.imwrite("2.jpg", img)
#cv.namedWindow('adaptive4', cv.WINDOW_NORMAL)


cv.namedWindow('adaptive41', cv.WINDOW_NORMAL)
cv.imshow('adaptive41', edged)

#cv.imshow('adaptive4', dst)

cv.namedWindow('adaptive41s', cv.WINDOW_NORMAL)
cv.imshow('adaptive41s', img)



cv.waitKey(0)&0xFF