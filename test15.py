# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from numba import jit
import bisect
import matplotlib.pyplot as plt
import scipy.signal as signal


def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r, c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r, c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r, c] = vd
    return dst


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


img = cv.imread('/home/niangu/桌面/答题卡识别/text/111.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# f = adapthisteq(f);
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cll = clahe.apply(gray)

# Low_High = stretchlim(f, [0.0 0.3]);
# f= imadjust(f, [0.0 0.25], [ ]);
adjust = imadjust(cll, 0, [0, 165])

# f = medfilt2(f, [7 5]);
img_median = cv.medianBlur(adjust, 5)

# f = im2bw(f, graythresh(f));
ret, thresh = cv.threshold(img_median, 0, 255, cv.THRESH_OTSU)

# f = imopen(f, strel('square', 4));
# f = imclose(f, strel('square', 4));
kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
ret1 = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
ret2 = cv.morphologyEx(ret1, cv.MORPH_CLOSE, kernel, iterations=1)

# f = imerode(f, strel('square', 4));
kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
erosion = cv.erode(ret2, kernel, iterations=1)

# rotated = cv.rotate(erosion, 90)
# rotated_img = cv.rotate(img, 90)

edged = cv.Canny(img, 50, 150, apertureSize=3)

# minLineLength = 500
# maxLineGap = 100
# lines = cv.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength, maxLineGap)
# for x1, y1, x2, y2 in lines[0]:
#     cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

lines = cv.HoughLines(edged, 1, np.pi / 180, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# img = cv.rotate(rotated_img, -90)

# gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)
# edged = cv.Canny(gaussian_bulr, 50, 150)
cv.namedWindow('edges', cv.WINDOW_NORMAL)
cv.imshow('edges', img)