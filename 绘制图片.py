import numpy as np
import cv2

import cv2
import imutils
import numpy as np

img = cv2.imread('/home/niangu/桌面/答题卡识别/答案提取.png')
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测


mask = np.zeros(edged.shape, np.uint8)

cv2.line(mask, (0, 7), (400, 7), (255, 0, 0), 1)
for i in range(0, 5):
    j = i*100
    cv2.line(mask, (j, 0), (j, 700), (255, 0, 0), 2)

for i in range(0, 5):
    if i == 0:
        g = 20
    if i > 0:
        g = 20 + i*16
    cv2.line(mask, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 122
    if i > 0:
        g = 125 + i*16
    cv2.line(mask, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 225
    if i > 0:
        g = 230 + i*16
    cv2.line(mask, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 328
    if i > 0:
        g = 332 + i*16
    cv2.line(mask, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 8):
    j = i*100
    #cv2.line(img, (0, j), (400, j), (255, 0, 0), 2)
    d = j + 25 + i
    cv2.line(mask, (0, d), (400, d), (255, 0, 0), 1)
    if i < 3:
        f = d + 20
        cv2.line(mask, (0, f), (400, f), (255, 0, 0), 1)
    if i >= 3:
        f = d + 18
        cv2.line(mask, (0, f), (400, f), (255, 0, 0), 1)
    c = f + 15
    cv2.line(mask, (0, c), (400, c), (255, 0, 0), 1)
    z = c + 16
    cv2.line(mask, (0, z), (400, z), (255, 0, 0), 1)
    e = z + 15 + 3
    cv2.line(mask, (0, e), (400, e), (255, 0, 0), 1)
    #cv2.line(img, (0, x), (400, x), (255, 0, 0), 1)
    u = z + 29
    cv2.line(mask, (0, u), (400, u), (255, 0, 0), 1)
cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
#docCnt = None
print(len(cnts))
i = 0
g = 0
if len(cnts) > 0:
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)# 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)                                      # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.1*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            g = g+1

      #      print(1)
            if i == 1:
                docCnt = approx
            if i == 2:
                docCnt2 = approx
            #break
        #if len(approx) == 4:
            #cv2.drawContours(img, c, -1, (0, 0, 255), cv2.LINE_8, 10)
print("的方法：", g)
img3 = mask[0:100, 0:100]
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', mask)

cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.imshow('img1', img3)
cv2.waitKey(0)
