import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test2/D.jpg')
img = cv2.imread('/home/niangu/桌面/答题卡识别/答案提取.png')
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测
cv2.line(img, (0, 7), (400, 7), (255, 0, 0), 1)
for i in range(0, 5):
    j = i*100
    cv2.line(img, (j, 0), (j, 700), (255, 0, 0), 2)

for i in range(0, 5):
    if i == 0:
        g = 20
    if i > 0:
        g = 20 + i*16
    cv2.line(img, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 122
    if i > 0:
        g = 125 + i*16
    cv2.line(img, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 225
    if i > 0:
        g = 230 + i*16
    cv2.line(img, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 328
    if i > 0:
        g = 332 + i*16
    cv2.line(img, (g, 0), (g, 700), (255, 0, 0), 1)
"""
for i in range(0, 8):
    j = i*100
    #cv2.line(img, (0, j), (400, j), (255, 0, 0), 2)
    d = j + 25 + i
    cv2.line(img, (0, d), (400, d), (255, 0, 0), 1)
    if i < 3:
        f = d + 20
        cv2.line(img, (0, f), (400, f), (255, 0, 0), 1)
    if i >= 3:
        f = d + 18
        cv2.line(img, (0, f), (400, f), (255, 0, 0), 1)
    c = f + 15
    cv2.line(img, (0, c), (400, c), (255, 0, 0), 1)
    z = c + 16
    cv2.line(img, (0, z), (400, z), (255, 0, 0), 1)
    e = z + 15 + 3
    cv2.line(img, (0, e), (400, e), (255, 0, 0), 1)
    #cv2.line(img, (0, x), (400, x), (255, 0, 0), 1)
    u = z + 29
    cv2.line(img, (0, u), (400, u), (255, 0, 0), 1)

"""
"""
for i in range(0, 4):
    f = 125 + (i*15)
    cv2.line(img, (0, f), (400, f), (255, 0, 0), 1)
"""
img2 = img[0:100, 0:100]
#cv2.imwrite('答题卡提取答案.png', edged)
#cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
#cv2.imshow('img2', edged)
cv2.namedWindow('img21', cv2.WINDOW_NORMAL)
cv2.imshow('img21', img2)
cv2.waitKey(0)
