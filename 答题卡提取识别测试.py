import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (30).jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
#docCnt = None
i = 0
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)# 根据轮廓面积从大到小排序
    for c in cnts:
        i += 1
        peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.01*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            if i == 1:
                docCnt = approx
            if i == 2:
                docCnt2 = approx
            #break
        #if len(approx) == 4:
            #cv2.drawContours(img, c, -1, (0, 0, 255), cv2.LINE_8, 10)

a = []
a2 = []
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    a.append(peak)

for peak2 in docCnt2:
    peak2 = peak2[0]
    cv2.circle(img, tuple(peak2), 10, (255, 0, 0))
    a2.append(peak2)

for i in range(0, len(a)):
    if i == 0:
        b = a[0]
    if i==1:
        c = a[1]
    if i==2:
        d = a[2]
    if i == 3:
        e = a[3]

for i in range(0, len(a2)):
    if i == 0:
        h = a2[0]
    if i==1:
        j = a2[1]
    if i==2:
        k = a2[2]
    if i == 3:
        l = a2[3]

#src = np.float32([c, b, d, e])
src = np.float32([b, e, c, d])
print(src)
dst = np.float32([[0, 0], [400, 0], [0, 700], [400, 700]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, m, (400, 700))

src2 = np.float32([h, l, j, k])
print(src2)
dst2 = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m2 = cv2.getPerspectiveTransform(src2, dst2)
result2 = cv2.warpPerspective(img, m2, (337, 488))

#第二部分
clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
dilate2 = cv2.dilate(blurred2, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged2 = cv2.Canny(dilate2, 30, 120, 3)            # 边缘检测
cv2.line(edged2, (0, 7), (400, 7), (255, 0, 0), 1)
for i in range(0, 5):
    j = i*100
    cv2.line(edged2, (j, 0), (j, 700), (255, 0, 0), 2)

for i in range(0, 5):
    if i == 0:
        g = 20
    if i > 0:
        g = 20 + i*16
    cv2.line(edged2, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 122
    if i > 0:
        g = 125 + i*16
    cv2.line(edged2, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 225
    if i > 0:
        g = 230 + i*16
    cv2.line(edged2, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 5):
    if i == 0:
        g = 328
    if i > 0:
        g = 332 + i*16
    cv2.line(edged2, (g, 0), (g, 700), (255, 0, 0), 1)

for i in range(0, 8):
    j = i*100
    #cv2.line(img, (0, j), (400, j), (255, 0, 0), 2)
    d = j + 25 + i
    cv2.line(edged2, (0, d), (400, d), (255, 0, 0), 1)
    if i < 3:
        f = d + 20
        cv2.line(edged2, (0, f), (400, f), (255, 0, 0), 1)
    if i >= 3:
        f = d + 18
        cv2.line(edged2, (0, f), (400, f), (255, 0, 0), 1)
    c = f + 15
    cv2.line(edged2, (0, c), (400, c), (255, 0, 0), 1)
    z = c + 16
    cv2.line(edged2, (0, z), (400, z), (255, 0, 0), 1)
    e = z + 15 + 3
    cv2.line(edged2, (0, e), (400, e), (255, 0, 0), 1)
    #cv2.line(img, (0, x), (400, x), (255, 0, 0), 1)
    u = z + 29
    cv2.line(edged2, (0, u), (400, u), (255, 0, 0), 1)

"""
for i in range(0, 4):
    f = 125 + (i*15)
    cv2.line(img, (0, f), (400, f), (255, 0, 0), 1)
"""

#cv2.imwrite('答题卡提取答案.png', edged)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', edged2)
#cv2.namedWindow('img21', cv2.WINDOW_NORMAL)
#cv2.imshow('img21', img)



#cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
#cv2.imshow('result2', result2)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)
#15034941755