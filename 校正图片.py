import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
img = cv2.imread('/home/niangu/文档/webwxgetmsgimg.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 0, 110, 3)            # 边缘检测

a, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
#docCnt = None
print(cnts)
i = 0
is_approx = False
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)# 根据轮廓面积从大到小排序
    for c in cnts:
        i += 1
        peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.1*peri, True)                       #轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            print("")

            if i == 1:
                
                # 轮廓的性质
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                print("边界矩形的宽高比：", aspect_ratio)
                if aspect_ratio > 0.8 and aspect_ratio <=1:#0.9
                   
                    docCnt = approx
                #if aspect_ratio > 1.3 and aspect_ratio < 1.8:
                    #docCnt2 = approx

            if i == 2:
                
                # 轮廓的性质
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                print("边界矩形的宽高比2：", aspect_ratio)
                if aspect_ratio > 1.3 and aspect_ratio < 1.8:#1.8
                    docCnt2 = approx
                    is_approx = True
                    
            if i == 2:
                #if is_approx == False:
                    docCnt2 = approx


            #break
        #if len(approx) == 4:
            #cv2.drawContours(img, c, -1, (0, 0, 255), cv2.LINE_8, 10)

a = []
a2 = []
for peak in docCnt:
    peak = peak[0]
    #cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    a.append(peak)

for peak2 in docCnt2:
    peak2 = peak2[0]
    #cv2.circle(img, tuple(peak2), 10, (255, 0, 0))
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
src = np.float32([c, b, d, e])
print(src)
dst = np.float32([[0, 0], [400, 0], [0, 700], [400, 700]])

m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, m, (400, 700))

src2 = np.float32([j, h, k, l])
print(src2)
dst2 = np.float32([[0, 0], [500, 0], [0, 700], [500, 700]])
m2 = cv2.getPerspectiveTransform(src2, dst2)
result2 = cv2.warpPerspective(img, m2, (500, 700))

#cv2.imwrite("aaaaa.png", result)
cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
cv2.imshow('result2', result2)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
cv2.imwrite("aaaaa是否缺考.png", result)
cv2.imwrite("aaaaa试卷类型.png", result2)

cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
cv2.imshow('edged', edged)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)
