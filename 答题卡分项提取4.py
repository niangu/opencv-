import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
img3 = cv2.imread('/home/niangu/桌面/答题卡识别/a答题卡提取2.png')
img2 = cv2.imread('/home/niangu/桌面/答题卡识别/a答题卡分项提取3.png')
img4 = img3[0:65, 0:18]
img = img2[0:65, 0:18]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
#docCnt = None
j = 0
print("总数：", len(cnts))
g= 0
if len(cnts) > 0:
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)# 根据轮廓面积从大到小排序
    for c in cnts:

        peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.01*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            g = g+1
            if j == 1:
                docCnt = approx
            if j == 2:
                docCnt2 = approx
            #break
        #if len(approx) == 4:
            #cv2.drawContours(img, c, -1, (0, 0, 255), cv2.LINE_8, 10)
print("的峨峨：", g)
a = []
a2 = []
#r = img2[0:65, 0:17]
#cv2.namedWindow('F', cv2.WINDOW_NORMAL)
#cv2.imshow('F', r)
for i in range(0, 5):
    j = j+1
    if i==0:
        r1 = img2[0:65, i*17:j*17]
        cv2.namedWindow('F1', cv2.WINDOW_NORMAL)
        cv2.imshow('F1', r1)
    if i==1:
        r2 = img2[0:65, i * 17:j*17]
        cv2.namedWindow('F2', cv2.WINDOW_NORMAL)
        cv2.imshow('F2', r2)
    if i==2:
        r3 = img2[0:65, i * 17:j*17]
        cv2.namedWindow('F3', cv2.WINDOW_NORMAL)
        cv2.imshow('F3', r3)
    if i==3:
        r4 = img2[0:65, i * 17:j*17]
        cv2.namedWindow('F4', cv2.WINDOW_NORMAL)
        cv2.imshow('F4', r4)
    if i==4:
        r5 = img2[0:65, i * 17:j * 17]
        cv2.namedWindow('F5', cv2.WINDOW_NORMAL)
        cv2.imshow('F5', r5)

for i in range(0, 4):
    j = i+1
    if i ==0:
        F = img4[i*16:j*16, 0:17]
        cv2.namedWindow('F', cv2.WINDOW_NORMAL)
        cv2.imshow('F', F)
        print("F")


for i in range(0, 4):
    j = i+1
    if i ==0:
        A = img[i*16:j*16, 0:17]
        cv2.namedWindow('AB', cv2.WINDOW_NORMAL)
        cv2.imshow('AB', A)
        #print("A")
        dif = cv2.subtract(A, F)
        result = not np.any(dif)#if difference is all zeros it will return False
        if result is True:
            print("未选A")
        else:
            print("选则了A")

    if i ==1:
        B = img[i*16:j*16, 0:17]
        cv2.namedWindow('B', cv2.WINDOW_NORMAL)
        cv2.imshow('B', B)
        #print("B")
        dif = cv2.subtract(B, F)
        result = not np.any(dif)  # if difference is all zeros it will return False
        if result is True:
            print("未选B")
        else:
            print("选则了B")

    if i ==2:
        C = img[i*16:j*16, 0:17]
        cv2.namedWindow('C', cv2.WINDOW_NORMAL)
        cv2.imshow('C', C)
        #print("C")
        dif = cv2.subtract(C, F)
        result = not np.any(dif)  # if difference is all zeros it will return False
        if result is True:
            print("未选C")
        else:
            print("选则了C")
    if i ==3:
        D = img[i*16:j*16, 0:17]
        cv2.namedWindow('D', cv2.WINDOW_NORMAL)
        cv2.imshow('D', D)
        #print("D")
        dif = cv2.subtract(D, F)
        result = not np.any(dif)  # if difference is all zeros it will return False
        if result is True:
            print("未选D")
        else:
            print("选则了D")

"""
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    a.append(peak)

for i in range(0, len(a)):
    if i == 0:
        b = a[0]
    if i==1:
        c = a[1]
    if i==2:
        d = a[2]
    if i == 3:
        e = a[3]


#src = np.float32([c, b, d, e])
src = np.float32([b, e, c, d])
print(src)
dst = np.float32([[0, 0], [400, 0], [0, 700], [400, 700]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(img, m, (400, 700))



#cv2.imwrite('答案提取.png', result)

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
"""


#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#cv2.imshow('img', img)
#cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
#cv2.imshow('img1', img4)
cv2.waitKey(0)
