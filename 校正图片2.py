import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
img2 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged2 = cv2.Canny(dilate, 0, 250, 3)            # 边缘检测
a, cnts2,_ = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
img2 = cv2.drawContours(img2, cnts2, -1, (255, 0, 0), )
edged = cv2.Canny(img2, 60, 200, 1)            # 边缘检测
a, cnts,_ = cv2.findContours(edged2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cv2.imwrite("轮廓提取实验2.png", edged)
img = cv2.imread("轮廓提取实验2.png")
i =0
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)# 根据轮廓面积从大到小排序
    for c in cnts:
        i += 1
        peri = cv2.arcLength(c, True)                                       # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.01*peri, True)                       #轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            if i == 5:
                """
                # 轮廓的性质
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                print("边界矩形的宽高比：", aspect_ratio)
                if aspect_ratio > 0.8 and aspect_ratio <=1:#0.9
                   """
                docCnt = approx
                print(docCnt)
                #if aspect_ratio > 1.3 and aspect_ratio < 1.8:
                    #docCnt2 = approx

            if i == 6:
                """
                # 轮廓的性质
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                print("边界矩形的宽高比2：", aspect_ratio)
                if aspect_ratio > 1.3 and aspect_ratio < 1.8:#1.8
                    docCnt2 = approx
                    is_approx = True
                    """
            if i == 2:
                #if is_approx == False:
                    docCnt2 = approx
                    print(docCnt2)



#print(approx2)
cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
cv2.imshow('edged', edged)

cv2.waitKey(0)