# -*- coding:utf-8 -*-
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
from PIL import Image
import pytesseract

# 加载一个图片到opencv中
img = cv.imread('/home/niangu/桌面/opencv-python/canny.jpg')
# 先对图片进行透视变换
# 转化成灰度图片
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cv.imshow("gray",gray)

gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊

# cv.imshow("gaussian",gaussian_bulr)

edged = cv.Canny(gaussian_bulr, 50, 150)  # 边缘检测,灰度值小于2参这个值的会被丢弃，大于3参这个值会被当成边缘，在中间的部分，自动检测
"""
lines = cv.HoughLines(edged, 1, np.pi / 180, 100)
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

# cv.imshow("edged",img)
"""
# 寻找轮廓
cts, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.imshow('img',image)
# 给轮廓加标记，便于我们在原图里面观察，注意必须是原图才能画出红色，灰度图是没有颜色的
cv.drawContours(img, cts, -1, (0, 0, 255), 3)

# 按面积大小对所有的轮廓排序
list = sorted(cts, key=cv.contourArea, reverse=True)

# for c in list:
# 周长，第1个参数是轮廓，第二个参数代表是否是闭环的图形

c = list[0]
peri = 0.01 * cv.arcLength(c, True)
# 获取多边形的所有定点，如果是四个定点，就代表是矩形
approx = cv.approxPolyDP(c, peri, True)
# print(approx)
# 打印定点个数
print("顶点个数：", len(approx))

if len(approx) == 4:  # 矩形
    # 透视变换提取原图内容部分
    ox_sheet = four_point_transform(img, approx.reshape(4, 2))
    # 透视变换提取灰度图内容部分
    tx_sheet = four_point_transform(gray, approx.reshape(4, 2))
    # 使用ostu二值化算法对灰度图做一个二值化处理
    ret, thresh2 = cv.threshold(tx_sheet, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#  cv.imshow("thresh2", thresh2)

sp = thresh2.shape
sz1 = sp[0]  # height(rows) of image
sz2 = sp[1]  # width(colums) of image

# testImg = thresh2[sz1 // 35:sz1 // 7, sz2 // 4:2 * (sz2 // 4)]
testImg = thresh2[(sz1 // 35) + 16:sz1 // 7, 0: (sz2 // 4)]
cv.imshow('testimg', testImg)

"""

# testImg膨胀
kerX=0
kerY=0
while (1>0):
     if kerX<15:
        kerX=kerX+1


     kernelImg = np.ones((kerX,kerY),np.uint8) 
     dilationImg = cv.dilate(testImg,kernelImg,iterations=1)
     #cv.imshow('dilationImg',dilationImg) 
     cnts = cv.findContours(dilationImg.copy(), cv.RETR_EXTERNAL,
     cv.CHAIN_APPROX_SIMPLE)
     cnts = cnts[0] if imutils.is_cv2() else cnts[1]

     questionCnts = []
# 对每一个轮廓进行循环处理
     for c in cnts:
    # 计算轮廓的边界框，然后利用边界框数据计算宽高比
         (x, y, w, h) = cv.boundingRect(c)
         ar = w / float(h)
    # 为了辨别一个轮廓是一个气泡，要求它的边界框不能太小，在这里边至少是20个像素，而且它的宽高比要近似于1
    #if  h >= 3 :
         questionCnts.append(c)

# 以从顶部到底部的方法将我们的气泡轮廓进行排序，然后初始化正确答案数的变量。
     questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]
     print('len(questionCnts)')
     print(len(questionCnts))
     if len(questionCnts)==20:
        print('找到了膨胀点数')
        break
     if kerX==15:
        kerY=kerY+1
        if kerY==15:
           break


cv.imshow('dilationImg',dilationImg) 
 """

# testImg1=testImg[0:sz22,0:sz21//5]
# cv.imshow('testImg1', testImg1)

# 二值化后图片先腐蚀后膨胀
kernel = np.ones((5, 5), np.uint8)

erosion = cv.erode(testImg, kernel, iterations=1)
opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel)
# cv.imshow('opening',opening)
# 滤波
img = cv.medianBlur(opening, 5)
# cv.imshow("3",img)

# 直接膨胀
dilationT = cv.dilate(testImg, kernel, iterations=1)
cv.imshow("dilationT", dilationT)

spTwo = dilationT.shape
sz22 = spTwo[0]  # height(rows) of image
sz21 = spTwo[1]  # width(colums) of image

testImg1 = dilationT[0:sz22, 0:sz21 // 5]

kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(testImg1, kernel, iterations=1)
cv.imshow('erosion', erosion)

# 膨胀
dilation = cv.dilate(img, kernel, iterations=1)
# cv.imshow("dilation",dilation)

# 在二值图像中查找轮廓，然后初始化题目对应的轮廓列表
cnts, hierarchy = cv.findContours(erosion.copy(), cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
# cv.imshow('dilation',dilation)

cnts = cnts[0] if imutils.is_cv4() else cnts[1]
questionCnts = []


print("打印答题卡的轮廓个数：", len(cnts))#DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD


# 对每一个轮廓进行循环处理
for c in cnts:
    #   print(cv.boundingRect(cnts))
    # 计算轮廓的边界框，然后利用边界框数据计算宽高比
    h = cv.boundingRect(cnts)[0] if imutils.is_cv4() else cv.boundingRect(cnts)[1]
    # ar = w / float(h)
    # # 为了辨别一个轮廓是一个气泡，要求它的边界框不能太小，在这里边至少是20个像素，而且它的宽高比要近似于1
    if h >= 3:
        questionCnts.append(c)

# 以从顶部到底部的方法将我们的气泡轮廓进行排序，然后初始化正确答案数的变量。
#questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
#print('questionCnts')
#print(len(questionCnts))

i = 0
answer = []
for (q, i) in enumerate(np.arange(0, len(questionCnts), 1)):
    # 从左到右为当前题目的气泡轮廓排序，然后初始化被涂画的气泡变量

    cnts = contours.sort_contours(questionCnts[i:i + 1])[0] if imutils.is_cv4() else \
        contours.sort_contours(questionCnts[i:i + 1])[1]
    bubble_rows = []
    # 对一行从左到右排列好的气泡轮廓进行遍历
    for (j, c) in enumerate(cnts):
        i = i + 1
        # 构造只有当前气泡轮廓区域的掩模图像
        #  mask = np.zeros(dilation.shape[1], dtype="uint8")
        mask = np.zeros(erosion.shape[:2], dtype="uint8")
        ctr = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
        cv.drawContours(mask, ctr, 0, (0, 255, 0), -1)
        # 对二值图像应用掩模图像，然后就可以计算气泡区域内的非零像素点。
        mask = cv.bitwise_and(erosion, erosion, mask=mask)
        total = cv.countNonZero(mask)
        # print(total)
        # 如果像素点数最大，就连同气泡选项序号一起记录下来
        bubble_rows.append((total, j))
        print('bubble')
        print(bubble_rows)
        bubble_rows = sorted(bubble_rows, key=lambda x: x[0], reverse=True)
        # 选择的答案序号
        choice_num = bubble_rows[0][0]
        if choice_num > 600:
            if i == 2 or i == 1:
                answer.append('A')
            elif i == 5 or i == 4:
                answer.append('B')
            elif i == 8 or i == 7:
                answer.append('C')
            elif i == 10 or i == 9 or i == 11:
                answer.append('D')

print(answer)

# -----------------------------------------------------------------------
# 感兴趣区域ROI
# ROI = np.zeros(testImg1.shape, np.uint8)
# 提取轮廓
# proimage2=cv.adaptiveThreshold(testImg1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,7,7)
# proimage3,contours,hierarchy=cv.findContours(proimage2,cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE) #提取所有的轮廓
# ROI提取
# cv.drawContours(ROI, contours, 1,(255,255,255),-1)       #ROI区域填充白色，轮廓ID1


cv.waitKey()