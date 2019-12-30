# -*- coding:utf-8 -*-
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2 as cv
from PIL import Image
import pytesseract

# 加载一个图片到opencv中
img = cv.imread('/home/niangu/桌面/答题卡识别/webwxgetmsgimg (17).jpeg')
# 先对图片进行透视变换
# 转化成灰度图片
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# cv.imshow("gray",gray)

gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊

# cv.imshow("gaussian",gaussian_bulr)

dged = cv.Canny(gaussian_bulr, 50, 150)  # 边缘检测,灰度值小于2参这个值的会被丢弃，大于3参这个值会被当成边缘，在中间的部分，自动检测
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
cv.namedWindow('approx', cv.WINDOW_NORMAL)

cv.imshow('approx', gray)
# 寻找轮廓
cts, hierarchy = cv.findContours(gray.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    cv.imshow("thresh2", thresh2)

sp = thresh2.shape
sz1 = sp[0]  # height(rows) of image
sz2 = sp[1]  # width(colums) of image

# testImg = thresh2[sz1 // 35:sz1 // 7, sz2 // 4:2 * (sz2 // 4)]
testImg = thresh2[(sz1 // 35) + 16:sz1 // 7, 0: (sz2 // 4)]
cv.imshow('testimg', testImg)


cv.waitKey(0)&0xFF
#cv2.destroyAllWindows()