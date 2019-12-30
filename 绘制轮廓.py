import numpy as np
import cv2
import cv2

# 读取图片
img = cv2.imread("/home/niangu/桌面/答题卡识别/Canny边缘检测.png")
# 转灰度图片
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# 轮廓检测
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(contours)

print("轮廓层次:", hierarchy)
# 新打开一个图片，我这里这张图片是一张纯白图片
newImg = cv2.imread("/home/niangu/桌面/答题卡识别/空白.png")
newImg = cv2.resize(newImg, (300,300))




# 画图
cv2.drawContours(newImg, contours, -1, (0,0,0), 3)

cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

# 展示
cv2.imshow("edges", newImg)

cv2.imshow("img", img)
cv2.waitKey(0)

""""
im = cv2.imread('/home/niangu/桌面/Answer-Card-Recognition-master/pic/1.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContour(im, hierarchy, -1, (0, 255, 0), 3)
#img2 = cv2.drawContours(im, contours, 3, (0,255,0), 3)

print(len(contours))
cnt = contours[0]
M = cv2.moments(cnt)
area = cv2.contourArea(cnt)
print("面积：", area)


cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', im)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()
"""