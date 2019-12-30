import cv2
import numpy as np

img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)#腐蚀
#膨胀
#膨胀
dilation = cv2.dilate(img, kernel, iterations=1)
#开运算
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#闭运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#形态学梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#礼帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#黑帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

"""
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
cv2.namedWindow('image4', cv2.WINDOW_NORMAL)
cv2.namedWindow('image5', cv2.WINDOW_NORMAL)
cv2.namedWindow('image6', cv2.WINDOW_NORMAL)
cv2.imshow('image', erosion)#腐蚀
cv2.imshow('image1', dilation)#膨胀
cv2.imshow('image2', opening)#开运算
cv2.imshow('image3', closing)#闭运算
cv2.imshow('image4', gradient)#形态学梯度
cv2.imshow('image5', tophat)#礼帽
cv2.imshow('image6', blackhat)#黑帽
"""


#cv2.imshow('image', img)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()