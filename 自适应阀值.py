import cv2
import numpy as np
from matplotlib import pyplot as plt


#选择th2实心，th3空心
img = cv2.imread('/home/niangu/桌面/答题卡识别/4.jpeg',0)
img = cv2.medianBlur(img, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
         'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

"""
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)

cv2.namedWindow('th1', cv2.WINDOW_NORMAL)
cv2.imshow('th1', th1)

cv2.namedWindow('th2', cv2.WINDOW_NORMAL)
cv2.imshow('th2', th2)

cv2.namedWindow('th3', cv2.WINDOW_NORMAL)
cv2.imshow('th3', th3)




cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()