#Otsu's二值化
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)
ret1,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
images = [img, 0, th1,
         img, 0, th2,
         blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global', 'Thresholding(v=127)',
         'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
         'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

cv2.namedWindow('th1', cv2.WINDOW_NORMAL)
cv2.imshow('th1', th1)

cv2.namedWindow('th2', cv2.WINDOW_NORMAL)
cv2.imshow('th2', th2)

cv2.namedWindow('th3', cv2.WINDOW_NORMAL)
cv2.imshow('th3', th3)




cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()
"""
for i in range(3):
    plt.subplot(3,3,i*3+1)
    plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,3,i*3+2)
    plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,3,i*3+3)
    plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2])
    plt.xticks([])
    plt.yticks([])
    plt.show()
"""