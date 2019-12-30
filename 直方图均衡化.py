import cv2
import numpy as np
#opencv中的直方图均衡化
img = cv2.imread('/home/niangu/桌面/Answer-Card-Recognition-master/pic/IMG_20170510_155426.jpg', 0)

equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))

#CLAHE有限对比适应性直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

"""
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
color = ('b', 'g', 'r')
for i,col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()
#kernel = np.ones((5,5), np.uint8)
#equ = cv.equalizeHist(img)
#res = np.hstack((img, equ))

#CLAHE有限对比适应性直方图均衡化
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#cl1 = clahe.apply(img)
#opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
"""
cv2.namedWindow('直方图均衡化', cv2.WINDOW_NORMAL)
cv2.namedWindow('CLAHE直方图均衡化', cv2.WINDOW_NORMAL)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('epu', cv2.WINDOW_NORMAL)

cv2.imshow('epu', equ)
cv2.imshow('image', img)
cv2.imshow('直方图均衡化', res)
cv2.imshow('CLAHE直方图均衡化', cl1)

cv2.waitKey(0)&0xFF