#图像模糊(图像平滑)
#平均
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)

#2D卷积
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
#图像模糊：平均
blur2 = cv2.blur(img,(5,5))
#高斯模糊
blur3 = cv2.GaussianBlur(img, (5, 5), 0)
#中值模糊
median = cv2.medianBlur(img, 5)
##双边滤波
blur4 = cv2.bilateralFilter(img,9,75,75)

cv2.namedWindow('adaptive', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive', blur2)

cv2.namedWindow('adaptive1', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive1', dst)

cv2.namedWindow('adaptive2', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive2', blur3)

cv2.namedWindow('adaptive3', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive3', median)

cv2.namedWindow('adaptive4', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive4', blur4)




cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()

"""
plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(blur)
plt.title('Blurred')
plt.xticks([])
plt.yticks([])
plt.show()
"""