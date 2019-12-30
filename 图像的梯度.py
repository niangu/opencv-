import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)


cv2.namedWindow('laplacian', cv2.WINDOW_NORMAL)
cv2.namedWindow('sobelx', cv2.WINDOW_NORMAL)
cv2.namedWindow('sobely', cv2.WINDOW_NORMAL)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

cv2.imshow('laplacian', laplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)


cv2.imshow('img', img)
cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()
"""
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.show()
"""