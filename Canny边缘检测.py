import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)
edges = cv2.Canny(img, 100, 200)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', edges)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()
"""
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()
"""