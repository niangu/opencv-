#图像平滑
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img, -1, kernel)

cv2.namedWindow('adaptive', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive', dst)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()



"""
plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([])
plt.yticks([])
plt.show()
"""