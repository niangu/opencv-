import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('/home/niangu/桌面/答题卡识别/1.jpeg', 0)




#CLAHE有限对比适应性直方图均衡化
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)

#cv.imshow("gaussian", gaussian_bulr)
edged1 = cv.Canny(cl1, 50, 100)


cv.imwrite("Canny边缘检测.png",edged1)

#cv.namedWindow('Canny边缘检测', cv.WINDOW_NORMAL)
cv.namedWindow('原图', cv.WINDOW_NORMAL)
cv.namedWindow('灰度图', cv.WINDOW_NORMAL)
#cv.imshow("Canny边缘检测", edged)
cv.imshow("原图", img)
cv.imshow("灰度图", edged1)


cv.waitKey(0)
