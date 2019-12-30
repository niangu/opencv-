import cv2
import numpy as np


# 加载一个图片到opencv中
img = cv2.imread('/home/niangu/桌面/Answer-Card-Recognition-master/pic/1.jpg')

# 先对图片进行透视变换
# 转化成灰度图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("gray",gray)

gaussian_bulr = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊

# cv.imshow("gaussian",gaussian_bulr)

edged = cv2.Canny(gaussian_bulr, 50, 150)  # 边缘检测,灰度值小于2参这个值的会被丢弃，大于3参这个值会被当成边缘，在中间的部分，自动检测

lines = cv2.HoughLines(edged, 1, np.pi / 180, 100)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    ttt = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)







cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', edged)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()