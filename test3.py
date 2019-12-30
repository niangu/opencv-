import cv2
import numpy as np
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (30).jpeg')
#img2 = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg',0)
#CLAHE有限对比适应性直方图均衡化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)#自适应阀值
#直方图均衡化
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
#th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#归一化
dst2 = np.zeros_like(cl1)
cv2.normalize(cl1, dst2, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#中值模糊
median = cv2.medianBlur(dst2, 5)
kernel = np.ones((4, 4), np.uint8)
#开运算
opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
#闭运算
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

erosion = cv2.erode(closing, kernel, iterations=1)#腐蚀
#edges = cv2.Canny(cl1, 100, 200)#Canny边缘检测
#ret1,th1 = cv2.threshold(erosion, 150, 200, cv2.THRESH_BINARY)#二值化
edges = cv2.Canny(cl1, 100, 200)







#edges = cv2.Canny(th1, 100, 200)#Canny边缘检测


#cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
#cv2.imshow('erosion', th1)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', edges)
#显示图像

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()
