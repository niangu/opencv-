import cv2
import numpy as np
img = cv2.imread('/home/niangu/桌面/Answer-Card-Recognition-master/pic/1.jpg')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

#寻找所有轮廓
contours, cnt = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取所有的轮廓
areas = list()
#for i in enumerate(contours):
    #areas.append(contours[i])
print(len(contours))
a2 = sorted(areas, key=lambda d:d[1], reverse=True)#按面积大小，从大到小排序


print("AAAAA")


cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', opening)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()