import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/试卷类型匹配.jpeg')
img = cv2.imread('/home/niangu/桌面/答题卡识别/a答题卡特征提取检测23.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 120, 150, 3)            # 边缘检测


img3 = edged[0:700, 394:583]
imgA = img3[80:200]
maskA = np.zeros(imgA.shape, np.uint8)
imgB = img3[540:630]
maskB = np.zeros(imgB.shape, np.uint8)
dif = cv2.subtract(imgA, maskA)
result = not np.any(dif)  # if difference is all zeros it will return False
if result is True:
    pass
else:
    print("A")

dif = cv2.subtract(imgB, maskB)
result = not np.any(dif)  # if difference is all zeros it will return False
if result is True:
    pass
else:
    print("B")




cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', imgA)
cv2.namedWindow('img21', cv2.WINDOW_NORMAL)
cv2.imshow('img21', imgB)

cv2.namedWindow('img211', cv2.WINDOW_NORMAL)
cv2.imshow('img211', edged)

cv2.namedWindow('img211', cv2.WINDOW_NORMAL)
cv2.imshow('img211', img)
cv2.waitKey(0)
































