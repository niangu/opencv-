import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test2/D.jpg')
img = cv2.imread('/home/niangu/桌面/答题卡识别/a答题卡分项提取3.png')
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

mask = np.zeros(edged.shape, np.uint8)
for i in range(0, 5):
    #if i == 0:
        if i==0:
            A = img[0:10, 0:17]
            cv2.namedWindow('A', cv2.WINDOW_NORMAL)
            cv2.imshow('A', A)

        if i==1:
            B = img[17:27, 0:17]
            cv2.namedWindow('B', cv2.WINDOW_NORMAL)
            cv2.imshow('B', B)

        if i==2:
            C = img[34:44, 0:17]
            cv2.namedWindow('C', cv2.WINDOW_NORMAL)
            cv2.imshow('C', C)

        if i==3:
            D = img[51:62, 0:17]

            cv2.namedWindow('D', cv2.WINDOW_NORMAL)
            cv2.imshow('D', D)

cv2.namedWindow('D1', cv2.WINDOW_NORMAL)
cv2.imshow('D1', img)

cv2.waitKey(0)