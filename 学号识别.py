import cv2
import imutils
import numpy as np
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/A10.png')
#for i in range(1 ,10):
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0.4)#调0.4
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 50, 250, 3)#调100， 250            # 边缘检测

mask = np.zeros(edged.shape, np.uint8)
img2 = img[166:690, 0:500]
edged2 =edged[120:500, 0:700]
for j in range(0, 12):
    if j==1:
        A = img2[0:400, 5+j*57:38+j*57]
        Amask = mask[0:400, 5+j*57:38+j*57]
        edged3 = edged2[0:400, 5+j*57:38+j*57]
        for i in range(0, 9):
            B = A[11+i*39:34+i*39, 0:33]
            Bmask = Amask[11+i*39:34+i*39, 0:33]
            edged4 = edged3[11+i*39:34+i*39, 0:33]
            dif = cv2.subtract(edged4, Bmask)
            result = not np.any(dif)  # if difference is all zeros it will return False
            if result is True:
                pass
            else:
                print(i)

    else:
        A = img2[0:400, 5 + j * 59:38 + j * 59]
        Amask = mask[0:400, 5 + j * 59:38 + j * 59]
        edged3 = edged2[0:400, 5 + j * 59:38 + j * 59]
        for i in range(0, 9):
            B = A[11 + (i * 39):34 + (i * 39), 0:33]
            Bmask = Amask[11 + (i * 39):34 + (i * 39), 0:33]
            edged4 = edged3[11 + (i * 39):34 + (i * 39), 0:33]
            dif = cv2.subtract(edged4, Bmask)

            dif = cv2.subtract(edged4, Bmask)
            result = not np.any(dif)  # if difference is all zeros it will return False
            if result is True:
                pass
            else:
                print(i)

cv2.namedWindow('D1', cv2.WINDOW_NORMAL)
cv2.imshow('D1', edged)
cv2.namedWindow('D1A', cv2.WINDOW_NORMAL)
cv2.imshow('D1A', img2)

cv2.waitKey(0)
































