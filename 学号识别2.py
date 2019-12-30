import cv2
import imutils
import numpy as np
img = cv2.imread('/home/niangu/桌面/answer_sheet/foo/aaaaa.png')
#for i in range(1 ,10):
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0.4)#调0.4
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 60, 250, 3)#调100， 250            # 边缘检测

mask = np.zeros(edged.shape, np.uint8)
img2 = img[166:690, 0:500]
edged2 =edged[166:690, 0:500]
for i in range(0, 13):
    B = img2[0:524, 10+i*40+i:35+i*40+i]
    Amask = mask[0:524, 10+i*40+i:35+i*40+i]
    edged3 = edged2[0:524, 10+i*40+i:35+i*40+i]
    #cv2.namedWindow('D%d'%(i), cv2.WINDOW_NORMAL)
    #cv2.imshow('D%d'%(i), B)
    for j in range(0, 9):
        C = B[7+j*54:32+j*54, 3:24]
        Bmask = Amask[7+j*54:32+j*54, 3:24]
        edged4 = edged3[7+j*54:32+j*54, 3:24]
        dif = cv2.subtract(edged4, Bmask)
        result = not np.any(dif)  # if difference is all zeros it will return False
        if result is True:
            pass
        else:
            cv2.rectangle(B, (3, 10+j*54), (24, 32+j*54), (0, 255, 0), 2)
            if i == 0:
                print(j)
            else:
                print(j)
        #cv2.namedWindow('D%d%d' % (i,j), cv2.WINDOW_NORMAL)
        #cv2.imshow('D%d%d' % (i,j), C)


#cv2.namedWindow('D1', cv2.WINDOW_NORMAL)
#cv2.imshow('D1', edged)
cv2.namedWindow('D1A', cv2.WINDOW_NORMAL)
cv2.imshow('D1A', img)
cv2.namedWindow('D1A1', cv2.WINDOW_NORMAL)
cv2.imshow('D1A1', edged)
cv2.waitKey(0)

