import cv2
import imutils
import numpy as np
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/aaa2.png')
#for i in range(1 ,10):
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0.4)#调0.4
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 60, 250, 3)#调100， 250            # 边缘检测

mask = np.zeros(edged.shape, np.uint8)
img2 = img[47:200, 0:200]
edged2 =edged[47:200, 0:200]
for i in range(0, 12):
    if i<=5:
        B = img2[0:153, 3+i*16+i:13+16*i+i]
        Amask = mask[0:153, 3+i*16+i:13+16*i+i]
        edged3 = edged2[0:153, 3+i*16+i:13+16*i+i]
        #cv2.namedWindow('D%d'%(i), cv2.WINDOW_NORMAL)
        #cv2.imshow('D%d'%(i), B)
    if i>5 and i!=11:
        B = img2[0:153, 3 + i * 16 +i-1:13 + 16 * i + i-2]
        Amask = mask[0:153, 3 + i * 16 +i-1:13 + 16 * i + i-2]
        edged3 = edged2[0:153, 3 + i * 16 +i-1:13 + 16 * i + i-2]
        #cv2.namedWindow('D%d' % (i), cv2.WINDOW_NORMAL)
        #cv2.imshow('D%d' % (i), B)

    if i == 11:
        B = img2[0:153, 186+1:198]
        Amask = mask[0:153, 186+1:198]
        edged3 = edged2[0:153, 186+1:198]
        cv2.namedWindow('D%d' % (i), cv2.WINDOW_NORMAL)
        cv2.imshow('D%d' % (i), B)

    for j in range(0, 10):
        if j <=5:
            C = B[3 + j * 15:11 + j * 15, 0:9]
            Cmask = Amask[3 + j * 15:11 + j * 15, 0:9]
            edged4 = edged3[3 + j * 15:11 + j * 15, 0:9]
            # cv2.namedWindow('D%d%d' % (i, j), cv2.WINDOW_NORMAL)
            # cv2.imshow('D%d%d' % (i, j), C)
            dif = cv2.subtract(edged4, Cmask)
            result = not np.any(dif)  # if difference is all zeros it will return False
            if result is True:
                pass
            else:
                cv2.rectangle(B, (0, 3+j*15), (9, 11+j*15), (0, 255, 0), 2)
                if i == 0:
                    print(j)
                else:
                    print(j)

        if j>5:
            C = B[3 + j * 15:11 + j * 15, 0:9]
            Cmask = Amask[3 + j * 15:11 + j * 15, 0:9]
            edged4 = edged3[3 + j * 15:11 + j * 15, 0:9]
            # cv2.namedWindow('D%d%d' % (i, j), cv2.WINDOW_NORMAL)
            # cv2.imshow('D%d%d' % (i, j), C)
            dif = cv2.subtract(edged4, Cmask)
            result = not np.any(dif)  # if difference is all zeros it will return False
            if result is True:
                pass
            else:
                cv2.rectangle(B, (0, 3 + j * 15), (9, 11 + j * 15), (0, 255, 0), 2)
                if i == 0:
                    print(j)
                else:
                    print(j)

        # cv2.namedWindow('D%d%d' % (i,j), cv2.WINDOW_NORMAL)
        # cv2.imshow('D%d%d' % (i,j), C)


cv2.namedWindow('D1', cv2.WINDOW_NORMAL)
cv2.imshow('D1', edged)
cv2.namedWindow('D1A', cv2.WINDOW_NORMAL)
cv2.imshow('D1A', img)
cv2.namedWindow('D1A1', cv2.WINDOW_NORMAL)
cv2.imshow('D1A1', img2)
cv2.waitKey(0)
