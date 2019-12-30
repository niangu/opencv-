import cv2
import imutils
import numpy as np

answer_3 = {1: 'B', 2: 'D', 3: 'B', 4: 'A', 5: 'D', 6: 'B', 7: 'A', 8: 'B', 9: 'C', 10: 'A', 11: 'A', 12: 'B', 13: 'C', 14: 'A',
          17: 'B', 19: 'A', 26: 'A', 27: 'C', 28: 'B', 29: 'D', 30: 'B', 31: 'B', 32: 'B', 33: 'A', 34: 'A', 35: 'D', 36: 'B',
          37: 'C', 39: 'C', 46: 'C', 47: 'B', 48: 'C', 49: 'B', 50: 'B', 52: 'C', 53: 'C', 54: 'C', 56: 'C', 57: 'B', 58: 'B',
          59: 'B', 60: 'B', 71: 'A', 73: 'A', 74: 'B', 76: 'C', 81: 'B', 82: 'C', 85: 'A', 92: 'C', 98: 'C', 101: 'C', 102: 'B',
          103: 'A', 104: 'A', 106: 'A', 107: 'B', 108: 'B', 109: 'B', 110: 'A', 111: 'C', 112: 'C', 115: 'D', 116: 'C', 117: 'A',
          118: 'B', 119: 'D', 125: 'D', 129: 'A', 130: 'A', 131: 'B', 132: 'A', 133: 'C', 134: 'D', 135: 'C', 136: 'B', 137: 'B',
          138: 'C', 139: 'B', 140: 'A'}

img2 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/a答题卡特征提取检测24.png')
#for i in range(1 ,10):
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0.4)#调0.4
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 60, 250, 3)#调100， 250            # 边缘检测
#cv2.namedWindow('img3%d'%i, cv2.WINDOW_NORMAL)
#cv2.imshow('img3%d'%i, edged)

cv2.namedWindow('img366', cv2.WINDOW_NORMAL)
cv2.imshow('img366', img2)

mask = np.zeros(edged.shape, np.uint8)
answer = {}
answer_1 = {} #选择了多项
answer_2 = []#没有选择
#img3 = img[30:95, 3:87]
#edged2 = edged[30:95, 3:87]
#mask2 = mask[30:95, 3:87]
compare = mask[0:16, 0:17]

for i in range(0, 7):
    for j in range(0, 4):

        img3 = img2[30 + i * 100+i:95 + i * 100, 3 + j * 104:87 + j * 103]
        edged2 = edged[30 + i * 100+i:95 + i * 100, 3 + j * 104:87 + j * 103]
        mask2 = mask[30 + i * 100+i:95 + i * 100, 3 + j * 104:87 + j * 103]
        """
        if i==3:
            img3 = img2[33 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
            edged2 = edged[3 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
            mask2 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
        if i>3:
            img3 = img2[35 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
            edged2 = edged[3 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
            mask2 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 103]
        """
        #cv2.namedWindow('img%d'%((i*20) + j) ,cv2.WINDOW_NORMAL)
        #cv2.imshow('img%d'%((i*20) + j), img3)

        for h in range(0, 5):
            l = h + 1

            img30 = img3[0:65, 2+h * 17: 13+h*17]
            edged30 = edged2[0:65, 2+h * 17: 13+h*17]
            mask30 = mask2[0:65, 2+h * 17: 13+h*17]
            answer_decide = []
            answer_decide_2 = []
            #cv2.namedWindow('img%d' % ( (i*20) + (5*j) + h+1), cv2.WINDOW_NORMAL)
            #cv2.imshow('img%d' % ((i*20) + (5*j) + h+1), img30)
            for q in range(0, 4):

                w = q + 1
                if q == 0:
                    imgA = img30[0+q*17:10+q*17, 0:11]
                    A = edged30[0+q*17:10+q*17, 0:11]
                    #A = edged30[0:9, 0:11]
                    compareA = mask30[0+q*17:10+q*17, 0:11]
                    #cv2.namedWindow('img%d' % ((i * 80)+(j*20)+(5*h)+q), cv2.WINDOW_NORMAL)
                    #cv2.imshow('img%d' % ((i * 80)+(j*20)+(5*h)+q), imgA)
                    dif = cv2.subtract(A, compareA)
                    result = not np.any(dif)  # if difference is all zeros it will return False
                    if result is True:
                        pass
                    else:
                        answer_decide.append("A")
                        cv2.rectangle(img30, (0, 0+q*17), (11, 10+q*17), (0, 255, 0), 1)
                        answer_decide_2.append(A)

                        #cv2.namedWindow('img%d' % ((i * 20) + (5 * j) + h + 1), cv2.WINDOW_NORMAL)
                        #cv2.imshow('img%d' % ((i * 20) + (5 * j) + h + 1), img30)

                if q == 1:
                    B = edged30[0 + q * 17:10 + q * 17, 0:11]
                    #B = edged30[18:26, 0:11]
                    compareB = mask30[0+q*17:10+q*17, 0:11]
                    #cv2.namedWindow('img%d' % ((i * 80)+(j*20)+(5*h)+q ), cv2.WINDOW_NORMAL)
                    #cv2.imshow('img%d' % ((i * 80)+(j*20)+(5*h)+q ), B)
                    dif = cv2.subtract(B, compareB)
                    result = not np.any(dif)  # if difference is all zeros it will return False
                    if result is True:
                        pass
                    else:
                        answer_decide.append("B")
                        cv2.rectangle(img30, (0, 0 + q * 17), (11, 10 + q * 17), (0, 255, 0), 1)
                        answer_decide_2.append(B)
                if q == 2:
                    C = edged30[0 + q * 17:10 + q * 17, 0:11]
                    #C = edged30[38:43, 0:11]
                    compareC = mask30[0+q*17:10+q*17, 0:11]
                    #cv2.namedWindow('img%d' % ((i * 80)+(j*20)+(5*h)+q ), cv2.WINDOW_NORMAL)
                    #cv2.imshow('img%d' % ((i * 80)+(j*20)+(5*h)+q ), C)
                    dif = cv2.subtract(C, compareC)
                    result = not np.any(dif)  # if difference is all zeros it will return False
                    if result is True:
                        pass
                    else:
                        answer_decide.append("C")
                        cv2.rectangle(img30, (0, 0 + q * 17), (11, 10 + q * 17), (0, 255, 0), 1)
                        answer_decide_2.append(C)
                if q == 3:
                    D = edged30[0 + q * 17:10 + q * 17, 0:11]
                    #D = edged30[55:60, 0:11]
                    compareD = mask30[0+q*17:10+q*17, 0:11]
                    #cv2.namedWindow('img%d' % ((i * 80)+(j*20)+(5*h)+q), cv2.WINDOW_NORMAL)
                    #cv2.imshow('img%d' % ((i * 80)+(j*20)+(5*h)+q), D)
                    dif = cv2.subtract(D, compareD)
                    result = not np.any(dif)  # if difference is all zeros it will return False
                    if result is True:
                        pass
                    else:
                        answer_decide.append("D")
                        cv2.rectangle(img30, (0, 0 + q * 17), (11, 10 + q * 17), (0, 255, 0), 1)
                        answer_decide_2.append(D)

                    # 最后判断
                    count = (i*20) + (5*j) + h+1
                    if len(answer_decide) == 0:
                        answer_2.append(count)
                    if len(answer_decide) == 1:
                        answer[count] = answer_decide[0]

                        #if answer_3[count] == answer_decide[0]:
                         #   if answer_decide[0] == "A":

                          #      pass

                        """
                            if answer_decide[0] == "B":
                                cv2.rectangle(A, (2, 21), (11, 26), (0, 255, 0), 2)
                            if answer_decide[0] == "C":
                                cv2.rectangle(A, (2, 38), (11, 43), (0, 255, 0), 2)
                            if answer_decide[0] == "D":
                                cv2.rectangle(A, (2, 55), (11, 60), (0, 255, 0), 2)
                            
                            """
                        #print(count, ":\n", answer_decide_2[0])
                    if len(answer_decide) > 1:
                        answer_1[count] = answer_decide

                    if len(answer_decide) > 1:
                        answer_1[count] = answer_decide
            cv2.namedWindow('img%d' % ((i * 20) + (5 * j) + h + 1), cv2.WINDOW_NORMAL)
            cv2.imshow('img%d' % ((i * 20) + (5 * j) + h + 1), img30)
print(answer)
print(answer_1)
#print(len(answer_1))
print(answer_2)
print(len(answer_2))
#print(len(answer))
#cv2.namedWindow('img366', cv2.WINDOW_NORMAL)
#cv2.imshow('img366', edged)

#cv2.namedWindow('img31', cv2.WINDOW_NORMAL)
#cv2.imshow('img31', img)
cv2.waitKey(0)