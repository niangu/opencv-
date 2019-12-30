import cv2
import imutils
import numpy as np
img1 = cv2.imread('/home/niangu/桌面/答题卡识别/test2/D.jpg')
img = cv2.imread('/home/niangu/桌面/答题卡识别/答案提取.png')
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

mask = np.zeros(edged.shape, np.uint8)
answer = {}
answer_1 = {} #选择了多项
answer_2 = []#没有选择
#img3 = img[30:95, 3:87]
#edged2 = edged[30:95, 3:87]
#mask2 = mask[30:95, 3:87]
compare = mask[0:16, 0:17]

#img3 = img[30+100:95+100,3:87]
for i in range(0, 8):
    if i == 0:
        """
        img6 = img[30+i*100:95+i*100, 3:87]
        edged6 = edged[30+i*100:95+i*100, 3:87]
        mask6 = mask[30+i*100:95+i*100, 3:87]
        cv2.namedWindow('img6', cv2.WINDOW_NORMAL)
        cv2.imshow('img6', img6)
        """
        for j in range(0, 4):
            if j == 0:
                img2 = img[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                edged2 = edged[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                mask2 = mask[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
                cv2.imshow('img2', img2)
                cv2.imwrite("a答题卡分项提取3.png", img2)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img30 = img2[0:65, h * 17:l * 17]
                        edged30 = edged2[0:65, h * 17:l * 17]
                        mask30 = mask2[0:65, h * 17:l * 17]
                        cv2.namedWindow('img30', cv2.WINDOW_NORMAL)
                        cv2.imshow('img30', img30)
                        #分离答案判断选项
                        answer_decide = []
                        for i in range(0, 4):
                            j = i + 1
                            if i == 0:
                                A = edged30[i * 16:j * 16, 0:17]
                                #cv2.namedWindow('A', cv2.WINDOW_NORMAL)
                                #cv2.imshow('A', A)
                                dif = cv2.subtract(A, compare)
                                result = not np.any(dif)  # if difference is all zeros it will return False
                                if result is True:
                                    #print("未选A")
                                    pass
                                else:
                                    #print("选则了A")
                                    answer_decide.append("A")

                            if i == 1:
                                B = edged30[i * 16:j * 16, 0:17]
                                #cv2.namedWindow('B', cv2.WINDOW_NORMAL)
                                #cv2.imshow('B', B)
                                # print("B")
                                dif = cv2.subtract(B, compare)
                                result = not np.any(dif)  # if difference is all zeros it will return False
                                if result is True:
                                    #print("未选B")
                                    pass
                                else:
                                    #print("选则了B")
                                    answer_decide.append("B")
                            if i == 2:
                                C = edged30[i * 16:j * 16, 0:17]
                                #cv2.namedWindow('C', cv2.WINDOW_NORMAL)
                                #cv2.imshow('C', C)
                                # print("C")
                                dif = cv2.subtract(C, compare)
                                result = not np.any(dif)  # if difference is all zeros it will return False
                                if result is True:
                                    #print("未选C")
                                    pass
                                else:
                                    #print("选则了C")
                                    answer_decide.append("C")
                            if i == 3:
                                D = edged30[i * 16:j * 16, 0:17]
                                #cv2.namedWindow('D', cv2.WINDOW_NORMAL)
                                #cv2.imshow('D', D)
                                #print("D")
                                dif = cv2.subtract(D, compare)
                                result = not np.any(dif)  # if difference is all zeros it will return False
                                if result is True:
                                    #print("未选D")
                                    pass
                                else:
                                    #print("选则了D")
                                    answer_decide.append("D")
                                #最后判断
                                if len(answer_decide) == 0:
                                    answer_2.append(1)
                                if len(answer_decide) == 1:
                                    answer[1] = answer_decide[0]
                                if len(answer_decide) > 1:
                                    answer_1[1] = answer_decide
                                #print("1",answer_decide)
                                #print("2", answer_2)
                                #print("3", answer_1)
                                #print("4", answer)


                    if h == 1:
                        img31 = img2[0:65, h * 17:l * 17]
                        edged31 = edged2[0:65, h * 17:l * 17]
                        mask31 = mask2[0:65, h * 17:l * 17]
                        cv2.namedWindow('img31', cv2.WINDOW_NORMAL)
                        cv2.imshow('img31', img31)



                    if h == 2:
                        img32 = img2[0:65, h * 17:l * 17]
                        edged32 = edged2[0:65, h * 17:l * 17]
                        mask32 = mask2[0:65, h * 17:l * 17]
                        cv2.namedWindow('img32', cv2.WINDOW_NORMAL)
                        cv2.imshow('img32', img32)


                    if h == 3:
                        img33 = img2[0:65, h * 17:l * 17]
                        edged33 = edged2[0:65, h * 17:l * 17]
                        mask33 = mask2[0:65, h * 17:l * 17]
                        cv2.namedWindow('img33', cv2.WINDOW_NORMAL)
                        cv2.imshow('img33', img33)


                    if h == 4:
                        img34 = img2[0:65, h * 17:l * 17]
                        edged34 = edged2[0:65, h * 17:l * 17]
                        mask34 = mask2[0:65, h * 17:l * 17]
                        cv2.namedWindow('img34', cv2.WINDOW_NORMAL)
                        cv2.imshow('img34', img34)



            if j == 1:
                img3 = img[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                edged3 = edged[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                mask3 = mask[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
                cv2.imshow('img3', img3)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img35 = img3[0:65, h * 17:l * 17]
                        edged35 = edged3[0:65, h * 17:l * 17]
                        mask35 = mask3[0:65, h * 17:l * 17]
                        cv2.namedWindow('img35', cv2.WINDOW_NORMAL)
                        cv2.imshow('img35', img35)


                    if h == 1:
                        img36 = img3[0:65, h * 17:l * 17]
                        edged36 = edged3[0:65, h * 17:l * 17]
                        mask36 = mask3[0:65, h * 17:l * 17]
                        cv2.namedWindow('img36', cv2.WINDOW_NORMAL)
                        cv2.imshow('img36', img36)

                    if h == 2:
                        img37 = img3[0:65, h * 17:l * 17]
                        edged37 = edged3[0:65, h * 17:l * 17]
                        mask37 = mask3[0:65, h * 17:l * 17]
                        cv2.namedWindow('img37', cv2.WINDOW_NORMAL)
                        cv2.imshow('img37', img37)

                    if h == 3:
                        img38 = img3[0:65, h * 17:l * 17]
                        edged38 = edged3[0:65, h * 17:l * 17]
                        mask38 = mask3[0:65, h * 17:l * 17]
                        cv2.namedWindow('img38', cv2.WINDOW_NORMAL)
                        cv2.imshow('img38', img38)

                    if h == 4:
                        img39 = img3[0:65, h * 17:l * 17]
                        edged39 = edged3[0:65, h * 17:l * 17]
                        mask39 = mask3[0:65, h * 17:l * 17]
                        cv2.namedWindow('img39', cv2.WINDOW_NORMAL)
                        cv2.imshow('img39', img39)

            if j == 2:
                img4 = img[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                edged4 = edged[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                mask4 = mask[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
                cv2.imshow('img4', img4)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img40 = img4[0:65, h * 17:l * 17]
                        edged40 = edged4[0:65, h * 17:l * 17]
                        mask40 = mask4[0:65, h * 17:l * 17]
                        cv2.namedWindow('img40', cv2.WINDOW_NORMAL)
                        cv2.imshow('img40', img40)

                    if h == 2:
                        img42 = img4[0:65, h * 17:l * 17]
                        edged42 = edged4[0:65, h * 17:l * 17]
                        mask42 = mask4[0:65, h * 17:l * 17]
                        cv2.namedWindow('img42', cv2.WINDOW_NORMAL)
                        cv2.imshow('img42', img42)

                    if h == 3:
                        img43 = img4[0:65, h * 17:l * 17]
                        edged43 = edged4[0:65, h * 17:l * 17]
                        mask43 = mask4[0:65, h * 17:l * 17]
                        cv2.namedWindow('img43', cv2.WINDOW_NORMAL)
                        cv2.imshow('img43', img43)

                    if h == 4:
                        img44 = img4[0:65, h * 17:l * 17]
                        edged44 = edged4[0:65, h * 17:l * 17]
                        mask44 = mask4[0:65, h * 17:l * 17]
                        cv2.namedWindow('img44', cv2.WINDOW_NORMAL)
                        cv2.imshow('img44', img44)

            if j == 3:
                img5 = img[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                edged5 = edged[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                mask5 = mask[30+i*100:95+i*100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img5', cv2.WINDOW_NORMAL)
                cv2.imshow('img5', img5)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img45 = img5[0:65, h * 17:l * 17]
                        edged45 = edged5[0:65, h * 17:l * 17]
                        mask45 = mask5[0:65, h * 17:l * 17]
                        cv2.namedWindow('img45', cv2.WINDOW_NORMAL)
                        cv2.imshow('img45', img45)

                    if h == 1:
                        img46 = img5[0:65, h * 17:l * 17]
                        edged46 = edged5[0:65, h * 17:l * 17]
                        mask46 = mask5[0:65, h * 17:l * 17]
                        cv2.namedWindow('img46', cv2.WINDOW_NORMAL)
                        cv2.imshow('img46', img46)

                    if h == 2:
                        img47 = img5[0:65, h * 17:l * 17]
                        edged47 = edged5[0:65, h * 17:l * 17]
                        mask47 = mask5[0:65, h * 17:l * 17]
                        cv2.namedWindow('img47', cv2.WINDOW_NORMAL)
                        cv2.imshow('img47', img47)

                    if h == 3:
                        img48 = img5[0:65, h * 17:l * 17]
                        edged48 = edged5[0:65, h * 17:l * 17]
                        mask48 = mask5[0:65, h * 17:l * 17]
                        cv2.namedWindow('img48', cv2.WINDOW_NORMAL)
                        cv2.imshow('img48', img48)

                    if h == 4:
                        img49 = img5[0:65, h * 17:l * 17]
                        edged49 = edged5[0:65, h * 17:l * 17]
                        mask49 = mask5[0:65, h * 17:l * 17]
                        cv2.namedWindow('img49', cv2.WINDOW_NORMAL)
                        cv2.imshow('img49', img49)



    if i==1:
        
        for j in range(0, 4):
            if j == 0:
                img6 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged6 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask6 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img6', cv2.WINDOW_NORMAL)
                cv2.imshow('img6', img6)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img50 = img6[0:65, h * 17:l * 17]
                        edged50 = edged6[0:65, h * 17:l * 17]
                        mask50 = mask6[0:65, h * 17:l * 17]
                        cv2.namedWindow('img50', cv2.WINDOW_NORMAL)
                        cv2.imshow('img50', img50)

                    if h == 1:
                        img51 = img6[0:65, h * 17:l * 17]
                        edged51 = edged6[0:65, h * 17:l * 17]
                        mask51 = mask6[0:65, h * 17:l * 17]
                        cv2.namedWindow('img51', cv2.WINDOW_NORMAL)
                        cv2.imshow('img51', img51)

                    if h == 2:
                        img52 = img6[0:65, h * 17:l * 17]
                        edged52 = edged6[0:65, h * 17:l * 17]
                        mask52 = mask6[0:65, h * 17:l * 17]
                        cv2.namedWindow('img52', cv2.WINDOW_NORMAL)
                        cv2.imshow('img52', img52)

                    if h == 3:
                        img53 = img6[0:65, h * 17:l * 17]
                        edged53 = edged6[0:65, h * 17:l * 17]
                        mask53 = mask6[0:65, h * 17:l * 17]
                        cv2.namedWindow('img53', cv2.WINDOW_NORMAL)
                        cv2.imshow('img53', img53)

                    if h == 4:
                        img54 = img6[0:65, h * 17:l * 17]
                        edged54 = edged6[0:65, h * 17:l * 17]
                        mask54 = mask6[0:65, h * 17:l * 17]
                        cv2.namedWindow('img54', cv2.WINDOW_NORMAL)
                        cv2.imshow('img54', img54)


            if j == 1:
                img7 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged7 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask7 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img7', cv2.WINDOW_NORMAL)
                cv2.imshow('img7', img7)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img55 = img7[0:65, h * 17:l * 17]
                        edged55 = edged7[0:65, h * 17:l * 17]
                        mask55 = mask7[0:65, h * 17:l * 17]
                        cv2.namedWindow('img55', cv2.WINDOW_NORMAL)
                        cv2.imshow('img55', img55)

                    if h == 1:
                        img56 = img7[0:65, h * 17:l * 17]
                        edged56 = edged7[0:65, h * 17:l * 17]
                        mask56 = mask7[0:65, h * 17:l * 17]
                        cv2.namedWindow('img56', cv2.WINDOW_NORMAL)
                        cv2.imshow('img56', img56)

                    if h == 2:
                        img57 = img7[0:65, h * 17:l * 17]
                        edged57 = edged7[0:65, h * 17:l * 17]
                        mask57 = mask7[0:65, h * 17:l * 17]
                        cv2.namedWindow('img57', cv2.WINDOW_NORMAL)
                        cv2.imshow('img57', img57)

                    if h == 3:
                        img58 = img7[0:65, h * 17:l * 17]
                        edged58 = edged7[0:65, h * 17:l * 17]
                        mask58 = mask7[0:65, h * 17:l * 17]
                        cv2.namedWindow('img58', cv2.WINDOW_NORMAL)
                        cv2.imshow('img58', img58)

                    if h == 4:
                        img59 = img7[0:65, h * 17:l * 17]
                        edged59 = edged7[0:65, h * 17:l * 17]
                        mask59 = mask7[0:65, h * 17:l * 17]
                        cv2.namedWindow('img59', cv2.WINDOW_NORMAL)
                        cv2.imshow('img59', img59)

            if j == 2:
                img8 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged8 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask8 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img8', cv2.WINDOW_NORMAL)
                cv2.imshow('img8', img8)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img60 = img8[0:65, h * 17:l * 17]
                        edged60 = edged8[0:65, h * 17:l * 17]
                        mask60= mask8[0:65, h * 17:l * 17]
                        cv2.namedWindow('img60', cv2.WINDOW_NORMAL)
                        cv2.imshow('img60', img60)

                    if h == 1:
                        img61 = img8[0:65, h * 17:l * 17]
                        edged61 = edged8[0:65, h * 17:l * 17]
                        mask61 = mask8[0:65, h * 17:l * 17]
                        cv2.namedWindow('img61', cv2.WINDOW_NORMAL)
                        cv2.imshow('img61', img61)

                    if h == 2:
                        img62 = img8[0:65, h * 17:l * 17]
                        edged62 = edged8[0:65, h * 17:l * 17]
                        mask62 = mask8[0:65, h * 17:l * 17]
                        cv2.namedWindow('img62', cv2.WINDOW_NORMAL)
                        cv2.imshow('img62', img62)

                    if h == 3:
                        img63 = img8[0:65, h * 17:l * 17]
                        edged63 = edged8[0:65, h * 17:l * 17]
                        mask63 = mask8[0:65, h * 17:l * 17]
                        cv2.namedWindow('img63', cv2.WINDOW_NORMAL)
                        cv2.imshow('img63', img63)
                    if h == 4:
                        img64 = img8[0:65, h * 17:l * 17]
                        edged64 = edged8[0:65, h * 17:l * 17]
                        mask64 = mask8[0:65, h * 17:l * 17]
                        cv2.namedWindow('img64', cv2.WINDOW_NORMAL)
                        cv2.imshow('img64', img64)

            if j == 3:
                img9 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged9 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask9 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img9', cv2.WINDOW_NORMAL)
                cv2.imshow('img9', img9)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img65 = img9[0:65, h * 17:l * 17]
                        edged65 = edged9[0:65, h * 17:l * 17]
                        mask65 = mask9[0:65, h * 17:l * 17]
                        cv2.namedWindow('img65', cv2.WINDOW_NORMAL)
                        cv2.imshow('img65', img65)
                    if h == 1:
                        img66 = img9[0:65, h * 17:l * 17]
                        edged66 = edged9[0:65, h * 17:l * 17]
                        mask66 = mask9[0:65, h * 17:l * 17]
                        cv2.namedWindow('img66', cv2.WINDOW_NORMAL)
                        cv2.imshow('img66', img66)
                    if h == 2:
                        img67 = img9[0:65, h * 17:l * 17]
                        edged67 = edged9[0:65, h * 17:l * 17]
                        mask67 = mask9[0:65, h * 17:l * 17]
                        cv2.namedWindow('img67', cv2.WINDOW_NORMAL)
                        cv2.imshow('img67', img67)
                    if h == 3:
                        img68 = img9[0:65, h * 17:l * 17]
                        edged68 = edged9[0:65, h * 17:l * 17]
                        mask68 = mask9[0:65, h * 17:l * 17]
                        cv2.namedWindow('img68', cv2.WINDOW_NORMAL)
                        cv2.imshow('img68', img68)
                    if h == 4:
                        img69 = img9[0:65, h * 17:l * 17]
                        edged69 = edged9[0:65, h * 17:l * 17]
                        mask69 = mask9[0:65, h * 17:l * 17]
                        cv2.namedWindow('img69', cv2.WINDOW_NORMAL)
                        cv2.imshow('img69', img69)

    if i==2:
        for j in range(0, 4):
            if j == 0:
                img10 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged10 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask10 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img10', cv2.WINDOW_NORMAL)
                cv2.imshow('img10', img10)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img70 = img10[0:65, h * 17:l * 17]
                        edged70 = edged10[0:65, h * 17:l * 17]
                        mask70 = mask10[0:65, h * 17:l * 17]
                        cv2.namedWindow('img70', cv2.WINDOW_NORMAL)
                        cv2.imshow('img70', img70)
                    if h == 1:
                        img71 = img10[0:65, h * 17:l * 17]
                        edged71 = edged10[0:65, h * 17:l * 17]
                        mask71 = mask10[0:65, h * 17:l * 17]
                        cv2.namedWindow('img71', cv2.WINDOW_NORMAL)
                        cv2.imshow('img71', img71)
                    if h == 2:
                        img72 = img10[0:65, h * 17:l * 17]
                        edged72 = edged10[0:65, h * 17:l * 17]
                        mask72 = mask10[0:65, h * 17:l * 17]
                        cv2.namedWindow('img72', cv2.WINDOW_NORMAL)
                        cv2.imshow('img72', img72)
                    if h == 3:
                        img73 = img10[0:65, h * 17:l * 17]
                        edged73 = edged10[0:65, h * 17:l * 17]
                        mask73 = mask10[0:65, h * 17:l * 17]
                        cv2.namedWindow('img73', cv2.WINDOW_NORMAL)
                        cv2.imshow('img73', img73)
                    if h == 4:
                        img74 = img10[0:65, h * 17:l * 17]
                        edged74 = edged10[0:65, h * 17:l * 17]
                        mask74 = mask10[0:65, h * 17:l * 17]
                        cv2.namedWindow('img74', cv2.WINDOW_NORMAL)
                        cv2.imshow('img74', img74)

            if j == 1:
                img11 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged11 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask11 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img11', cv2.WINDOW_NORMAL)
                cv2.imshow('img11', img11)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img75 = img11[0:65, h * 17:l * 17]
                        edged75 = edged11[0:65, h * 17:l * 17]
                        mask75 = mask11[0:65, h * 17:l * 17]
                        cv2.namedWindow('img75', cv2.WINDOW_NORMAL)
                        cv2.imshow('img75', img75)
                    if h == 1:
                        img76 = img11[0:65, h * 17:l * 17]
                        edged76 = edged11[0:65, h * 17:l * 17]
                        mask76 = mask11[0:65, h * 17:l * 17]
                        cv2.namedWindow('img76', cv2.WINDOW_NORMAL)
                        cv2.imshow('img76', img76)
                    if h == 2:
                        img77 = img11[0:65, h * 17:l * 17]
                        edged77 = edged11[0:65, h * 17:l * 17]
                        mask77 = mask11[0:65, h * 17:l * 17]
                        cv2.namedWindow('img77', cv2.WINDOW_NORMAL)
                        cv2.imshow('img77', img77)
                    if h == 3:
                        img78 = img11[0:65, h * 17:l * 17]
                        edged78 = edged11[0:65, h * 17:l * 17]
                        mask78 = mask11[0:65, h * 17:l * 17]
                        cv2.namedWindow('img78', cv2.WINDOW_NORMAL)
                        cv2.imshow('img78', img78)
                    if h == 4:
                        img79 = img11[0:65, h * 17:l * 17]
                        edged79 = edged11[0:65, h * 17:l * 17]
                        mask79 = mask11[0:65, h * 17:l * 17]
                        cv2.namedWindow('img79', cv2.WINDOW_NORMAL)
                        cv2.imshow('img79', img79)

            if j == 2:
                img12 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged12 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask12 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img12', cv2.WINDOW_NORMAL)
                cv2.imshow('img12', img12)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img80 = img12[0:65, h * 17:l * 17]
                        edged80 = edged12[0:65, h * 17:l * 17]
                        mask80 = mask12[0:65, h * 17:l * 17]
                        cv2.namedWindow('img80', cv2.WINDOW_NORMAL)
                        cv2.imshow('img80', img80)
                    if h == 1:
                        img81 = img12[0:65, h * 17:l * 17]
                        edged81 = edged12[0:65, h * 17:l * 17]
                        mask81 = mask12[0:65, h * 17:l * 17]
                        cv2.namedWindow('img81', cv2.WINDOW_NORMAL)
                        cv2.imshow('img81', img81)
                    if h == 2:
                        img82 = img12[0:65, h * 17:l * 17]
                        edged82 = edged12[0:65, h * 17:l * 17]
                        mask82 = mask12[0:65, h * 17:l * 17]
                        cv2.namedWindow('img82', cv2.WINDOW_NORMAL)
                        cv2.imshow('img82', img82)
                    if h == 3:
                        img83 = img12[0:65, h * 17:l * 17]
                        edged83 = edged12[0:65, h * 17:l * 17]
                        mask83 = mask12[0:65, h * 17:l * 17]
                        cv2.namedWindow('img83', cv2.WINDOW_NORMAL)
                        cv2.imshow('img83', img83)
                    if h == 4:
                        img84 = img12[0:65, h * 17:l * 17]
                        edged84 = edged12[0:65, h * 17:l * 17]
                        mask84 = mask12[0:65, h * 17:l * 17]
                        cv2.namedWindow('img84', cv2.WINDOW_NORMAL)
                        cv2.imshow('img84', img84)

            if j == 3:
                img13 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged13 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask13 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img13', cv2.WINDOW_NORMAL)
                cv2.imshow('img13', img13)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img85 = img13[0:65, h * 17:l * 17]
                        edged85 = edged13[0:65, h * 17:l * 17]
                        mask85 = mask13[0:65, h * 17:l * 17]
                        cv2.namedWindow('img85', cv2.WINDOW_NORMAL)
                        cv2.imshow('img85', img85)
                    if h == 1:
                        img86 = img13[0:65, h * 17:l * 17]
                        edged86 = edged13[0:65, h * 17:l * 17]
                        mask86 = mask13[0:65, h * 17:l * 17]
                        cv2.namedWindow('img86', cv2.WINDOW_NORMAL)
                        cv2.imshow('img86', img86)
                    if h == 2:
                        img87 = img13[0:65, h * 17:l * 17]
                        edged87 = edged13[0:65, h * 17:l * 17]
                        mask87 = mask13[0:65, h * 17:l * 17]
                        cv2.namedWindow('img87', cv2.WINDOW_NORMAL)
                        cv2.imshow('img87', img87)
                    if h == 3:
                        img88 = img13[0:65, h * 17:l * 17]
                        edged88 = edged13[0:65, h * 17:l * 17]
                        mask88 = mask13[0:65, h * 17:l * 17]
                        cv2.namedWindow('img88', cv2.WINDOW_NORMAL)
                        cv2.imshow('img88', img88)
                    if h == 4:
                        img89 = img13[0:65, h * 17:l * 17]
                        edged89 = edged13[0:65, h * 17:l * 17]
                        mask89 = mask13[0:65, h * 17:l * 17]
                        cv2.namedWindow('img89', cv2.WINDOW_NORMAL)
                        cv2.imshow('img89', img89)

    if i==3:
        
        for j in range(0, 4):
            if j == 0:
                img14 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged14 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask14 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img14', cv2.WINDOW_NORMAL)
                cv2.imshow('img14', img14)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img90 = img14[0:65, h * 17:l * 17]
                        edged90 = edged14[0:65, h * 17:l * 17]
                        mask90 = mask14[0:65, h * 17:l * 17]
                        cv2.namedWindow('img90', cv2.WINDOW_NORMAL)
                        cv2.imshow('img90', img90)
                    if h == 1:
                        img91 = img14[0:65, h * 17:l * 17]
                        edged91 = edged14[0:65, h * 17:l * 17]
                        mask91 = mask14[0:65, h * 17:l * 17]
                        cv2.namedWindow('img91', cv2.WINDOW_NORMAL)
                        cv2.imshow('img91', img91)
                    if h == 2:
                        img92 = img14[0:65, h * 17:l * 17]
                        edged92 = edged14[0:65, h * 17:l * 17]
                        mask92 = mask14[0:65, h * 17:l * 17]
                        cv2.namedWindow('img92', cv2.WINDOW_NORMAL)
                        cv2.imshow('img92', img92)
                    if h == 3:
                        img93 = img14[0:65, h * 17:l * 17]
                        edged93 = edged14[0:65, h * 17:l * 17]
                        mask93 = mask14[0:65, h * 17:l * 17]
                        cv2.namedWindow('img93', cv2.WINDOW_NORMAL)
                        cv2.imshow('img93', img93)
                    if h == 4:
                        img93 = img14[0:65, h * 17:l * 17]
                        edged93 = edged14[0:65, h * 17:l * 17]
                        mask93 = mask14[0:65, h * 17:l * 17]
                        cv2.namedWindow('img93', cv2.WINDOW_NORMAL)
                        cv2.imshow('img93', img93)

            if j == 1:
                img15 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged15 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask15 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img15', cv2.WINDOW_NORMAL)
                cv2.imshow('img15', img15)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img94 = img15[0:65, h * 17:l * 17]
                        edged94 = edged15[0:65, h * 17:l * 17]
                        mask94 = mask15[0:65, h * 17:l * 17]
                        cv2.namedWindow('img94', cv2.WINDOW_NORMAL)
                        cv2.imshow('img94', img94)
                    if h == 1:
                        img95 = img15[0:65, h * 17:l * 17]
                        edged95 = edged15[0:65, h * 17:l * 17]
                        mask95 = mask15[0:65, h * 17:l * 17]
                        cv2.namedWindow('img95', cv2.WINDOW_NORMAL)
                        cv2.imshow('img95', img95)
                    if h == 2:
                        img96 = img15[0:65, h * 17:l * 17]
                        edged96 = edged15[0:65, h * 17:l * 17]
                        mask96 = mask15[0:65, h * 17:l * 17]
                        cv2.namedWindow('img96', cv2.WINDOW_NORMAL)
                        cv2.imshow('img96', img96)
                    if h == 3:
                        img97 = img15[0:65, h * 17:l * 17]
                        edged97 = edged15[0:65, h * 17:l * 17]
                        mask97 = mask15[0:65, h * 17:l * 17]
                        cv2.namedWindow('img97', cv2.WINDOW_NORMAL)
                        cv2.imshow('img97', img97)
                    if h == 4:
                        img98 = img15[0:65, h * 17:l * 17]
                        edged98 = edged15[0:65, h * 17:l * 17]
                        mask98 = mask15[0:65, h * 17:l * 17]
                        cv2.namedWindow('img98', cv2.WINDOW_NORMAL)
                        cv2.imshow('img98', img98)

            if j == 2:
                img16 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged16 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask16 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img16', cv2.WINDOW_NORMAL)
                cv2.imshow('img16', img16)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img99 = img16[0:65, h * 17:l * 17]
                        edged99 = edged16[0:65, h * 17:l * 17]
                        mask99 = mask16[0:65, h * 17:l * 17]
                        cv2.namedWindow('img99', cv2.WINDOW_NORMAL)
                        cv2.imshow('img99', img99)
                    if h == 1:
                        img100 = img16[0:65, h * 17:l * 17]
                        edged100 = edged16[0:65, h * 17:l * 17]
                        mask100 = mask16[0:65, h * 17:l * 17]
                        cv2.namedWindow('img100', cv2.WINDOW_NORMAL)
                        cv2.imshow('img100', img100)
                    if h == 2:
                        img101 = img16[0:65, h * 17:l * 17]
                        edged101 = edged16[0:65, h * 17:l * 17]
                        mask101 = mask16[0:65, h * 17:l * 17]
                        cv2.namedWindow('img101', cv2.WINDOW_NORMAL)
                        cv2.imshow('img101', img101)
                    if h == 3:
                        img102 = img16[0:65, h * 17:l * 17]
                        edged102 = edged16[0:65, h * 17:l * 17]
                        mask102 = mask16[0:65, h * 17:l * 17]
                        cv2.namedWindow('img102', cv2.WINDOW_NORMAL)
                        cv2.imshow('img102', img102)
                    if h == 4:
                        img103 = img16[0:65, h * 17:l * 17]
                        edged103 = edged16[0:65, h * 17:l * 17]
                        mask103 = mask16[0:65, h * 17:l * 17]
                        cv2.namedWindow('img103', cv2.WINDOW_NORMAL)
                        cv2.imshow('img103', img103)

            if j == 3:
                img17 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged17 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask17 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img17', cv2.WINDOW_NORMAL)
                cv2.imshow('img17', img17)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img104 = img17[0:65, h * 17:l * 17]
                        edged104 = edged17[0:65, h * 17:l * 17]
                        mask104 = mask17[0:65, h * 17:l * 17]
                        cv2.namedWindow('img104', cv2.WINDOW_NORMAL)
                        cv2.imshow('img104', img104)
                    if h == 1:
                        img105 = img17[0:65, h * 17:l * 17]
                        edged105 = edged17[0:65, h * 17:l * 17]
                        mask105 = mask17[0:65, h * 17:l * 17]
                        cv2.namedWindow('img105', cv2.WINDOW_NORMAL)
                        cv2.imshow('img105', img105)
                    if h == 2:
                        img106 = img17[0:65, h * 17:l * 17]
                        edged106 = edged17[0:65, h * 17:l * 17]
                        mask106 = mask17[0:65, h * 17:l * 17]
                        cv2.namedWindow('img106', cv2.WINDOW_NORMAL)
                        cv2.imshow('img106', img106)
                    if h == 3:
                        img107 = img17[0:65, h * 17:l * 17]
                        edged107 = edged17[0:65, h * 17:l * 17]
                        mask107 = mask17[0:65, h * 17:l * 17]
                        cv2.namedWindow('img107', cv2.WINDOW_NORMAL)
                        cv2.imshow('img107', img107)
                    if h == 4:
                        img108 = img17[0:65, h * 17:l * 17]
                        edged108 = edged17[0:65, h * 17:l * 17]
                        mask108 = mask17[0:65, h * 17:l * 17]
                        cv2.namedWindow('img108', cv2.WINDOW_NORMAL)
                        cv2.imshow('img108', img108)

    if i==4:
        
        for j in range(0, 4):
            if j == 0:
                img18 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged18 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask18 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img18', cv2.WINDOW_NORMAL)
                cv2.imshow('img18', img18)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img109 = img18[0:65, h * 17:l * 17]
                        edged109 = edged18[0:65, h * 17:l * 17]
                        mask109 = mask18[0:65, h * 17:l * 17]
                        cv2.namedWindow('img109', cv2.WINDOW_NORMAL)
                        cv2.imshow('img109', img109)
                    if h == 1:
                        img110 = img18[0:65, h * 17:l * 17]
                        edged110 = edged18[0:65, h * 17:l * 17]
                        mask110 = mask18[0:65, h * 17:l * 17]
                        cv2.namedWindow('img110', cv2.WINDOW_NORMAL)
                        cv2.imshow('img110', img110)
                    if h == 2:
                        img111 = img18[0:65, h * 17:l * 17]
                        edged111 = edged18[0:65, h * 17:l * 17]
                        mask111 = mask18[0:65, h * 17:l * 17]
                        cv2.namedWindow('img111', cv2.WINDOW_NORMAL)
                        cv2.imshow('img111', img111)
                    if h == 3:
                        img112 = img18[0:65, h * 17:l * 17]
                        edged112 = edged18[0:65, h * 17:l * 17]
                        mask112 = mask18[0:65, h * 17:l * 17]
                        cv2.namedWindow('img112', cv2.WINDOW_NORMAL)
                        cv2.imshow('img112', img112)
                    if h == 4:
                        img113 = img18[0:65, h * 17:l * 17]
                        edged113 = edged18[0:65, h * 17:l * 17]
                        mask113 = mask18[0:65, h * 17:l * 17]
                        cv2.namedWindow('img113', cv2.WINDOW_NORMAL)
                        cv2.imshow('img113', img113)

            if j == 1:
                img19 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged19 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask19 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img19', cv2.WINDOW_NORMAL)
                cv2.imshow('img19', img19)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img114 = img19[0:65, h * 17:l * 17]
                        edged114 = edged19[0:65, h * 17:l * 17]
                        mask114 = mask19[0:65, h * 17:l * 17]
                        cv2.namedWindow('img114', cv2.WINDOW_NORMAL)
                        cv2.imshow('img114', img114)
                    if h == 1:
                        img115 = img19[0:65, h * 17:l * 17]
                        edged115 = edged19[0:65, h * 17:l * 17]
                        mask115 = mask19[0:65, h * 17:l * 17]
                        cv2.namedWindow('img115', cv2.WINDOW_NORMAL)
                        cv2.imshow('img115', img115)
                    if h == 2:
                        img116 = img19[0:65, h * 17:l * 17]
                        edged116 = edged19[0:65, h * 17:l * 17]
                        mask116 = mask19[0:65, h * 17:l * 17]
                        cv2.namedWindow('img116', cv2.WINDOW_NORMAL)
                        cv2.imshow('img116', img116)
                    if h == 3:
                        img117 = img19[0:65, h * 17:l * 17]
                        edged117 = edged19[0:65, h * 17:l * 17]
                        mask117 = mask19[0:65, h * 17:l * 17]
                        cv2.namedWindow('img117', cv2.WINDOW_NORMAL)
                        cv2.imshow('img117', img117)
                    if h == 4:
                        img118 = img19[0:65, h * 17:l * 17]
                        edged118 = edged19[0:65, h * 17:l * 17]
                        mask118 = mask19[0:65, h * 17:l * 17]
                        cv2.namedWindow('img118', cv2.WINDOW_NORMAL)
                        cv2.imshow('img118', img118)

            if j == 2:
                img20 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged20 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask20 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img20', cv2.WINDOW_NORMAL)
                cv2.imshow('img20', img20)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img119 = img20[0:65, h * 17:l * 17]
                        edged119 = edged20[0:65, h * 17:l * 17]
                        mask119 = mask20[0:65, h * 17:l * 17]
                        cv2.namedWindow('img119', cv2.WINDOW_NORMAL)
                        cv2.imshow('img119', img119)
                    if h == 1:
                        img120 = img20[0:65, h * 17:l * 17]
                        edged120 = edged20[0:65, h * 17:l * 17]
                        mask120 = mask20[0:65, h * 17:l * 17]
                        cv2.namedWindow('img120', cv2.WINDOW_NORMAL)
                        cv2.imshow('img120', img120)
                    if h == 2:
                        img121 = img20[0:65, h * 17:l * 17]
                        edged121 = edged20[0:65, h * 17:l * 17]
                        mask121 = mask20[0:65, h * 17:l * 17]
                        cv2.namedWindow('img121', cv2.WINDOW_NORMAL)
                        cv2.imshow('img121', img121)
                    if h == 3:
                        img122 = img20[0:65, h * 17:l * 17]
                        edged122 = edged20[0:65, h * 17:l * 17]
                        mask122 = mask20[0:65, h * 17:l * 17]
                        cv2.namedWindow('img122', cv2.WINDOW_NORMAL)
                        cv2.imshow('img122', img122)
                    if h == 4:
                        img123 = img20[0:65, h * 17:l * 17]
                        edged123 = edged20[0:65, h * 17:l * 17]
                        mask123 = mask20[0:65, h * 17:l * 17]
                        cv2.namedWindow('img123', cv2.WINDOW_NORMAL)
                        cv2.imshow('img123', img123)

            if j == 3:
                img21 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged21 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask21 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img21', cv2.WINDOW_NORMAL)
                cv2.imshow('img21', img21)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img124 = img21[0:65, h * 17:l * 17]
                        edged124 = edged21[0:65, h * 17:l * 17]
                        mask124 = mask21[0:65, h * 17:l * 17]
                        cv2.namedWindow('img124', cv2.WINDOW_NORMAL)
                        cv2.imshow('img124', img124)
                    if h == 1:
                        img125 = img21[0:65, h * 17:l * 17]
                        edged125 = edged21[0:65, h * 17:l * 17]
                        mask125 = mask21[0:65, h * 17:l * 17]
                        cv2.namedWindow('img125', cv2.WINDOW_NORMAL)
                        cv2.imshow('img125', img125)
                    if h == 2:
                        img126 = img21[0:65, h * 17:l * 17]
                        edged126 = edged21[0:65, h * 17:l * 17]
                        mask126 = mask21[0:65, h * 17:l * 17]
                        cv2.namedWindow('img126', cv2.WINDOW_NORMAL)
                        cv2.imshow('img126', img126)
                    if h == 3:
                        img127 = img21[0:65, h * 17:l * 17]
                        edged127 = edged21[0:65, h * 17:l * 17]
                        mask127 = mask21[0:65, h * 17:l * 17]
                        cv2.namedWindow('img127', cv2.WINDOW_NORMAL)
                        cv2.imshow('img127', img127)
                    if h == 4:
                        img128 = img21[0:65, h * 17:l * 17]
                        edged128 = edged21[0:65, h * 17:l * 17]
                        mask128 = mask21[0:65, h * 17:l * 17]
                        cv2.namedWindow('img128', cv2.WINDOW_NORMAL)
                        cv2.imshow('img128', img128)

    if i==5:
        
        for j in range(0, 4):
            if j == 0:
                img22 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged22 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask22 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img22', cv2.WINDOW_NORMAL)
                cv2.imshow('img22', img22)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img129 = img22[0:65, h * 17:l * 17]
                        edged129 = edged22[0:65, h * 17:l * 17]
                        mask129 = mask22[0:65, h * 17:l * 17]
                        cv2.namedWindow('img129', cv2.WINDOW_NORMAL)
                        cv2.imshow('img129', img129)
                    if h == 1:
                        img130 = img22[0:65, h * 17:l * 17]
                        edged130 = edged22[0:65, h * 17:l * 17]
                        mask130 = mask22[0:65, h * 17:l * 17]
                        cv2.namedWindow('img130', cv2.WINDOW_NORMAL)
                        cv2.imshow('img130', img130)
                    if h == 2:
                        img131 = img22[0:65, h * 17:l * 17]
                        edged131 = edged22[0:65, h * 17:l * 17]
                        mask131 = mask22[0:65, h * 17:l * 17]
                        cv2.namedWindow('img131', cv2.WINDOW_NORMAL)
                        cv2.imshow('img131', img131)
                    if h == 3:
                        img132 = img22[0:65, h * 17:l * 17]
                        edged132 = edged22[0:65, h * 17:l * 17]
                        mask132 = mask22[0:65, h * 17:l * 17]
                        cv2.namedWindow('img132', cv2.WINDOW_NORMAL)
                        cv2.imshow('img132', img132)
                    if h == 4:
                        img133 = img22[0:65, h * 17:l * 17]
                        edged133 = edged22[0:65, h * 17:l * 17]
                        mask133 = mask22[0:65, h * 17:l * 17]
                        cv2.namedWindow('img133', cv2.WINDOW_NORMAL)
                        cv2.imshow('img133', img133)

            if j == 1:
                img23 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged23 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask23 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img23', cv2.WINDOW_NORMAL)
                cv2.imshow('img23', img23)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img134 = img23[0:65, h * 17:l * 17]
                        edged134 = edged23[0:65, h * 17:l * 17]
                        mask134 = mask23[0:65, h * 17:l * 17]
                        cv2.namedWindow('img134', cv2.WINDOW_NORMAL)
                        cv2.imshow('img134', img134)
                    if h == 1:
                        img135 = img23[0:65, h * 17:l * 17]
                        edged135 = edged23[0:65, h * 17:l * 17]
                        mask135 = mask23[0:65, h * 17:l * 17]
                        cv2.namedWindow('img135', cv2.WINDOW_NORMAL)
                        cv2.imshow('img135', img135)
                    if h == 2:
                        img136 = img23[0:65, h * 17:l * 17]
                        edged136 = edged23[0:65, h * 17:l * 17]
                        mask136 = mask23[0:65, h * 17:l * 17]
                        cv2.namedWindow('img136', cv2.WINDOW_NORMAL)
                        cv2.imshow('img136', img136)
                    if h == 3:
                        img137 = img23[0:65, h * 17:l * 17]
                        edged137 = edged23[0:65, h * 17:l * 17]
                        mask137 = mask23[0:65, h * 17:l * 17]
                        cv2.namedWindow('img137', cv2.WINDOW_NORMAL)
                        cv2.imshow('img137', img137)
                    if h == 4:
                        img138 = img23[0:65, h * 17:l * 17]
                        edged138 = edged23[0:65, h * 17:l * 17]
                        mask138 = mask23[0:65, h * 17:l * 17]
                        cv2.namedWindow('img138', cv2.WINDOW_NORMAL)
                        cv2.imshow('img138', img138)

            if j == 2:
                img24 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged24 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask24 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img24', cv2.WINDOW_NORMAL)
                cv2.imshow('img24', img24)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img139 = img24[0:65, h * 17:l * 17]
                        edged139 = edged24[0:65, h * 17:l * 17]
                        mask139 = mask24[0:65, h * 17:l * 17]
                        cv2.namedWindow('img139', cv2.WINDOW_NORMAL)
                        cv2.imshow('img139', img139)
                    if h == 1:
                        img140 = img24[0:65, h * 17:l * 17]
                        edged140 = edged24[0:65, h * 17:l * 17]
                        mask140 = mask24[0:65, h * 17:l * 17]
                        cv2.namedWindow('img140', cv2.WINDOW_NORMAL)
                        cv2.imshow('img140', img140)
                    if h == 2:
                        img141 = img24[0:65, h * 17:l * 17]
                        edged141 = edged24[0:65, h * 17:l * 17]
                        mask141 = mask24[0:65, h * 17:l * 17]
                        cv2.namedWindow('img141', cv2.WINDOW_NORMAL)
                        cv2.imshow('img141', img141)
                    if h == 3:
                        img142 = img24[0:65, h * 17:l * 17]
                        edged142 = edged24[0:65, h * 17:l * 17]
                        mask142 = mask24[0:65, h * 17:l * 17]
                        cv2.namedWindow('img142', cv2.WINDOW_NORMAL)
                        cv2.imshow('img142', img142)
                    if h == 4:
                        img143 = img24[0:65, h * 17:l * 17]
                        edged143 = edged24[0:65, h * 17:l * 17]
                        mask143 = mask24[0:65, h * 17:l * 17]
                        cv2.namedWindow('img143', cv2.WINDOW_NORMAL)
                        cv2.imshow('img143', img143)

            if j == 3:
                img25 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged25 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask25 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img25', cv2.WINDOW_NORMAL)
                cv2.imshow('img25', img25)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img144 = img25[0:65, h * 17:l * 17]
                        edged144 = edged25[0:65, h * 17:l * 17]
                        mask144 = mask25[0:65, h * 17:l * 17]
                        cv2.namedWindow('img144', cv2.WINDOW_NORMAL)
                        cv2.imshow('img144', img144)
                    if h == 1:
                        img145 = img25[0:65, h * 17:l * 17]
                        edged145 = edged25[0:65, h * 17:l * 17]
                        mask145 = mask25[0:65, h * 17:l * 17]
                        cv2.namedWindow('img145', cv2.WINDOW_NORMAL)
                        cv2.imshow('img145', img145)
                    if h == 2:
                        img146 = img25[0:65, h * 17:l * 17]
                        edged146 = edged25[0:65, h * 17:l * 17]
                        mask146 = mask25[0:65, h * 17:l * 17]
                        cv2.namedWindow('img146', cv2.WINDOW_NORMAL)
                        cv2.imshow('img146', img146)
                    if h == 3:
                        img147 = img25[0:65, h * 17:l * 17]
                        edged147 = edged25[0:65, h * 17:l * 17]
                        mask147 = mask25[0:65, h * 17:l * 17]
                        cv2.namedWindow('img147', cv2.WINDOW_NORMAL)
                        cv2.imshow('img147', img147)
                    if h == 4:
                        img148 = img25[0:65, h * 17:l * 17]
                        edged148 = edged25[0:65, h * 17:l * 17]
                        mask148 = mask25[0:65, h * 17:l * 17]
                        cv2.namedWindow('img148', cv2.WINDOW_NORMAL)
                        cv2.imshow('img148', img148)

    if i==6:
        for j in range(0, 4):
            if j == 0:
                img26 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged26 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask26 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img26', cv2.WINDOW_NORMAL)
                cv2.imshow('img26', img26)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img149 = img26[0:65, h * 17:l * 17]
                        edged149 = edged26[0:65, h * 17:l * 17]
                        mask149 = mask26[0:65, h * 17:l * 17]
                        cv2.namedWindow('img149', cv2.WINDOW_NORMAL)
                        cv2.imshow('img149', img149)
                    if h == 1:
                        img150 = img26[0:65, h * 17:l * 17]
                        edged150 = edged26[0:65, h * 17:l * 17]
                        mask150 = mask26[0:65, h * 17:l * 17]
                        cv2.namedWindow('img150', cv2.WINDOW_NORMAL)
                        cv2.imshow('img150', img150)
                    if h == 2:
                        img151 = img26[0:65, h * 17:l * 17]
                        edged151 = edged26[0:65, h * 17:l * 17]
                        mask151 = mask26[0:65, h * 17:l * 17]
                        cv2.namedWindow('img151', cv2.WINDOW_NORMAL)
                        cv2.imshow('img151', img151)
                    if h == 3:
                        img152 = img26[0:65, h * 17:l * 17]
                        edged152 = edged26[0:65, h * 17:l * 17]
                        mask152 = mask26[0:65, h * 17:l * 17]
                        cv2.namedWindow('img152', cv2.WINDOW_NORMAL)
                        cv2.imshow('img152', img152)
                    if h == 4:
                        img153 = img26[0:65, h * 17:l * 17]
                        edged153 = edged26[0:65, h * 17:l * 17]
                        mask153 = mask26[0:65, h * 17:l * 17]
                        cv2.namedWindow('img153', cv2.WINDOW_NORMAL)
                        cv2.imshow('img153', img153)

            if j == 1:
                img27 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged27 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask27 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img27', cv2.WINDOW_NORMAL)
                cv2.imshow('img27', img27)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img154 = img27[0:65, h * 17:l * 17]
                        edged154 = edged27[0:65, h * 17:l * 17]
                        mask154 = mask27[0:65, h * 17:l * 17]
                        cv2.namedWindow('img154', cv2.WINDOW_NORMAL)
                        cv2.imshow('img154', img154)
                    if h == 1:
                        img155 = img27[0:65, h * 17:l * 17]
                        edged155 = edged27[0:65, h * 17:l * 17]
                        mask155 = mask27[0:65, h * 17:l * 17]
                        cv2.namedWindow('img155', cv2.WINDOW_NORMAL)
                        cv2.imshow('img155', img155)
                    if h == 2:
                        img156 = img27[0:65, h * 17:l * 17]
                        edged156 = edged27[0:65, h * 17:l * 17]
                        mask156 = mask27[0:65, h * 17:l * 17]
                        cv2.namedWindow('img156', cv2.WINDOW_NORMAL)
                        cv2.imshow('img156', img156)
                    if h == 3:
                        img157 = img27[0:65, h * 17:l * 17]
                        edged157 = edged27[0:65, h * 17:l * 17]
                        mask157 = mask27[0:65, h * 17:l * 17]
                        cv2.namedWindow('img157', cv2.WINDOW_NORMAL)
                        cv2.imshow('img157', img157)
                    if h == 4:
                        img158 = img27[0:65, h * 17:l * 17]
                        edged158 = edged27[0:65, h * 17:l * 17]
                        mask158 = mask27[0:65, h * 17:l * 17]
                        cv2.namedWindow('img158', cv2.WINDOW_NORMAL)
                        cv2.imshow('img158', img158)

            if j == 2:
                img28 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged28 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask28 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img28', cv2.WINDOW_NORMAL)
                cv2.imshow('img28', img28)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img159 = img28[0:65, h * 17:l * 17]
                        edged159 = edged28[0:65, h * 17:l * 17]
                        mask159 = mask28[0:65, h * 17:l * 17]
                        cv2.namedWindow('img159', cv2.WINDOW_NORMAL)
                        cv2.imshow('img159', img159)
                    if h == 1:
                        img160 = img28[0:65, h * 17:l * 17]
                        edged160 = edged28[0:65, h * 17:l * 17]
                        mask160 = mask28[0:65, h * 17:l * 17]
                        cv2.namedWindow('img160', cv2.WINDOW_NORMAL)
                        cv2.imshow('img160', img160)
                    if h == 2:
                        img161 = img28[0:65, h * 17:l * 17]
                        edged161 = edged28[0:65, h * 17:l * 17]
                        mask161 = mask28[0:65, h * 17:l * 17]
                        cv2.namedWindow('img161', cv2.WINDOW_NORMAL)
                        cv2.imshow('img161', img161)
                    if h == 3:
                        img162 = img28[0:65, h * 17:l * 17]
                        edged162 = edged28[0:65, h * 17:l * 17]
                        mask162 = mask28[0:65, h * 17:l * 17]
                        cv2.namedWindow('img162', cv2.WINDOW_NORMAL)
                        cv2.imshow('img162', img162)
                    if h == 4:
                        img163 = img28[0:65, h * 17:l * 17]
                        edged163 = edged28[0:65, h * 17:l * 17]
                        mask163 = mask28[0:65, h * 17:l * 17]
                        cv2.namedWindow('img163', cv2.WINDOW_NORMAL)
                        cv2.imshow('img163', img163)

            if j == 3:
                img29 = img[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                edged29 = edged[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                mask29 = mask[30 + i * 100:95 + i * 100, 3 + j * 104:87 + j * 104]
                cv2.namedWindow('img29', cv2.WINDOW_NORMAL)
                cv2.imshow('img29', img29)
                for h in range(0, 5):
                    l = h + 1
                    if h == 0:
                        img164 = img29[0:65, h * 17:l * 17]
                        edged164 = edged29[0:65, h * 17:l * 17]
                        mask164 = mask29[0:65, h * 17:l * 17]
                        cv2.namedWindow('img164', cv2.WINDOW_NORMAL)
                        cv2.imshow('img164', img164)
                    if h == 1:
                        img165 = img29[0:65, h * 17:l * 17]
                        edged165 = edged29[0:65, h * 17:l * 17]
                        mask165 = mask29[0:65, h * 17:l * 17]
                        cv2.namedWindow('img165', cv2.WINDOW_NORMAL)
                        cv2.imshow('img165', img165)
                    if h == 2:
                        img166 = img29[0:65, h * 17:l * 17]
                        edged166 = edged29[0:65, h * 17:l * 17]
                        mask166 = mask29[0:65, h * 17:l * 17]
                        cv2.namedWindow('img166', cv2.WINDOW_NORMAL)
                        cv2.imshow('img166', img166)
                    if h == 3:
                        img167 = img29[0:65, h * 17:l * 17]
                        edged167 = edged29[0:65, h * 17:l * 17]
                        mask167 = mask29[0:65, h * 17:l * 17]
                        cv2.namedWindow('img167', cv2.WINDOW_NORMAL)
                        cv2.imshow('img167', img167)
                    if h == 4:
                        img168 = img29[0:65, h * 17:l * 17]
                        edged168 = edged29[0:65, h * 17:l * 17]
                        mask168 = mask29[0:65, h * 17:l * 17]
                        cv2.namedWindow('img168', cv2.WINDOW_NORMAL)
                        cv2.imshow('img168', img168)

"""
"""
for i in range(0, 4):
    if i==0:
        img2 = img[30:95, 3+i*104:87+i*104]
        edged2 = edged[30:95, 3+i*104:87+i*104]
        mask2 = mask[30:95, 3+i*104:87+i*104]
        cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
        cv2.imshow('img2', img2)
    if i==1:
        img3 = img[30:95, 3 + i * 104:87 + i * 104]
        edged3 = edged[30:95, 3 + i * 104:87 + i * 104]
        mask3= mask[30:95, 3 + i * 104:87 + i * 104]
        cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
        cv2.imshow('img3', img3)
    if i==2:
        img4 = img[30:95, 3 + i * 104:87 + i * 104]
        edged4 = edged[30:95, 3 + i * 104:87 + i * 104]
        mask4 = mask[30:95, 3 + i * 104:87 + i * 104]
        cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
        cv2.imshow('img4', img4)
    if i==3:
        img5 = img[30:95, 3 + i * 104:87 + i * 104]
        edged5 = edged[30:95, 3 + i * 104:87 + i * 104]
        mask5 = mask[30:95, 3 + i * 104:87 + i * 104]
        cv2.namedWindow('img5', cv2.WINDOW_NORMAL)
        cv2.imshow('img5', img5)

#cv2.line(img3, (83, 0), (83, 95), (255, 0, 0), 1)
#cv2.line(edged2, (83, 0), (83, 95), (255, 0, 0), 1)
#cv2.line(mask2, (83, 0), (83, 95), (255, 0, 0), 1)
#for i in range(1, 2):
  #  i = i*25
 #   img4 = img[30:95, i+0:i+90]


"""
for i in range(0, 7):
    g = i*17
    cv2.line(img3, (g, 0), (g, 700), (255, 0, 0), 1)
    cv2.line(edged2, (g, 0), (g, 700), (255, 0, 0), 1)
    cv2.line(mask2, (g, 0), (g, 700), (255, 0, 0), 1)
for i in range(0, 6):
    g = i*16
    cv2.line(img3, (0, g), (84, g), (255, 0, 0), 1)
    cv2.line(edged2, (0, g), (84, g), (255, 0, 0), 1)
    cv2.line(mask2, (0, g), (84, g), (255, 0, 0), 1)
"""
#cv2.imwrite("a答题卡提取.png", edged2)
#cv2.imwrite("a答题卡提取2.png", mask2)
#cv2.namedWindow('edged221', cv2.WINDOW_NORMAL)
#cv2.imshow('edged221', img8)
#cv2.namedWindow('edged21', cv2.WINDOW_NORMAL)
#cv2.namedWindow('edged22', cv2.WINDOW_NORMAL)
#cv2.namedWindow('edged23', cv2.WINDOW_NORMAL)
#cv2.imshow('edged21', img3)
#cv2.imshow('edged22', edged2)
#cv2.namedWindow('edged23', cv2.WINDOW_NORMAL)
#cv2.imshow('edged23', mask2)

#cv2.namedWindow('img', cv2.WINDOW_NORMAL)
#cv2.imshow('img', img3)
cv2.waitKey(0)
