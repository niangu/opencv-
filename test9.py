from imutils.perspective import four_point_transform
from imutils import contours
import cv2 as cv

ANSWER_KEY_SCORE = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

ANSWER_KEY = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
img = cv.imread('/home/niangu/桌面/答题卡识别/test3/aaa.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)

#cv.imshow("gaussian", gaussian_bulr)

edged = cv.Canny(gaussian_bulr, 75, 200)





cv.namedWindow('原图', cv.WINDOW_NORMAL)
cv.imshow("原图", img)



cv.waitKey(0)