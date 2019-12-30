import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (23).jpeg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
cv2.waitKey(0)