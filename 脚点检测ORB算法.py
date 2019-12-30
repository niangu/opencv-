import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (23).jpeg',0)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
#plt.imshow(img2), plt.show()



cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img2)
cv2.waitKey(0)

