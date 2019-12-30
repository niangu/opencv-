import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (23).jpeg',0)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
#cv2.imwrite('fast_true.png',img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
#cv2.imwrite('fast_false.png',img3)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img2)

cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.imshow('img1', img3)
cv2.waitKey(0)



