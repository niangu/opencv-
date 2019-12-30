import cv2
import sys
import numpy as np

#imgpath = sys.argv['/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (36).jpeg']
img = cv2.imread('/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (36).jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descroptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))

cv2.namedWindow('sift_keypoints', cv2.WINDOW_NORMAL)
cv2.imshow('sift_keypoints', img)
cv2.waitKey()
cv2.destroyAllWindows()