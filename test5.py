import cv2
import numpy as np
img = cv2.imread('/home/niangu/桌面/答题卡识别/模板匹配.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 100, 200)
edges2 = cv2.Canny(gray, 100, 200)


cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges',edges)


#显示图像

cv2.namedWindow('adaptive2', cv2.WINDOW_NORMAL)
cv2.imshow('adaptive2',edges2)


cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()


