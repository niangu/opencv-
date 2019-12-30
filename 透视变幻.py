#透视变换
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/home/niangu/桌面/答题卡识别/text/111.jpg')
rows, cols, ch = img.shape

#pts1 = np.float32([[120, 105], [940, 105], [120, 1340], [970, 1310]])#找到四个点
#pts2 = np.float32([[0, 0], [1060, 0], [0, 1430], [1060, 1430]])#替换的四个点

pts1 = np.float32([[120, 105], [940, 105], [120, 1340], [970, 1310]])#找到四个点
pts2 = np.float32([[0, 0], [1060, 0], [0, 1430], [1060, 1430]])#替换的四个点

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (1060, 1430))

#cv2.namedWindow('Canny边缘检测', cv2.WINDOW_NORMAL)
cv2.namedWindow('Canny边缘检测2', cv2.WINDOW_NORMAL)


cv2.imshow("Canny边缘检测2", dst)


#cv2.imshow("Canny边缘检测", )


cv2.waitKey(0)