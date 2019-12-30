"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img  = cv2.imread('/home/niangu/桌面/答题卡识别/test4/webwxgetmsgimg (33).jpeg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 450, 290)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:, :, np.newaxis]
cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
cv2.imshow('edged', img)
cv2.waitKey(0)
"""

