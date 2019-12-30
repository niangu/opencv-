import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('/home/niangu/桌面/答题卡识别/test3/webwxgetmsgimg (22).jpeg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#template = cv2.imread('/home/niangu/桌面/答题卡识别/裁剪2.jpeg',0)
template = cv2.imread('/home/niangu/桌面/答题卡识别/test3/剪切.jpeg', 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


cv2.imwrite('模板匹配3.png', img_rgb)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.imshow('edges', img_rgb)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()