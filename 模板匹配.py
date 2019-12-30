import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/niangu/桌面/答题卡识别/webwxgetmsgimg (16).jpeg', 0)
img2 = img.copy()
template = cv2.imread('/home/niangu/桌面/答题卡识别/test2/裁剪2.jpg', 0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
           'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']





for meth in methods:
    img = img2.copy()
    method = eval(meth)

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    """
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', res)
    cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    cv2.imshow('edges', img)

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    """
    plt.subplot(121)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(meth)

    plt.show()
