import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('/home/niangu/桌面/answer_sheet/model/answer_model.png', 0)
img2 = cv2.imread("/home/niangu/桌面/答题卡识别/test2/1724315007.jpg", 0)

#img1 =cv2.imread('/home/niangu/桌面/答题卡识别/test3/Mwebwxgetmsgimg (27).jpeg', 0)
#img2 = cv2.imread('/home/niangu/桌面/答题卡识别/test3/学号匹配2.jpeg', 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)


    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    dst2 = np.float32([[0, 700], [400, 700], [400, 0], [0, 0]])
    m = cv2.getPerspectiveTransform(dst, dst2)
    result = cv2.warpPerspective(img2, m, (400, 700))
    print("dst", dst)#img2图像位置
    print(m)
    #print("", )
    cv2.imwrite("a答题卡特征提取检测24.png", result)
    cv2.namedWindow('keypoints1vvvvvv', cv2.WINDOW_NORMAL)
    cv2.imshow('keypoints1vvvvvv', result)

    cv2.polylines(img2, [np.int32(dst)], True, 255, 10, cv2.LINE_AA)
else:
    print("Not enough matches are found -%d/%d"%(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#plt.imshow(img3, 'gray')
#plt.show()
cv2.namedWindow('imgvvvvvv', cv2.WINDOW_NORMAL)
cv2.imshow('imgvvvvvv', img3)
#cv2.imwrite("")

cv2.waitKey(0)






















