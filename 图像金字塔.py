import cv2

img = cv2.imread('/home/niangu/桌面/答题卡识别/1.jpeg', 0)
lower_reso = cv2.pyrUp(img)
higher_reso2 = cv2.pyrUp(lower_reso)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', higher_reso2)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()