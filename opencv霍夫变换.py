import cv2
import numpy as np

img = cv2.imread('/home/niangu/桌面/opencv-python/canny.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#ret1,th1 = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1),(x2, y2),(0,0,255),2)

#cv2.imwrite('houghlines3.jpg',img)
cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img3', img)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
minLineLength =100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi/180,100, minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
#cv2.imwrite('houghlines5.jpg', img)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.imshow('img2', img)

cv2.waitKey(0)&0xFF
cv2.destroyAllWindows()