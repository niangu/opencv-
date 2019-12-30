import cv2
import numpy as np

img = cv2.imread('/home/niangu/桌面/答题卡识别/webwxgetmsgimg (16).jpeg',cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 127, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, 1, 2)

#proimage2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 7)
contours, cnt = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取所有的轮廓

cnt = contours[0]
M = cv2.moments(cnt)#将计算得到的矩以一个字典的形式返回
print(M)
#根据这些距的值计算出对象的重心
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("中心：",cx, cy)
area = cv2.contourArea(cnt)
print("面积:", area)
perimeter = cv2.arcLength(cnt, True)#True指定对象的形状是闭合的
print("轮廓周长：", perimeter)

#轮廓近似
#epsilon = 0.1*cv2.arcLength(cnt, True)
#approx = cv2.approxPolyDP(cnt, epsilon, True)

#轮廓的性质
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print("边界矩形的宽高比：", aspect_ratio)
rect_area = w*h
extent = float(area)/rect_area#轮廓面积与边界矩形面积的比
#(x, y), (MA, ma), angle = cv2.fitEllipse(cnt)#对象的方向换会返回长轴和短轴的长度
#print("方向：", "(", x, y, ")", "(", MA, ma, ")")
#掩模和像素点
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)#一定要使用参数-1,绘制填充的轮廓
pixelpoints = np.transpose(np.nonzero(mask))
print("淹没", pixelpoints)
#最大值，最小值以及它们的位值
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img, mask=mask)
print("最大值，最小值以及它们的位值:", min_val, max_val, min_loc, max_loc)
#平均颜色和灰度
mean_val = cv2.mean(img, mask=mask)
print("平均颜色和灰度", mean_val)
#一个对象的最上面，最下面，最左边，最右边的点
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
print("一个对象的最上面，最下面，最左边，最右边的点", leftmost, rightmost, topmost, bottommost)


cv2.namedWindow('approx', cv2.WINDOW_NORMAL)

cv2.imshow('approx', img)
cv2.waitKey(0)&0xFF
#cv2.destroyAllWindows()

