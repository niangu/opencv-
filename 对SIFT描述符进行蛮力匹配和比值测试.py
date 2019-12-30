import numpy as np
import cv2
from matplotlib import as plt

img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box in scene.png', 0)

sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
