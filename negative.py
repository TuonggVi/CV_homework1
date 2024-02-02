import cv2
import numpy as np
#Load an image
img = cv2.imread('D:\ComputerVision\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nega = np.zeros_like(img)
nega = 255 - img
img_concat = cv2.hconcat([img, nega])
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.imshow("Images", img_concat)

cv2.waitKey(0)
cv2.destroyAllwindows()
