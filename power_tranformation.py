import cv2
import numpy as np
#Load an image
img = cv2.imread('D:\ComputerVision\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#power transformtaion
c = 1.7
Result = 255/pow(255,c)*np.power (img, c)
Result = np. uint8(Result)
#Show two images
img_concat = cv2.hconcat([img, Result])
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.imshow("Images", img_concat)
cv2.imwrite("photo.jpg", Result)
cv2.waitKey(0)
cv2.destroyAllwindows()