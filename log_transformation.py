import cv2
import numpy as np
#Load an image
img = cv2.imread('D:\ComputerVision\img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Log transformation
c = 255/np.log(1 + np.max(img))
log_transformed = c* (np.log(img + 1))
log_transformed = np. uint8(log_transformed)
#Show two images
img_concat = cv2.hconcat([img, log_transformed])
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.imshow("Images",img_concat)
cv2.waitKey(0)
cv2.destroyAllWindows()