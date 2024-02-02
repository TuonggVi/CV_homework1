import cv2
import numpy as np

# Load an image
img = cv2.imread('D:/ComputerVision/img.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

trans = np.zeros_like(img)

def piecewise_linear_transform(x):
    if x < 100:
        return (30/100) * x
    elif x < 155:
        return (128/35) * (x - 100) + 3
    else:
        return (33/100) * (x - 155) + 189

# Apply piecewise linear transform
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        trans[i, j] = piecewise_linear_transform(img[i, j])

# Show two images
img_concat = cv2.hconcat([img, trans])
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.imshow("Images", img_concat)

cv2.waitKey(0)
cv2.destroyAllWindows()  
