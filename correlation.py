import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
img = cv2.imread('D:/ComputerVision/img.jpg', 0) 
template = img[100:150, 150:200]

# Get the width and height of the template image 
w, h = template.shape[::-1]

# Perform correlation
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find the location with the maximum correlation value 
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Draw a rectangle around the matched area
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, 255, 2)

# Resize the correlation result image
res = np.abs(res)
res_img = np.uint8(255 * res)
res_img = cv2.resize(res_img, img.shape[::-1])

# Create a single subplot to display both images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image with the matched area highlighted
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display the correlation result
ax[1].imshow(res_img, cmap='gray')
ax[1].set_title('Correlation Result')
ax[1].axis('off')

plt.show()
