import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
img = cv2.imread('D:/ComputerVision/img.jpg', 0)
# Apply median filter
kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
sharp = cv2.filter2D(img, -1, kernel)
w,h = img.shape
sharp = cv2.resize(sharp, (h,w))

# Create a single subplot to display both images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')

# Display the image after median filter
ax[1].imshow(sharp, cmap='gray')
ax[1].set_title('Result image')
ax[1].axis('off')

plt.show()
