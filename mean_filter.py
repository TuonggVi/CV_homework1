import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image
img = cv2.imread('D:/ComputerVision/img.jpg', 0)

# Define kernel
kernel = np.ones((5, 5), np.float32) / 25

# Apply kernel using cv2.filter2D() function
dst = cv2.filter2D(img, -1, kernel)

# Create a single subplot to display both images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original image')
ax[0].axis('off')

# Display the filtered image
ax[1].imshow(dst, cmap='gray')
ax[1].set_title('Mean filter image')
ax[1].axis('off')

plt.show()
