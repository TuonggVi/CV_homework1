import cv2
import numpy as np
from matplotlib import pyplot as plt

#Load an image
img = cv2.imread('D:/ComputerVision/img.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Calculate histogram and show histogram function
def show_histogram(x):
    hist, bins = np.histogram(x.flatten(), 256, [0, 256])
    # Plot histogram
    plt.hist(x.flatten(), 256, [0, 256], color='r')
    #Set x-axis and y-axis labels
    plt.xlim([0, 256])
    plt.ylim([0, 5000])
    plt.xlabel('Pixel Values')
    plt.ylabel('Number of Pixels')

plt.subplot(2, 2, 1)
show_histogram(img)

equ = cv2.equalizeHist(img)
plt.subplot(2,2,2)
show_histogram(equ)

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image', y= -0.15)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image', y= -0.15)
plt.axis('off')

plt.show()

# Show two images side by side
img_concat = np.concatenate((img, equ), axis=1)
cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
cv2.imshow("Images", img_concat)

cv2.waitKey(0)
cv2.destroyAllWindows()  