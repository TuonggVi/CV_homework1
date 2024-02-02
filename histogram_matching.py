import cv2
import numpy as np
from matplotlib import pyplot as plt

#Load reference and source images
ref_img = cv2.imread('D:/ComputerVision/HW1/photo.jpg', cv2.IMREAD_GRAYSCALE)
src_img = cv2.imread('D:/ComputerVision/img.jpg', cv2.IMREAD_GRAYSCALE)

#Apply histogram equalization to the reference image
ref_img_eq = cv2.equalizeHist(ref_img)
#Calculate the histograms for both the reference and source images
ref_hist = cv2.calcHist([ref_img_eq], [0], None, [256], [0, 256])
src_hist = cv2.calcHist([src_img], [0], None, [256], [0, 256])
#Normalize the histograms
ref_hist_norm = ref_hist / ref_img.size
src_hist_norm = src_hist/src_img.size
#Calculate the cumulative sum of the normalized histograms
ref_cumsum = np.cumsum(ref_hist_norm) 
src_cumsum = np.cumsum(src_hist_norm)
#Map the source image's intensities to the reference image's intensities
lookup_table = np.interp(src_cumsum, ref_cumsum, range(256))
#Apply histogram matching to the source image
matched_img = cv2.LUT(src_img, lookup_table.astype('uint8'))
#Display the original and matched images
plt.subplot(2, 2, 1), plt.imshow(src_img, cmap="gray")
plt.title('Source Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(ref_img, cmap="gray")
plt.title('Reference Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(matched_img, cmap='gray')
plt.title('Matched Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.plot(ref_hist_norm)
plt.plot(src_hist_norm)
plt.title('Histograms'), plt.xlim([0, 256])
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()  