import cv2
import matplotlib.pyplot as plt

# Load a grayscale image
img = cv2.imread('sample_gray.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized = cv2.equalizeHist(img)

# Calculate histograms
hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Plot everything
plt.figure(figsize=(10, 6))

# Original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

# Equalized image
plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized, cmap='gray')
plt.axis('off')

# Original histogram
plt.subplot(2, 2, 3)
plt.title('Original Histogram')
plt.plot(hist_original, color='black')
plt.xlim([0, 256])

# Equalized histogram
plt.subplot(2, 2, 4)
plt.title('Equalized Histogram')
plt.plot(hist_equalized, color='black')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
