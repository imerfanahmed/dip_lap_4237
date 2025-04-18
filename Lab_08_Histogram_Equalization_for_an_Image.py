import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the input image
image = cv2.imread('sample_color.jpg', 0)  # Read image as grayscale

# Step 2: Calculate the histogram of the input image
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Step 3: Compute the cumulative distribution function (CDF)
cdf = hist.cumsum()

# Step 4: Normalize the CDF to the range [0, 255]
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Step 5: Map the old pixel values to new values
image_equalized = np.interp(image, range(0, 256), cdf_normalized).astype(np.uint8)

# Step 6: Plot histograms and images
plt.figure(figsize=(12, 6))

# Original image and histogram
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram of Original Image')
plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')

plt.show()

# Equalized image and histogram
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Equalized Image')
plt.imshow(image_equalized, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Histogram of Equalized Image')
plt.hist(image_equalized.ravel(), bins=256, range=(0, 256), color='black')

plt.show()

# Save the result
cv2.imwrite('equalized_image.jpg', image_equalized)
