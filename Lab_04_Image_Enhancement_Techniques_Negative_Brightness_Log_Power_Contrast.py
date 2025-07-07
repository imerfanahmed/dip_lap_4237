import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Apply enhancements
negative = 255 - img
bright = cv2.add(img, 50)
# formula will be s = c* log(1 + r)

log = 50*np.log(1+img.astype(np.float32))  # log1p is log(1 + x)


gamma = np.uint8(255 * (img / 255) ** 2)
contrast = cv2.equalizeHist(img)

# Titles and images
titles = ['Original', 'Negative', 'Brightness', 'Log', 'Gamma', 'Contrast']
images = [img, negative, bright, log, gamma, contrast]

# Show all in one window
plt.figure(figsize=(12, 8))
for i, (title, image) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
