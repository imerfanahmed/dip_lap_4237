import cv2
import numpy as np
import matplotlib.pyplot as plt

#read original image in rgb format
image = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)


# a) Negative Image
negative_img = 255 - image

# b) Brightness Enhancement
brightness_value = 50
bright_img = cv2.add(image, brightness_value)

# c) Log Transformation
c = 255 / np.log(1 + np.max(image))
log_img = c * (np.log(1 + image.astype(np.float32)))
log_img = np.array(log_img, dtype=np.uint8)

# d) Power-law (Gamma) Transformation
gamma = 2.0
gamma_img = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# e) Histogram Equalization
contrast_img = cv2.equalizeHist(image)

# Display Results
titles = ['Original', 'Negative', 'Brightness', 'Log', 'Gamma', 'Contrast']
images = [image, negative_img, bright_img, log_img, gamma_img, contrast_img]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
