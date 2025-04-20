# Import Required Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load Image using OpenCV
img = cv2.imread('sample_color.jpg')  # Replace with your image path

# Convert BGR to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Read image using PIL
img_pil = Image.open('sample_color.jpg')

# Convert to Grayscale using OpenCV
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prepare display
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Image Display using Different Libraries', fontsize=16)

# Original Image (OpenCV - BGR converted to RGB)
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image (RGB)')
axes[0, 0].axis('off')

# Grayscale Image
axes[0, 1].imshow(gray_img, cmap='gray')
axes[0, 1].set_title('Grayscale Image (OpenCV)')
axes[0, 1].axis('off')

# Image using PIL
axes[1, 0].imshow(img_pil)
axes[1, 0].set_title('Image (PIL)')
axes[1, 0].axis('off')

# PIL as NumPy Array
axes[1, 1].imshow(np.asarray(img_pil))
axes[1, 1].set_title('PIL as NumPy Array')
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
plt.show()
