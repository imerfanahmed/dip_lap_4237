import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure

# Load an image using OpenCV
image_cv = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Display the original image using OpenCV
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image (OpenCV)")
plt.imshow(image_cv, cmap='gray')
plt.axis('off')

# Image manipulation with OpenCV: Apply Gaussian blur
image_blur = cv2.GaussianBlur(image_cv, (5, 5), 0)

# Display the blurred image
plt.subplot(1, 2, 2)
plt.title("Blurred Image (OpenCV)")
plt.imshow(image_blur, cmap='gray')
plt.axis('off')
plt.show()

# Load the image using Pillow
image_pil = Image.open('input_image.jpg')

# Display the image using Pillow
plt.figure(figsize=(6, 6))
plt.title("Original Image (Pillow)")
plt.imshow(image_pil)
plt.axis('off')
plt.show()

# Convert the image to grayscale using Pillow
image_pil_gray = image_pil.convert('L')

# Display the grayscale image
plt.figure(figsize=(6, 6))
plt.title("Grayscale Image (Pillow)")
plt.imshow(image_pil_gray, cmap='gray')
plt.axis('off')
plt.show()

# Image manipulation with scikit-image: Histogram Equalization
image_skimage = exposure.equalize_hist(image_cv)

# Display the original image and the equalized image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image (scikit-image)")
plt.imshow(image_cv, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Equalized Image (scikit-image)")
plt.imshow(image_skimage, cmap='gray')
plt.axis('off')

plt.show()
