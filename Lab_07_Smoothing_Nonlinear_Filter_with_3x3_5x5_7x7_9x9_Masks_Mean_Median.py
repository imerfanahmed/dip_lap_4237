import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Kernel sizes
kernels = [3, 5, 7, 9]

# Create subplot
plt.figure(figsize=(15, 10))

for i, k in enumerate(kernels):
    mean = cv2.blur(image, (k, k))
    median = cv2.medianBlur(image, k)
    
    # Original
    plt.subplot(len(kernels), 3, i*3 + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Original (Kernel {k})')
    plt.axis('off')

    # Mean filter
    plt.subplot(len(kernels), 3, i*3 + 2)
    plt.imshow(mean, cmap='gray')
    plt.title(f'Mean ({k}x{k})')
    plt.axis('off')

    # Median filter
    plt.subplot(len(kernels), 3, i*3 + 3)
    plt.imshow(median, cmap='gray')
    plt.title(f'Median ({k}x{k})')
    plt.axis('off')

plt.tight_layout()
plt.show()
