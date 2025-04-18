import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load original image in grayscale
image = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Define kernel sizes
kernel_sizes = [3, 5, 7, 9]

# Apply Mean and Median filters for each kernel size
for k in kernel_sizes:
    mean_filtered = cv2.blur(image, (k, k))
    median_filtered = cv2.medianBlur(image, k)

    # Display results
    plt.figure(figsize=(10, 4))
    plt.suptitle(f'Kernel Size: {k}x{k}', fontsize=14)
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mean_filtered, cmap='gray')
    plt.title(f'Mean Filter ({k}x{k})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(median_filtered, cmap='gray')
    plt.title(f'Median Filter ({k}x{k})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
