import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)
kernels = [3, 5, 7, 9]

# Function to create normalized weighted kernel
def weighted_kernel(k):
    center = k // 2
    kernel = np.array([[1 / (1 + abs(i - center) + abs(j - center)) for j in range(k)] for i in range(k)], dtype=np.float32)
    return kernel / kernel.sum()

# Prepare plot
plt.figure(figsize=(15, 8))
plt.suptitle('Original, Average & Weighted Filters', fontsize=18)

for i, k in enumerate(kernels):
    avg = cv2.blur(image, (k, k))
    weight = weighted_kernel(k)
    weighted = cv2.filter2D(image, -1, weight)

    # Show Original (once per row, or just for reference)
    if i == 0:
        plt.subplot(3, len(kernels), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

    # Average filtered
    plt.subplot(3, len(kernels), len(kernels) + i + 1)
    plt.imshow(avg, cmap='gray')
    plt.title(f'Avg {k}x{k}')
    plt.axis('off')

    # Weighted filtered
    plt.subplot(3, len(kernels), 2 * len(kernels) + i + 1)
    plt.imshow(weighted, cmap='gray')
    plt.title(f'Weighted {k}x{k}')
    plt.axis('off')

plt.tight_layout()
plt.show()
