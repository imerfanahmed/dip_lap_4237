import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('sample_color.JPG', cv2.IMREAD_GRAYSCALE)

# Define different kernel sizes
kernel_sizes = [3, 5, 7, 9]

# Apply average filter for each kernel size
for k in kernel_sizes:
    avg_filtered = cv2.blur(image, (k, k))
    cv2.imwrite(f'avg_filtered_{k}x{k}.jpg', avg_filtered)

# Define a function to create a normalized weighted kernel
def weighted_kernel(size):
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            distance = abs(i - center) + abs(j - center)
            kernel[i, j] = 1 / (1 + distance)
    return kernel / np.sum(kernel)

# Apply weighted filter for each kernel size
for k in kernel_sizes:
    weight = weighted_kernel(k)
    weighted_filtered = cv2.filter2D(image, -1, weight)
    cv2.imwrite(f'weighted_filtered_{k}x{k}.jpg', weighted_filtered)