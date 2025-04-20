import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread('sample_color.JPG', cv2.IMREAD_GRAYSCALE)

# Define different kernel sizes
kernel_sizes = [3, 5, 7, 9]

# Store results for displaying later
avg_results = []
weighted_results = []

# Apply average filter for each kernel size
for k in kernel_sizes:
    avg_filtered = cv2.blur(image, (k, k))
    avg_results.append((f'Avg {k}x{k}', avg_filtered))
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
    weighted_results.append((f'Weighted {k}x{k}', weighted_filtered))
    cv2.imwrite(f'weighted_filtered_{k}x{k}.jpg', weighted_filtered)

# Display original and all filtered images
fig, axes = plt.subplots(3, len(kernel_sizes), figsize=(16, 8))
fig.suptitle('Filtering Comparison', fontsize=18)

# Show original image in first row
for i, k in enumerate(kernel_sizes):
    if i == 0:
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].set_title('Original')
    else:
        axes[0, i].axis('off')

# Show average filtered images
for i, (title, img) in enumerate(avg_results):
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].set_title(title)
    axes[1, i].axis('off')

# Show weighted filtered images
for i, (title, img) in enumerate(weighted_results):
    axes[2, i].imshow(img, cmap='gray')
    axes[2, i].set_title(title)
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()
