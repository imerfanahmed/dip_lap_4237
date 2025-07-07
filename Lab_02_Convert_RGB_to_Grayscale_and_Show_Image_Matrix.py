import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in RGB
img_rgb = cv2.cvtColor(cv2.imread('sample_color.jpg'), cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Show both images side by side
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original (RGB)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')
plt.tight_layout()
plt.show()

# Print grayscale matrix
print("Grayscale Image Matrix:\n", img_gray)
