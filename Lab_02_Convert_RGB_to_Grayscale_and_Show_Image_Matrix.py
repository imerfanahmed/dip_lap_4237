import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the RGB image
image_rgb = cv2.imread('sample_color.jpg')
image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Convert to grayscale
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Show grayscale image
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Print image matrix
print("Grayscale Image Matrix:")
print(image_gray)
