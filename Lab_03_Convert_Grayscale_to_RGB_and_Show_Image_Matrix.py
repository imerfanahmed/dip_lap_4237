import cv2
import matplotlib.pyplot as plt

# Load grayscale image
gray = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Convert grayscale to 3-channel RGB
rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Show image
plt.imshow(rgb)
plt.title('Grayscale → RGB')
plt.axis('off')
plt.show()

# Print matrix
print("RGB Image Matrix:\n", rgb)
