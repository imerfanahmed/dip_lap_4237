import cv2
import matplotlib.pyplot as plt

# Load grayscale image
gray_image = cv2.imread('sample_color.jpg', cv2.IMREAD_GRAYSCALE)

# Convert grayscale to RGB
rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

# Show the RGB image
plt.imshow(rgb_image)
plt.title('Grayscale to RGB Image')
plt.axis('off')
plt.show()

# Show the RGB image matrix
print("RGB Image Matrix:")
print(rgb_image)
