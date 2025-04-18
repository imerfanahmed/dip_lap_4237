import cv2
import numpy as np
import math

# Function to calculate Mean Squared Error
def calculate_mse(original, distorted):
    return np.mean((original - distorted) ** 2)

# Function to calculate Peak Signal-to-Noise Ratio
def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel / math.sqrt(mse))

# Load the original image in grayscale
original = cv2.imread('sample_gray.jpg', cv2.IMREAD_GRAYSCALE)

# Enhanced/distorted versions from Lab 04
negative = 255 - original
bright = cv2.add(original, 50)
log_image = np.array(255 / np.log(1 + np.max(original)) * np.log(1 + original.astype(np.float32)), dtype=np.uint8)
gamma = np.array(255 * (original / 255) ** 2.0, dtype='uint8')
contrast = cv2.equalizeHist(original)

# Store all images and labels
enhanced_images = {
    'Negative': negative,
    'Brightness': bright,
    'Log': log_image,
    'Gamma': gamma,
    'Contrast': contrast
}

# Output MSE and PSNR for each image
print(f"{'Image':<12}{'MSE':<15}{'PSNR (dB)'}")
print("-" * 35)
for name, img in enhanced_images.items():
    mse = calculate_mse(original, img)
    psnr = calculate_psnr(mse)
    print(f"{name:<12}{mse:<15.2f}{psnr:.2f}")
