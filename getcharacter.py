import cv2
import numpy as np
import os

# Load the source image with black shapes
source_image = cv2.imread("test.png")

# Convert the source image to grayscale
source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to extract black shapes
_, binary_image = cv2.threshold(source_gray, 128, 255, cv2.THRESH_BINARY)

# Create a directory to save the extracted shapes*
inverted_image = cv2.bitwise_not(binary_image)

# Find contours in the inverted image
contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas to combine all black shapes
combined_image = np.zeros_like(source_image)
output_dir = "extracted_shapes"
os.makedirs(output_dir, exist_ok=True)

# Process each contour and save as a separate image
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    shape = source_gray[y:y+h, x:x+w]

    # Save each shape in a separate file
    shape_filename = os.path.join(output_dir, f"shape_{i}.jpg")

    cv2.imwrite(shape_filename, shape)

# Display the number of extracted shapes
print(f"Extracted {len(contours)} shapes.")