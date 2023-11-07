import cv2
import numpy as np
import os

source_image = cv2.imread("lettre.webp")
source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(source_gray, 128, 255, cv2.THRESH_BINARY)
inverted_image = cv2.bitwise_not(binary_image)
contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
combined_image = np.zeros_like(source_image)
output_dir = "extracted_shapes"
os.makedirs(output_dir, exist_ok=True)

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    shape = source_gray[y:y+h, x:x+w]
    shape_filename = os.path.join(output_dir, f"letter_{i}.jpg")
    cv2.imwrite(shape_filename, shape)

print(f"Extracted {len(contours)} shapes.")