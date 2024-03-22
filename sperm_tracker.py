import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/content/OpticalFlow_ss2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Get the 3 largest contours
largest_contours = contours[:3]

# Draw the 3 largest contours on the image
cv2.drawContours(image, largest_contours, -1, (0, 255, 0), 2)

# Calculate and display the areas and centroids of the 3 largest contours
for i, contour in enumerate(largest_contours):
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)

# Convert the BGR image to RGB (for displaying with matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()