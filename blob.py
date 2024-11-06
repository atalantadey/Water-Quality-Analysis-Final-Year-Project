# Algae Impurity Detection in Images

## Overview
#This Python script uses OpenCV and NumPy to detect algae-like impurities in images,
# specifically targeting circular shapes that resemble algae bodies.
# It processes an input image by defining a region of interest (ROI),
# enhancing the image, applying thresholding, edge detection,
# and blob detection to identify and visualize potential impurities.

#### Description
#1. Image Reading: Loads the image from the specified path.
#2. ROI Definition: Computes the dimensions of the ROI based on the specified fraction, centered in the original image.
#. Grayscale Conversion: Converts the cropped ROI to grayscale for further processing.
#4. Contrast Enhancement: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve image contrast.
#5. Adaptive Thresholding: Uses adaptive thresholding to highlight algae-like bodies in the image.
#6. Edge Detection: Applies the Canny edge detection algorithm to emphasize circular shapes.
#7. Blob Detection: Utilizes `SimpleBlobDetector` to identify circular blobs based on specified parameters such as area, circularity, and inertia.
#8. Contour Filtering: Finds contours in the edge-detected image and filters them based on area and circularity criteria.
#9. Visualization: Draws both detected blobs and valid contours on the original image and displays the result using Matplotlib.

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Code scripted by Atalanta Dey - CSE2021116

def detect_algae_impurities(image_path, roi_fraction=0.99):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return 0  # Return 0 if image is not found

    # Step 2: Get image dimensions and define ROI
    height, width = img.shape[:2]
    roi_height = int(height * roi_fraction)
    roi_width = int(width * roi_fraction)
    start_x = (width - roi_width) // 2
    start_y = (height - roi_height) // 2
    cropped_img = img[start_y:start_y + roi_height, start_x:start_x + roi_width]

    # Step 3: Convert to grayscale and enhance contrast using CLAHE
    grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(grayscale_img)

    # Step 4: Use adaptive thresholding to highlight algae-like bodies
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    # Step 5: Apply edge detection to emphasize circular shapes
    edges = cv2.Canny(adaptive_thresh, 10, 150)

    # Step 6: Use SimpleBlobDetector to find algae-like circular shapes
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 70
    params.maxArea = 9000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced_img)

    # Step 7: Filter contours for large shapes along edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 90
    max_contour_area = 9000
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < max_contour_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.5 < circularity < 1.9:
                valid_contours.append(cnt)

    # Step 8: Draw both detected blobs and valid contours for visualization
    output_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, valid_contours, -1, (0, 255, 0), 2)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output_img, (x, y), int(kp.size / 2), (0, 255, 0), 2)

    # Step 9: Place the cropped output back onto the original image for context
    img_with_contours = img.copy()
    img_with_contours[start_y:start_y + roi_height, start_x:start_x + roi_width] = output_img

    # Display the result
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Return the impurity count
    return len(keypoints) + len(valid_contours)


# List to store impurity counts
impurity_counts = []

# Loop through the 10 water samples
for i in range(1, 11):
    image_path = f'water_samples/river_water_{i}.jpg'  # Construct the image path
    impurity_count = detect_algae_impurities(image_path)
    impurity_counts.append(impurity_count)  # Store the impurity count

    # Determine if the water is drinkable
    if impurity_count < 20:
        drinkability = "Drinkable"
    else:
        drinkability = "Not Drinkable"

    print(f"Number of algae-like impurities detected in river_water_{i}.jpg: {impurity_count} - {drinkability}")

# Calculate average impurity count
average_impurity = np.mean(impurity_counts)
print(f"\nAverage number of algae-like impurities detected: {average_impurity:.2f}")

# Determine if the average impurity level is drinkable
if average_impurity < 20:
    overall_drinkability = "Drinkable"
else:
    overall_drinkability = "Not Drinkable"

print(f"Overall water quality based on average impurities: {overall_drinkability}")
