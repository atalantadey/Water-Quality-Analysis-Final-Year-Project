import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def detect_algae_impurities(image_path, roi_fraction=0.99):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}.")
        return 0  # Return 0 if image is not found

    # Step 2: Get image dimensions and define ROI
    height, width = img.shape[:2]
    roi_height = int(height * roi_fraction)
    roi_width = int(width * roi_fraction)
    start_x = (width - roi_width) // 2
    start_y = (height - roi_height) // 2
    cropped_img = img[start_y:start_y + roi_height, start_x:start_x + roi_width]

    # Step 3: Convert to grayscale
    grayscale_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Gaussian blur to reduce high-frequency noise
    blurred_img = cv2.GaussianBlur(grayscale_img, (7, 7), 0)

    # Step 5: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(blurred_img)

    # Step 6: Apply median filter for further noise reduction
    denoised_img = cv2.medianBlur(enhanced_img, 5)

    # Step 7: Adaptive threshold to detect darker particles
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 12)

    # Step 8: Morphological transformations to clean small noise
    kernel = np.ones((4, 4), np.uint8)
    morph_open = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 9: Edge detection
    edges = cv2.Canny(morph_close, 30, 100)

    # Step 10: Blob detection for algae-like shapes with irregular edges
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100  # Reduced min area to capture relevant particles
    params.maxArea = 9000
    params.filterByCircularity = False  # Disabled to allow irregular shapes
    params.filterByConvexity = True
    params.minConvexity = 0.6  # Adjusted to filter irregular yet significant particles
    params.filterByInertia = False  # Disabled for irregular shapes

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(denoised_img)

    # Step 11: Filter contours for darker, irregular shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 200  # Moderately low min area to capture darker particles
    max_contour_area = 9000
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_contour_area < area < max_contour_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.4 < circularity < 1.9:  # Broaden circularity for darker, irregular shapes
                valid_contours.append(cnt)

    # Step 12: Draw both detected blobs and valid contours for visualization
    output_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, valid_contours, -1, (0, 255, 0), 2)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output_img, (x, y), int(kp.size / 2), (0, 255, 0), 2)

    # Step 13: Place the cropped output back onto the original image for context
    img_with_contours = img.copy()
    img_with_contours[start_y:start_y + roi_height, start_x:start_x + roi_width] = output_img

    # Display the result
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Return the impurity count
    return len(keypoints) + len(valid_contours)


# List of sample directories
sample_directories = {
    "River Water": 'water_samples/river_water',
    "Drinking Water": 'water_samples/drinking_water'
}

# Process each sample type
for sample_type, path in sample_directories.items():
    impurity_counts = []
    print(f"\nProcessing {sample_type} samples...")

    for i in range(1, 11):
        image_path = f'{path}_{i}.jpg'  # Construct the image path
        impurity_count = detect_algae_impurities(image_path)
        impurity_counts.append(impurity_count)  # Store the impurity count

        # Determine if the water is drinkable
        drinkability = "Drinkable" if impurity_count < 20 else "Not Drinkable"
        print(f"Number of algae-like impurities detected in {sample_type}_{i}.jpg: {impurity_count} - {drinkability}")

    # Calculate average impurity count for this sample type
    average_impurity = np.mean(impurity_counts)
    print(f"\nAverage number of algae-like impurities detected in {sample_type}: {average_impurity:.2f}")

    # Determine if the average impurity level is drinkable
    overall_drinkability = "Drinkable" if average_impurity < 20 else "Not Drinkable"
    print(f"Overall water quality based on average impurities for {sample_type}: {overall_drinkability}")

#Final Script

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



# Code scripted by Atalanta Dey - CSE2021116