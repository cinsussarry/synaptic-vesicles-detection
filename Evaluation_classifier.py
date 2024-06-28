#evaluacion del modelo

import numpy as np
from scipy.spatial.distance import cdist
import os
import cv2


def find_pairs(ground_truth, prediction, threshold=30):
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    
    gt_coords = np.argwhere(ground_truth >= 253)  # Find white pixel coordinates in ground truth
    pred_coords = np.argwhere(prediction >= 253)  # Find white pixel coordinates in prediction
    
    # Iterate over ground truth white pixels
    for gt_coord in gt_coords:
        distances = cdist([gt_coord], pred_coords)  # Compute distances to prediction white pixels
        min_distance = np.min(distances)
        print(min_distance)
        
        if min_distance <= threshold:
            true_positives += 1
            # Remove the corresponding prediction coordinate to avoid pairing it with other ground truth pixels
            pred_coords = np.delete(pred_coords, np.argmin(distances), axis=0)
        else:
            false_negatives += 1
    
    # Any remaining prediction coordinates are false positives
    false_positives = len(pred_coords)
    
    return true_positives, false_positives, false_negatives

# Example usage:
# ground_truth_image and prediction_image are your black and white images represented as numpy arrays
# Make sure both images have the same dimensions
ground_truth_image=cv2.imread("/Users/Milagros/Downloads/ground_truth_imagen0.jpg")
ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
# Convertir la imagen a un array numpy
ground_truth_image = np.array(ground_truth_image)
print(np.max(ground_truth_image))
print(ground_truth_image.shape)

prediction_image=cv2.imread("/Users/Milagros/Downloads/centers_pmap0.jpg")
prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2GRAY)
# Convertir la imagen a un array numpy
prediction_image = np.array(prediction_image)
print(prediction_image.shape)

# Assuming you have loaded your images into numpy arrays ground_truth_image and prediction_image
true_positives, false_positives, false_negatives = find_pairs(ground_truth_image, prediction_image)

print("True Positives:", true_positives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)