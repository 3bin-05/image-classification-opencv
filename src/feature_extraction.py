
import cv2
import numpy as np

from skimage.feature import hog


def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients (HOG)
    features from an image.
    """

    # Convert RGB to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to a fixed size
    resized = cv2.resize(gray, (32, 32))

    # Extract HOG features
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )

    return features
