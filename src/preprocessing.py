
import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess an image for classification.

    Steps:
    1. Convert RGB to grayscale
    2. Resize to 32x32
    3. Normalize pixel values
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize image
    resized = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixel values to range 0-1
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def adjust_brightness_contrast(image, alpha=1.5, beta=30):
    """
    alpha = contrast factor
    beta = brightness factor
    """

    enhanced = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )

    return enhanced


def apply_gaussian_blur(image):
    """
    Apply Gaussian smoothing to reduce noise.
    """

    return cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )


def apply_median_filter(image):
    """
    Apply median filtering to remove impulse noise.
    """

    return cv2.medianBlur(image, 5)


def apply_edge_detection(image):
    """
    Detect edges using the Canny algorithm.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        gray,
        threshold1=100,
        threshold2=200
    )

    return edges


if __name__ == "__main__":
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    processed = preprocess_image(image)

    print("Processed image shape:", processed.shape)
    print("Data type:", processed.dtype)
