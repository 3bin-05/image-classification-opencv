
import cv2
import matplotlib.pyplot as plt

from src.data_loader import load_dataset, CLASS_NAMES
from src.preprocessing import (
    preprocess_image,
    adjust_brightness_contrast,
    apply_gaussian_blur,
    apply_median_filter,
    apply_edge_detection
)


def demonstrate_image_processing():

    # Load one sample image
    x_train, y_train, _, _ = load_dataset()

    image = x_train[0]

    # Apply processing techniques
    enhanced = adjust_brightness_contrast(image)

    blurred = apply_gaussian_blur(image)

    median = apply_median_filter(image)

    edges = apply_edge_detection(image)

    processed = preprocess_image(image)

    # Display results
    plt.figure(figsize=(15, 8))

    images = [
        image,
        enhanced,
        blurred,
        median,
        edges,
        processed
    ]

    titles = [
        "Original Image",
        "Brightness & Contrast",
        "Gaussian Blur",
        "Median Filter",
        "Canny Edge Detection",
        "Grayscale & Normalized"
    ]

    for i in range(len(images)):

        plt.subplot(2, 3, i + 1)

        if i == 5:
            plt.imshow(images[i], cmap="gray")
        elif i == 4:
            plt.imshow(images[i], cmap="gray")
        else:
            plt.imshow(images[i])

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "outputs/image_processing.png",
        dpi=300
    )

    plt.show()

    print("Image class:", CLASS_NAMES[y_train[0]])


if __name__ == "__main__":

    demonstrate_image_processing()

    print("\nImage processing demonstration completed.")
