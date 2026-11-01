
import tensorflow as tf
import numpy as np

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_dataset():
    """
    Download and load the CIFAR-10 dataset.
    """

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    # Flatten labels from (n, 1) to (n,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()

    print("\nDataset loaded successfully!")
    print("First image label:", CLASS_NAMES[y_train[0]])
