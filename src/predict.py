
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_dataset, CLASS_NAMES
from src.feature_extraction import extract_hog_features


def display_predictions():

    # Load test dataset
    _, _, x_test, y_test = load_dataset()

    # Load trained model
    with open("outputs/svm_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Select 10 random test images
    np.random.seed(42)

    indices = np.random.choice(
        len(x_test),
        size=10,
        replace=False
    )

    images = x_test[indices]
    actual_labels = y_test[indices]

    # Extract features
    features = np.array([
        extract_hog_features(image)
        for image in images
    ])

    # Predict
    predictions = model.predict(features)

    # Display results
    plt.figure(figsize=(15, 6))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(images[i])

        actual = CLASS_NAMES[actual_labels[i]]
        predicted = CLASS_NAMES[predictions[i]]

        color = "green" if actual == predicted else "red"

        plt.title(
            f"Actual: {actual}\nPredicted: {predicted}",
            color=color,
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        "outputs/predictions.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    display_predictions()
