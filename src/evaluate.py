
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data_loader import load_dataset, CLASS_NAMES
from src.feature_extraction import extract_hog_features


def evaluate_model():

    # Load dataset
    _, _, x_test, y_test = load_dataset()

    # Use 2000 test images initially
    test_size = 2000

    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

    # Load trained model
    with open("outputs/svm_model.pkl", "rb") as file:
        model = pickle.load(file)

    print("Extracting test features...")

    test_features = np.array([
        extract_hog_features(image)
        for image in x_test
    ])

    # Predict labels
    predictions = model.predict(test_features)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    # Classification report
    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(10),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(10)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.title("CIFAR-10 SVM Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "outputs/confusion_matrix.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    evaluate_model()
