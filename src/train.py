
import os
import pickle
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data_loader import load_dataset, CLASS_NAMES
from src.feature_extraction import extract_hog_features


def train_model():

    # Load dataset
    x_train, y_train, x_test, y_test = load_dataset()

    # Use a subset for faster initial training
    train_size = 10000

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    print("\nExtracting HOG features...")

    # Extract HOG features
    train_features = np.array([
        extract_hog_features(image)
        for image in x_train
    ])

    print("Feature shape:", train_features.shape)

    # Create an SVM pipeline
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        )
    )

    print("\nTraining SVM model...")

    model.fit(train_features, y_train)

    # Save model
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/svm_model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("\nModel trained and saved successfully!")

    return model, x_test, y_test


if __name__ == "__main__":
    train_model()
