# Image Classification and Image Processing using OpenCV

## Overview

This project implements a traditional machine learning-based image classification system using OpenCV and Support Vector Machine (SVM).

The project demonstrates image preprocessing, image enhancement, feature extraction, model training, and evaluation using the CIFAR-10 dataset.

## Objectives

- Perform image preprocessing using OpenCV.
- Apply brightness and contrast adjustments.
- Demonstrate Gaussian blur, median filtering, and edge detection.
- Extract image features using Histogram of Oriented Gradients (HOG).
- Train an SVM image classification model.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

## Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow
- Scikit-learn
- Scikit-image
- Matplotlib

## Dataset

The CIFAR-10 dataset contains 60,000 color images divided into 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

The dataset is downloaded automatically using TensorFlow.

## Methodology

### 1. Image Preprocessing

Images are converted from RGB to grayscale, resized to 32 × 32 pixels, and normalized.

### 2. Image Enhancement

The following techniques are demonstrated:

- Brightness and contrast adjustment
- Gaussian blur
- Median filtering
- Canny edge detection

### 3. Feature Extraction

Histogram of Oriented Gradients (HOG) is used to extract edge and gradient information from images.

### 4. Model Training

A Linear Support Vector Machine classifier is trained using the extracted HOG features.

### 5. Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Project Structure

```text
image-classification-opencv/
├── dataset/
├── outputs/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/image-classification-opencv.git

cd image-classification-opencv

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Execution

```bash
python src/data_loader.py

python main.py

python src/train.py

python src/evaluate.py

python src/predict.py
```

## Results

The trained model is evaluated on a held-out subset of CIFAR-10 test images.

| Metric | Result |
|---|---|
| Training images | 10,000 |
| Testing images | 2,000 |
| Feature extraction | HOG |
| Classifier | Linear SVM |
| Accuracy | Add your measured result |
| F1-score | Add your measured result |

## Output Visualizations

- `outputs/image_processing.png`
- `outputs/confusion_matrix.png`
- `outputs/predictions.png`

## Conclusion

This project demonstrates how traditional machine learning algorithms can be applied to image classification using OpenCV for preprocessing and HOG for feature extraction.

It also provides practical experience in image enhancement, feature engineering, model evaluation, and visualization.

## AUTHOR
Ebin Reji - www.ebinreji.online
