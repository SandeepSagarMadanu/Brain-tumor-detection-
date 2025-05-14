
# Brain Tumor Detection using Deep Learning

## Overview

This project focuses on detecting brain tumors from MRI images using a convolutional neural network (CNN). The notebook handles the complete pipeline from data preprocessing to model training, evaluation, and prediction. The objective is to accurately classify MRI images as either tumor or no tumor.

## Dataset

The dataset is loaded from a specified directory containing two classes:

- `yes/` — MRI images with tumors
- `no/` — MRI images without tumors

The images are resized and preprocessed for compatibility with the CNN model.

## Preprocessing

- Images are resized to **150x150** pixels.
- Converted to arrays using Keras utilities.
- Labels are encoded (1 for tumor, 0 for no tumor).
- Normalization applied by dividing pixel values by 255.
- Data is split into training and testing sets.

## Data Augmentation

To enhance generalization, the training data is augmented using:

- Rotation
- Zoom
- Shear
- Horizontal flip
- Width and height shifts

Implemented via `ImageDataGenerator`.

## Model Architecture

A Sequential CNN model is built with the following layers:

- Convolutional layers (Conv2D) with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers for classification
- Dropout for regularization

The final layer uses **sigmoid** activation for binary classification.

## Compilation and Training

- **Loss function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- Trained over **20 epochs**

## Evaluation

- Accuracy and loss are plotted for training and validation sets.
- Confusion matrix is generated to evaluate classification performance.
- Classification report includes precision, recall, and F1-score.

## Prediction

- Model is used to predict unseen images.
- Outputs classification results with confidence scores.
- Sample predictions are visualized with corresponding labels.

## Dependencies

Make sure to install the following Python libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

## How to Run

1. Clone the repository or copy the `.ipynb` file.
2. Ensure the dataset is structured into `yes/` and `no/` folders.
3. Open the notebook in Jupyter or Colab.
4. Run all cells sequentially.

## Future Enhancements

- Integrate a web UI for interactive predictions.
- Incorporate YOLOv10 for real-time tumor localization.
- Expand dataset for better generalization.
