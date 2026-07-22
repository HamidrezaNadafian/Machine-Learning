# Kaggle Digit Recognizer - PyTorch CNN

## Overview
This repository contains a PyTorch-based Convolutional Neural Network (CNN) designed for the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition. The model effectively classifies 28x28 grayscale images of handwritten digits (0-9) and achieves a highly competitive accuracy.

**Kaggle Public Score:** `0.99232`

## Dataset & Preprocessing
* **Data:** The dataset consists of MNIST handwritten digits provided by Kaggle (`train.csv` and `test.csv`).
* **Reshaping:** Tabular pixel data was reshaped to `(-1, 1, 28, 28)` to be compatible with PyTorch's 2D convolutional layers.
* **Normalization:** Pixel values were scaled down from `[0, 255]` to `[0.0, 1.0]`.
* **Validation Split:** 20% of the training data was partitioned for validation using `scikit-learn` to monitor generalization.

## Model Architecture
The `DigitClassifierCNN` is a deep convolutional neural network structurally designed to extract hierarchical spatial features while mitigating overfitting.

* **Feature Extraction (Convolutional Blocks):**
  * **Block 1:** Conv2D(1 → 32) → BatchNorm → ReLU → Conv2D(32 → 64) → BatchNorm → ReLU → MaxPool2D(2x2) → Dropout(0.25)
  * **Block 2:** Conv2D(64 → 128) → BatchNorm → ReLU → Conv2D(128 → 128) → BatchNorm → ReLU → MaxPool2D(2x2) → Dropout(0.25)
* **Classification (Fully Connected Layers):**
  * Flatten → Linear(128 * 7 * 7 → 128) → ReLU → Dropout(0.4) → Linear(128 → 10)

## Training Setup
* **Optimizer:** AdamW (`lr=0.001`) - utilized for robust weight decay and smooth convergence.
* **Loss Function:** CrossEntropyLoss.
* **Batch Size:** 64
* **Epochs:** 10
* **Regularization:** Gradient clipping (`max_norm=1.0`) was applied to prevent exploding gradients and ensure stable training, acting in tandem with Batch Normalization and Dropout layers.
* **Hardware:** Configured to dynamically allocate to GPU (`cuda`) if available, falling back to `cpu`.

## Requirements
* Python 3.x
* `torch`, `torchvision`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`

## Usage
1. Ensure `train.csv` and `test.csv` are in the project root directory.
2. Execute the notebook to initiate the training loop.
3. The pipeline automatically evaluates validation accuracy per epoch and exports final predictions to `submission.csv` for Kaggle scoring.
