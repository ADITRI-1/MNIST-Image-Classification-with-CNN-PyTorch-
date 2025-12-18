# MNIST-Image-Classification-with-CNN-PyTorch-
## Overview
This project builds and evaluates a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset. The MNIST dataset is loaded strictly via torchvision.datasets.MNIST. The workflow covers dataset exploration, CNN design, training, and detailed evaluation.

The final deliverables include:

* A trained model checkpoint

* An evaluation report with accuracy, precision, recall, F1-score, and analysis plots

## Tech Stack

* Python

* PyTorch

* torchvision

* NumPy

* Matplotlib

## Dataset

MNIST is loaded using:
       
       * torchvision.datasets.MNIST

The dataset consists of:

  * 60,000 training images
  * 10,000 test images

    Each image is grayscale, size 28x28, and belongs to one of 10 digit classes (0–9).

## Project Workflow
#### 1. Data Loading and Exploration

* Loaded MNIST using torchvision.datasets.MNIST.

* Displayed sample images with their labels.

* Computed class distribution for train/test data.

* Visualized distribution using plots (bar chart / histogram).

#### 2. CNN Model

* Implemented a CNN with:

* At least two convolution layers

* Activation functions (e.g., ReLU)

* Pooling layers (e.g., MaxPool)

* At least one fully connected layer before output

* Output layer with 10 neurons

* Weight initialization (e.g., Kaiming/He initialization for conv/linear layers)

#### 3. Training

* Created data loaders for training and test sets.

* Defined:
    * Loss function (e.g., CrossEntropyLoss)
    * Optimizer (e.g., Adam / SGD)

* Training loop for 15+ epochs:
    * Tracks train loss/accuracy
    * Tracks validation loss/accuracy

* Plotted:
    * Train vs validation loss curves
    * Train vs validation accuracy curves

#### 4. Evaluation & Analysis

Evaluated the trained model on the test set:
     * Accuracy
     * Precision, Recall, F1-score (per-class + macro/weighted averages)

Analysis:
    * Confusion matrix visualization
    * Most commonly misclassified digits
     * At least one improvement suggestion (e.g., batch norm, dropout, deeper CNN, LR scheduling, augmentation)

## How to Run
#### 1. Clone the Repository: 
      git clone <repository-url>
      cd mnist-cnn-pytorch

#### 2. Install Dependencies
    pip install torch torchvision matplotlib numpy scikit-learn


Note: scikit-learn is used only for evaluation metrics (precision/recall/F1 + confusion matrix).

#### 3. Train the Model
    python train.py


This will:

* Download MNIST automatically (if not present)

* Train the CNN for the configured epochs

* Save the trained model checkpoint

#### 4. Evaluate the Model
    python evaluate.py


This will:

* Load the saved model

* Evaluate on the MNIST test set

* Print metrics and generate plots (including confusion matrix)

## Outputs

After running training + evaluation, you should have:

* Model checkpoint
  
      models/mnist_cnn.pth (or similar)

* Training curves

      outputs/loss_curve.png

      outputs/accuracy_curve.png

* Evaluation artifacts

      outputs/confusion_matrix.png

     * Printed classification report (precision/recall/F1)

### Recommended Project Structure
     mnist-cnn-pytorch/
            train.py
            evaluate.py
            model.py
            utils.py
    models/
             mnist_cnn.pth
    outputs/
            loss_curve.png
            accuracy_curve.png
            confusion_matrix.png
    README.md

### Possible Improvements

* Add Batch Normalization after convolution layers

* Use Dropout in fully-connected layers to reduce overfitting

* Apply learning rate scheduling (StepLR / CosineAnnealingLR)

* Use simple data augmentation (random rotation/shift)

* Increase model capacity (more channels / deeper CNN)
