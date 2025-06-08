# Digit Recognizer - Neural Network from Scratch

A neural network implementation from scratch using NumPy for the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer/overview).

## Overview

This project implements a 2-layer neural network to classify handwritten digits (0-9) from the MNIST dataset without using any deep learning frameworks like TensorFlow or PyTorch.

## Model Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit class)

## Key Features

- **Pure NumPy Implementation**: Built from scratch using only NumPy for matrix operations
- **Forward Propagation**: ReLU activation for hidden layer, Softmax for output
- **Backpropagation**: Custom gradient computation and parameter updates
- **Data Preprocessing**: Pixel normalization (0-255 → 0-1) for faster convergence

## Results

- **Training Accuracy**: ~84.7% after 500 iterations
- **Learning Rate**: 0.10
- **Training Time**: Fast convergence with clear accuracy improvement over iterations

## Code Structure

```python
# Core Functions
- init_params()      # Initialize weights and biases
- forward_prop()     # Forward pass through network
- backward_prop()    # Compute gradients via backpropagation
- update_params()    # Update weights using gradient descent
- gradient_descent() # Main training loop
```

## Usage

```python
# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# Make predictions
predictions = make_predictions(X_test, W1, b1, W2, b2)

# Test individual samples
test_prediction(index, W1, b1, W2, b2)
```

## Dataset

- **Training Set**: 42,000 labeled images
- **Test Set**: 28,000 unlabeled images for submission
- **Validation Set**: 1,000 samples held out from training data

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Implementation Highlights

- **One-Hot Encoding**: Converts labels to categorical format for multi-class classification
- **Vectorized Operations**: Efficient matrix computations for batch processing
- **Learning Visualization**: Real-time accuracy tracking during training
- **Prediction Testing**: Visual verification of model predictions with matplotlib

This implementation demonstrates fundamental deep learning concepts including gradient descent, backpropagation, and neural network architecture design using only basic mathematical operations.
