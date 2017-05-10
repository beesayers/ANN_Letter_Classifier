# Artifical Neural Network - Letter Classifier

#### Libraries
---
- NumPy (from SciPy kit)

#### Objective
---
Train a neural network to correctly classify hand-written letters.

#### Parameters
---
- Input layer of size 17 (1 for bias, 16 for inputs).
    - Required
- Hidden layer of size 70 (1 for bias, 69 for hidden units).
    - Variable
- Output layer of size 26 (26 classifiable letters)
    - Required
- Learning rate = .001
    - Variable between 0 and 1

#### Methods
---
1. Forward Propagation:
The neural network does forward propagation by computing the product of inputs and weights at each layer. The hidden layer uses a sigmoid activation function. The output layer uses a softmax activation function. The unit with the highest magnitude softmax output corresponds to the predicted letter.

2. Backward Propagation:
Back propagation is performed by finding the gradient of the error and adjusting each individual weight accordingly. This was made infinitely more simple using NumPy from the SciPy kit to perform matrix multiplication.

#### Additional Information
1. Input:
The input layer was implemented so input layer unit 1 = feature 1, unit 2 = feature 2, etc.
It takes feature vectors of length 16 re-scaled to 0-1.

2. Output:
The output layer was implemented so output layer unit 1 = A, unit 2 = B, etc.
The predicted output is determined by the unit with the highest magnitude softmax output.
