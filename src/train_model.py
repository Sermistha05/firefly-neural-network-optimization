# This script trains a neural network using the Firefly Algorithm to optimize its weights.
# The dataset used is the digits dataset from sklearn, which contains 8x8 pixel images
# of handwritten digits (0-9). The neural network has one hidden layer with 32 neurons and an output layer with 10 neurons (one for each digit class).
# The Firefly Algorithm is used to find the optimal weights for the neural network by minimizing the 
# classification error on the training set. After optimization, we evaluate the model's performance on 
# both the training and testing sets and plot the convergence curve of the optimization process.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.neural_network import NeuralNetwork
from firefly_algorithm import FireflyAlgorithm

# Load and preprocess dataset
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Initialize Neural Network
nn = NeuralNetwork()

# Calculate total weights
# formula for dimension=(input_size * hidden_layer_size + hidden_layer_size) + (hidden_layer_size * output_size + output_size)
dimension = 64*32 + 32 + 32*10 + 10
print(f"Total weights: {dimension}")

# Initialize Firefly Algorithm
fa = FireflyAlgorithm(
    population_size=20,
    dimension=dimension,
    alpha=0.2,
    beta0=1.0,
    gamma=1.0,
    max_iterations=50,
    neural_network=nn,
    X_train=X_train,
    y_train=y_train
)

# Run optimization
print("\nStarting optimization...")
fa.optimize()

# Evaluate accuracy
y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)

train_accuracy = np.mean(y_train_pred == y_train) * 100
test_accuracy = np.mean(y_test_pred == y_test) * 100

print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

# Plot convergence curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(fa.convergence_curve) + 1), fa.convergence_curve, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Firefly Algorithm Convergence Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig('convergence_curve.png')
plt.show()
print("\nConvergence curve saved as 'convergence_curve.png'")
