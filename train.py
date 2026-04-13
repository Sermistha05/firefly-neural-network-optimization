# we are training a neural network using the firefly algorithm to optimize the weights of the network. 
# The dataset used is the digits dataset from sklearn, which contains 8x8 pixel images of handwritten digits(0-9).
#  The neural network has one hidden layer with 32 neurons and an output layer with 10 neurons (one for each digit
#  class). The firefly algorithm is used to find the optimal weights for the neural network by minimizing the 
# classification error on the training set. After optimization, we evaluate the model's performance on both the
#  training and testing sets and plot the convergence curve of the optimization process.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # seaborn is used for better visualization of the confusion matrix
from sklearn.datasets import load_digits # sklearn is used to load the digits dataset
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# mainly sklearn is used for data preprocessing and evaluation metrics,
# while matplotlib and seaborn are used for visualization.
from src.neural_network import NeuralNetwork
from src.firefly_algorithm import FireflyAlgorithm
#src normally is used to import custom modules


def one_hot_encode(y, num_classes):
    # Convert labels to one-hot encoded format
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


# Load and preprocess dataset
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
y_train_onehot = one_hot_encode(y_train, 10)
y_test_onehot = one_hot_encode(y_test, 10)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# Initialize Neural Network
nn = NeuralNetwork()


# Calculate total weights
n_weights = (64 * 32 + 32) + (32 * 10 + 10)
print(f"Total weights to optimize: {n_weights}")


# Initialize Firefly Algorithm
fa = FireflyAlgorithm(
    population_size=80,
    dimension=n_weights,
    alpha=0.8,
    beta0=1,
    gamma=0.3,
    max_iterations=120,
    neural_network=nn,
    X_train=X_train,
    y_train=y_train_onehot
)


print("\nStarting Firefly Optimization...\n")

# Run optimization
best_weights = fa.optimize()

print("\nOptimization Finished!")


# Evaluate model
y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)

train_accuracy = np.mean(y_train_pred == y_train) * 100
test_accuracy = np.mean(y_test_pred == y_test) * 100

# training and testing accuracy
print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")

# Save results to file
with open('results.txt', 'a') as f:
    f.write(f"Population Size: {fa.population_size}, Iterations: {fa.max_iterations}\n")
    f.write(f"Training Accuracy: {train_accuracy:.2f}%\n")
    f.write(f"Testing Accuracy: {test_accuracy:.2f}%\n")
    f.write("-" * 50 + "\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Plot convergence curve
plt.figure(figsize=(10,6))
plt.plot(fa.convergence_curve, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("Firefly Algorithm Convergence")
plt.grid(True)

plt.savefig("convergence_curve.png")
plt.show()

print("\nConvergence curve saved as 'convergence_curve.png'")

# Visualize sample predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Sample Digit Predictions", fontsize=16)

for i in range(10):
    row = i // 5
    col = i % 5
    
    image = X_test[i].reshape(8, 8)
    axes[row, col].imshow(image, cmap='gray')
    
    prediction = nn.predict(X_test[i:i+1])[0]
    axes[row, col].set_title(f'Predicted: {prediction}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('digit_predictions.png')
plt.show()
