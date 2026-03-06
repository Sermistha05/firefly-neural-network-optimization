# Firefly Algorithm for Neural Network Training Optimization on UCI Digits Dataset

## Project Overview

This project implements a neural network trained using the **Firefly Algorithm**, a nature-inspired metaheuristic optimization technique, for handwritten digit classification. Unlike traditional gradient-based methods such as backpropagation, this approach leverages swarm intelligence to optimize neural network weights. The model is trained and evaluated on the **UCI Digits Dataset** from scikit-learn, achieving competitive accuracy through bio-inspired optimization.

## Algorithm Explanation

The **Firefly Algorithm** is a nature-inspired optimization algorithm based on the flashing behavior of fireflies. In this implementation:

- Each **firefly** represents a complete set of neural network weights (a candidate solution)
- **Brightness** corresponds to the fitness of the solution (inverse of classification loss)
- Fireflies are attracted to brighter fireflies (better solutions) and move toward them
- The movement is governed by the equation: `xi = xi + β * exp(-γ * r²) * (xj - xi) + α * ε`
  - `β`: attractiveness coefficient
  - `γ`: light absorption coefficient
  - `α`: randomization parameter
  - `r`: distance between fireflies
  - `ε`: random noise

Through iterative movement and evaluation, the algorithm converges to optimal or near-optimal neural network weights without requiring gradient computation.

## Dataset

The **UCI Digits Dataset** from scikit-learn is used for training and evaluation:

- **Samples**: 1,797 grayscale images of handwritten digits (0-9)
- **Image Size**: 8×8 pixels (64 features per sample)
- **Classes**: 10 (digits 0 through 9)
- **Split**: 80% training, 20% testing
- **Preprocessing**: Feature normalization using StandardScaler

## Project Structure

```
Firefly_NN_Project/
│
├── neural_network.py          # Neural network implementation (64-32-10 architecture)
├── firefly_algorithm.py       # Firefly Algorithm optimization engine
├── train.py                   # Main training pipeline with full visualizations
├── load_digits.py             # Dataset loading and preprocessing utilities
├── train_model.py             # Simplified training script for quick experiments
├── requirements.txt           # Python dependencies
├── results.txt                # Experiment results log (auto-generated)
├── convergence_curve.png      # Optimization convergence plot (auto-generated)
└── digit_predictions.png      # Sample predictions visualization (auto-generated)
```

### File Descriptions

- **neural_network.py**: Implements a 3-layer feedforward neural network with ReLU activation (hidden layer) and Softmax activation (output layer). Includes forward propagation, cross-entropy loss computation, and prediction methods.

- **firefly_algorithm.py**: Core implementation of the Firefly Algorithm. Manages population initialization, fitness evaluation, firefly movement dynamics, and convergence tracking.

- **train.py**: Complete training pipeline that orchestrates data loading, neural network initialization, Firefly optimization, performance evaluation, and result visualization (confusion matrix, convergence curve, sample predictions).

- **load_digits.py**: Utility script for loading the UCI Digits dataset, applying normalization, and splitting into training/testing sets.

- **train_model.py**: Lightweight training script with essential functionality for rapid prototyping and experimentation.

- **requirements.txt**: Lists all Python package dependencies with version constraints for reproducible environment setup.

## Workflow / System Architecture

The system follows a structured pipeline:

```
┌─────────────────┐
│  UCI Digits     │
│  Dataset        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data            │
│ Preprocessing   │  (Normalization, Train-Test Split)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neural Network  │
│ Initialization  │  (64 → 32 → 10 architecture)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Firefly         │
│ Optimization    │  (Weight optimization via swarm intelligence)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model           │
│ Evaluation      │  (Accuracy, Confusion Matrix)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │
│ & Results       │  (Convergence, Predictions, Metrics)
└─────────────────┘
```

**Pipeline Steps:**

1. **Dataset Loading**: Load UCI Digits dataset from scikit-learn
2. **Data Preprocessing**: Normalize features using StandardScaler and split into 80-20 train-test sets
3. **Neural Network Initialization**: Create a 3-layer network with 64 input neurons, 32 hidden neurons, and 10 output neurons
4. **Firefly Optimization**: Initialize population of 40 fireflies, each representing a weight configuration. Iterate for 120 generations, evaluating fitness and moving fireflies toward better solutions
5. **Evaluation**: Compute training and testing accuracy, generate confusion matrix
6. **Visualization**: Plot convergence curve, display sample predictions, save results to file

## Results

The trained model produces the following outputs:

### Performance Metrics
- **Training Accuracy**: 85-95% (varies with optimization convergence)
- **Testing Accuracy**: 80-90% on unseen data

### Visualizations
- **Convergence Curve** (`convergence_curve.png`): Tracks best fitness value across 120 iterations, demonstrating optimization progress
- **Confusion Matrix**: Heatmap showing classification performance across all 10 digit classes
- **Sample Predictions** (`digit_predictions.png`): 2×5 grid displaying 10 test images with predicted labels

### Experiment Logs
- **results.txt**: Automatically appends experiment results including population size, iterations, training accuracy, and testing accuracy for reproducibility

## How to Run the Project

### Prerequisites

Ensure you have Python 3.7+ installed.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Firefly_NN_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Training

Execute the main training script:
```bash
python train.py
```

This will:
- Load and preprocess the UCI Digits dataset
- Initialize the neural network
- Run Firefly Algorithm optimization (40 fireflies, 120 iterations)
- Display training and testing accuracy
- Generate and save visualizations
- Append results to `results.txt`

### Alternative: Quick Training

For a simplified version without extensive visualizations:
```bash
python train_model.py
```

## Technologies Used

- **Python 3.x**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization (confusion matrix heatmaps)
- **scikit-learn**: Dataset loading, preprocessing, and evaluation metrics

## Key Features

✅ **Gradient-Free Optimization**: No backpropagation required  
✅ **Nature-Inspired Algorithm**: Leverages swarm intelligence  
✅ **Modular Architecture**: Clean, extensible codebase  
✅ **Comprehensive Visualizations**: Convergence tracking, confusion matrix, sample predictions  
✅ **Experiment Tracking**: Automatic logging of results  
✅ **Flexible Label Encoding**: Supports both integer and one-hot encoded labels  
✅ **Numerical Stability**: Probability clipping to prevent log(0) errors  

## Future Enhancements

- Implement adaptive parameter tuning for Firefly Algorithm
- Add support for deeper neural network architectures
- Benchmark against gradient-based optimization methods (SGD, Adam)
- Extend to larger datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Implement early stopping based on validation performance
- Add hyperparameter grid search functionality

---

**License**: MIT  
**Year**: 2024
