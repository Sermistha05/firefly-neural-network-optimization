# Firefly Algorithm for Neural Network Training Optimization on UCI Digits Dataset

## Project Overview

This project implements a neural network trained using the **Firefly Algorithm** for handwritten digit classification. Instead of traditional gradient-based optimization methods like backpropagation, this project uses a nature-inspired metaheuristic algorithm (Firefly Algorithm) to optimize the neural network weights. The system classifies handwritten digits (0-9) from the UCI Digits dataset with a neural network architecture of 64-32-10 neurons.

## Dataset

The project uses the **UCI Digits Dataset** from scikit-learn, which contains:
- 1,797 samples of 8x8 pixel grayscale images of handwritten digits
- 10 classes (digits 0-9)
- 64 features per sample (flattened 8x8 pixel values)

The dataset is split into 80% training and 20% testing sets, with feature normalization applied using StandardScaler.

## Project Structure

```
Firefly_NN_Project/
│
├── neural_network.py          # Neural network class with forward pass, loss computation, and prediction
├── firefly_algorithm.py       # Firefly Algorithm implementation for weight optimization
├── train.py                   # Main training script with full pipeline and visualizations
├── load_digits.py             # Script to load and preprocess the UCI Digits dataset
├── train_model.py             # Alternative training script with basic functionality
├── results.txt                # Saved experiment results (accuracy, parameters)
├── convergence_curve.png      # Visualization of optimization convergence
└── digit_predictions.png      # Sample digit predictions visualization
```

### File Descriptions

- **neural_network.py**: Defines the NeuralNetwork class with ReLU activation for the hidden layer and Softmax for the output layer. Includes forward propagation, cross-entropy loss computation, and prediction methods.

- **firefly_algorithm.py**: Implements the Firefly Algorithm for optimizing neural network weights. Each firefly represents a complete set of network weights, and the algorithm iteratively improves solutions based on brightness (fitness).

- **train.py**: Complete training pipeline that loads data, initializes the neural network, runs Firefly optimization, evaluates performance, and generates visualizations (confusion matrix, convergence curve, sample predictions).

- **load_digits.py**: Standalone script for loading and exploring the UCI Digits dataset with basic statistics.

- **train_model.py**: Simplified training script with core functionality for quick experiments.

- **results.txt**: Stores experiment results including population size, iterations, training accuracy, and testing accuracy for multiple runs.

- **convergence_curve.png**: Graph showing the best fitness value over iterations during optimization.

- **digit_predictions.png**: 2x5 grid displaying sample test images with their predicted labels.

## How the System Works

The workflow follows these steps:

1. **Dataset Loading**: Load UCI Digits dataset and normalize features using StandardScaler
2. **Data Splitting**: Split into 80% training and 20% testing sets
3. **Neural Network Initialization**: Create a 3-layer network (64 → 32 → 10 neurons)
4. **Firefly Optimization**: 
   - Initialize population of fireflies (each representing network weights)
   - Evaluate fitness using cross-entropy loss
   - Move fireflies toward brighter ones (better solutions)
   - Iterate until convergence or max iterations reached
5. **Evaluation**: Calculate training and testing accuracy, generate confusion matrix
6. **Visualization**: Plot convergence curve and sample predictions

### Firefly Algorithm Parameters

- **Population Size**: 40 fireflies
- **Max Iterations**: 120
- **Alpha (α)**: 0.3 (randomization parameter)
- **Beta0 (β₀)**: 1.0 (attractiveness coefficient)
- **Gamma (γ)**: 0.9 (light absorption coefficient)

## How to Run the Project

### Prerequisites

Install required dependencies:
```bash
pip install numpy matplotlib seaborn scikit-learn
```

### Running the Training

Execute the main training script:
```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Train the neural network using Firefly Algorithm
- Display training and testing accuracy
- Generate and save visualizations
- Append results to results.txt

### Alternative Training Script

For a simpler version without all visualizations:
```bash
python train_model.py
```

## Results

The system produces the following outputs:

- **Training Accuracy**: Typically 85-95% depending on optimization convergence
- **Testing Accuracy**: Typically 80-90% on unseen data
- **Convergence Graph**: Shows fitness improvement over 120 iterations
- **Confusion Matrix**: Displays classification performance across all 10 digit classes
- **Sample Predictions**: Visual verification of model predictions on 10 test images

Results are automatically saved to `results.txt` for tracking experiments with different parameters.

## Technologies Used

- **Python 3.x**: Core programming language
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Enhanced visualization for confusion matrix
- **scikit-learn**: Dataset loading, preprocessing, and evaluation metrics

## Key Features

- ✅ Nature-inspired optimization (no backpropagation required)
- ✅ Modular and extensible code structure
- ✅ Comprehensive visualizations (convergence, confusion matrix, predictions)
- ✅ Experiment tracking with results logging
- ✅ Support for both integer and one-hot encoded labels
- ✅ Numerical stability with probability clipping

## Future Improvements

- Implement adaptive parameter tuning for Firefly Algorithm
- Add support for different neural network architectures
- Compare performance with gradient-based methods
- Extend to other datasets (MNIST, Fashion-MNIST)
- Add early stopping based on validation performance

---

**Author**: Machine Learning Enthusiast  
**License**: MIT  
**Year**: 2024
