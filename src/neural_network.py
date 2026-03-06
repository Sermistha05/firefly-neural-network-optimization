import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights with small random values
        self.W1 = np.random.randn(64, 32) * 0.01 # 64 input features, 32 hidden neurons
        self.b1 = np.zeros((1, 32)) # 1, 32 are the dimensions of the bias for the hidden layer
        self.W2 = np.random.randn(32, 10) * 0.01 # 32 hidden neurons, 10 output classes 
        self.b2 = np.zeros((1, 10)) # 1, 10 are the dimensions of the bias for the output layer
    
    def forward(self, X):
        # Hidden layer with ReLU activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        
        # Output layer with Softmax activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss with clipping to avoid log(0)
        epsilon = 1e-10
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Handle both integer labels and one-hot encoded labels
        if y_true.ndim == 1:  # Integer labels
            m = y_true.shape[0]
            log_likelihood = -np.log(y_pred_clipped[range(m), y_true])
        else:  # One-hot encoded labels
            log_likelihood = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        loss = np.mean(log_likelihood)
        return loss
    
    def predict(self, X):
        # Return predicted class labels
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
