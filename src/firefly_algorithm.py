import numpy as np

class FireflyAlgorithm:
    def __init__(self, population_size, dimension, alpha, beta0, gamma, max_iterations,
                 neural_network, X_train, y_train):
        self.population_size = population_size
        self.dimension = dimension
        self.convergence_curve = []
        
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.neural_network = neural_network
        self.X_train = X_train
        self.y_train = y_train
        
        self.fireflies = None
        self.fitness_values = None
        self.best_firefly = None
        self.best_fitness = -np.inf
    
    def initialize_population(self):
        # Initialize fireflies with random weight values
        self.fireflies = np.random.uniform(-2, 2, (self.population_size, self.dimension))
        self.fitness_values = np.zeros(self.population_size)
    
    def set_weights_to_network(self, weight_vector):
        # Unflatten and set weights to neural network
        idx = 0
        self.neural_network.W1 = weight_vector[idx:idx + 64*32].reshape(64, 32)
        idx += 64 * 32
        self.neural_network.b1 = weight_vector[idx:idx + 32].reshape(1, 32)
        idx += 32
        self.neural_network.W2 = weight_vector[idx:idx + 32*10].reshape(32, 10)
        idx += 32 * 10
        self.neural_network.b2 = weight_vector[idx:idx + 10].reshape(1, 10)
    
    def fitness(self, weight_vector):
        # Set weights and compute fitness
        self.set_weights_to_network(weight_vector)
        
        # Forward pass
        y_pred = self.neural_network.forward(self.X_train)
        predictions = np.argmax(y_pred, axis=1)
        
        # Handle both integer labels and one-hot labels
        if self.y_train.ndim > 1:
            y_true = np.argmax(self.y_train, axis=1)
        else:
            y_true = self.y_train
        
        accuracy = np.mean(predictions == y_true)
        return accuracy
    
    def move_fireflies(self):
        # Move fireflies based on brightness (fitness)
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.fitness_values[j] > self.fitness_values[i]:
                    # Calculate distance
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])

                    # Calculate attractiveness: beta = beta0 * exp(-gamma * r^2)
                    beta = self.beta0 * np.exp(-self.gamma * r * r)

                    # Centered random step for balanced exploration
                    random_step = self.alpha * (np.random.rand(self.dimension) - 0.5)

                    # Move firefly i towards j with random step
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + random_step

                    # Slight Gaussian noise boost to prevent premature convergence
                    self.fireflies[i] += 0.01 * np.random.randn(self.dimension)

                    # Clip weights to tighter range to maintain search stability
                    self.fireflies[i] = np.clip(self.fireflies[i], -1, 1)
    
    def optimize(self):
        # Initialize population
        self.initialize_population()
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all fireflies
            for i in range(self.population_size):
                self.fitness_values[i] = self.fitness(self.fireflies[i])
            
            # Track best firefly
            best_idx = np.argmax(self.fitness_values)
            if self.fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_firefly = self.fireflies[best_idx].copy()
            self.convergence_curve.append(self.best_fitness)

            # Move fireflies
            self.move_fireflies()
            self.alpha *= 0.98  # Optional: decay alpha over iterations
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {self.best_fitness:.6f}")
        
        # Set best weights to neural network
        self.set_weights_to_network(self.best_firefly)
        return self.best_firefly
