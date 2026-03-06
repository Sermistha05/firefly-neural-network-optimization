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
        self.fireflies = np.random.uniform(-1, 1, (self.population_size, self.dimension))
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
        
        # Forward pass and compute loss
        y_pred = self.neural_network.forward(self.X_train)
        loss = self.neural_network.compute_loss(self.y_train, y_pred)
        
        # Fitness = 1 / (loss + epsilon)
        return 1.0 / (loss + 1e-6)
    
    def move_fireflies(self):
        # Move fireflies based on brightness (fitness)
        for i in range(self.population_size):
            for j in range(self.population_size):
                if self.fitness_values[j] > self.fitness_values[i]:
                    # Calculate distance
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    
                    # Calculate attractiveness: beta = beta0 * exp(-gamma * r^2)
                    beta = self.beta0 * np.exp(-self.gamma * r**2)
                    
                    # Move firefly i towards j with random noise
                    random_noise = np.random.randn(self.dimension)
                    self.fireflies[i] = self.fireflies[i] + \
                                        beta * (self.fireflies[j] - self.fireflies[i]) + \
                                        self.alpha * random_noise
    
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
