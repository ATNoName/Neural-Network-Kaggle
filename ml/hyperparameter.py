import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import neuralnetwork as neural
class Hyperparameters:
    '''Class to perform hyperparameter tuning using Bayesian Optimization. 
    Uses Sequential Model-Based Optimization (SMBO) to optimize the hyperparameters of a given model.'''
    def __init__(self, model, **hyperparameters):
        '''Initialize the hyperparameters by declaring the variable and it's domain'''
        self.model = model
        self.hyperparameters = hyperparameters
        self.kernel = RBF(length_scale=1)
        self.surrogate_model = GaussianProcessRegressor(kernel=self.kernel)
    
    def optimize(self, samplex, sampley, val_data, train_function, loss_function, beta=2, iterations=10):
        '''Optimize the hyperparameters using Bayesian Optimization'''
        self.fit(samplex, sampley)
        best = None
        for _ in range(iterations):
            x = self.sample()
            y = self.acquisition_function(x, beta)
            if best is None or y > best:
                best = y
                best_x = x
        return best_x
    
    def sample(self):
        '''Sample the hyperparameters'''
        x = []
        for key, value in self.hyperparameters.items():
            x.append(np.random.uniform(value[0], value[1]))
        return x
    
    def fit(self, samplex, sampley):
        '''Fit the surrogate model'''
        return self.surrogate_model.fit(samplex, sampley)
    
    def acquisition_function(self, x, beta):
        '''Compute the acquisition function using Upper Confidence Bound (UCB)'''
        mu, sigma = self.surrogate_function(x)
        return mu + beta * sigma
    
    def objective_function(self, train_data, val_data, train_function, loss_function):
        '''Compute the objective function'''
        x = self.sample()
        y = train_function(self.model, train_data, val_data, loss_function, **dict(zip(self.hyperparameters.keys(), x)))
        return y
    
    def surrogate_function(self, x):
        '''Fit the surrogate model'''
        return self.surrogate_model.predict(x, return_std=True)
    
    def gaussian_distribution(x, mu, sigma):
        '''Compute the Gaussian distribution'''
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * math.pow((x - mu) / sigma, 2))
        
    def __str__(self):
        return 'Model: {}\nLearning Rate: {}\nBatch Size: {}\nEpochs: {}\nCriterion: {}\nOptimizer: {}'.format(
            type(self.model), self.learning_rate, self.batch_size, self.epochs, self.criterion, self.optimizer)