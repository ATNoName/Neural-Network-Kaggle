
class Hyperparameters:
    '''Class to perform hyperparameter tuning using Bayesian Optimization. 
    Uses Sequential Model-Based Optimization (SMBO) to optimize the hyperparameters of a given model.'''
    def __init__(self, model, **hyperparameters):
        '''Initialize the hyperparameters by declaring the variable and it's domain'''
        self.model = model
        self.hyperparameters = hyperparameters
        self.surrogate = None # Surrogate model uses Gaussian Process
        
    def tune():
        '''Main process of tuning the hyperparameters'''
        
        pass
    
    def acquisition_function():
        '''Compute the acquisition function using Upper Confidence Bound (UCB)'''
        
        pass
    
    def objective_function(self, train_data, val_data, train_function, loss_function):
        '''Compute the objective function'''
        pass
        
    def __str__(self):
        return 'Model: {}\nLearning Rate: {}\nBatch Size: {}\nEpochs: {}\nCriterion: {}\nOptimizer: {}'.format(
            type(self.model), self.learning_rate, self.batch_size, self.epochs, self.criterion, self.optimizer)