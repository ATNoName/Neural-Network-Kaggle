class Hyperparameter:
    def __init__(self, model, learning_rate, batch_size, epochs, criterion, optimizer, threshold):
        self.model_parameter = model.hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion_list = criterion
        self.optimizer_list = optimizer
        self.threshold = threshold
        
    def tune():
        '''Perform hyperparameter tuning using Bayes Optimization'''
        
        pass
    
    def acquisition_function():
        '''Compute the acquisition function'''
        
        pass
    
    def objective_function():
        '''Compute the objective function'''
        
        pass
    
    def surrogate_function():
        '''Compute the surrogate function'''
        
        pass
        
    def __str__(self):
        return 'Model: {}\nLearning Rate: {}\nBatch Size: {}\nEpochs: {}\nCriterion: {}\nOptimizer: {}'.format(
            self.model, self.learning_rate, self.batch_size, self.epochs, self.criterion, self.optimizer)