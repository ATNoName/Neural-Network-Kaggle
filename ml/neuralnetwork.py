import torch.nn as nn
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Dataset
class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(BasicNN, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.LeakyReLU()
        if hidden_layer > 1:
            self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        else:
            hidden_layer = 0
        self.repeat = hidden_layer - 1
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
    def hyperparameters(self):
        '''Return the hyperparameters of the model'''
        return {'hidden_size': self.input_to_hidden.out_features, 'hidden_layer': self.repeat + 1}
        
    def forward(self, x):
        '''Forward pass'''
        x = self.input_to_hidden(x)
        x = self.activation(x)
        if self.repeat >= 0:
            for _ in range(self.repeat):
                x = self.hidden_to_hidden(x)
                x = self.activation(x)
        x = self.hidden_to_output(x)
        return x

def validate(model, val_set):
    '''Validate the model'''
    accuracy = 0
    for data, target in val_set:
        output = run(model, data)
        output = t.argmax(output, dim=1)
        accuracy += t.sum(output == target).item()
    return accuracy / len(val_set.dataset)
    
def train_model(model, train_set, val_set, criterion, optimizer, epochs, threshold=0.99):
    '''Train the model'''
    for epoch in range(1, epochs+1):
        for data, target in train_set:
            optimizer.zero_grad() 
            output = run(model, data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        accuracy = validate(model, val_set)
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Accuracy: {:.4f}".format(accuracy), end = ' ')
        print("Loss: {:.4f}".format(loss.item()))
        if accuracy > threshold:
            break
        
def run(model, data):
    '''Test the model'''
    return model(data)

