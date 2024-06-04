import torch.nn as nn
import numpy as np
import torch as t
from torch.utils.data import DataLoader, Dataset
class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(BasicNN, self).__init__()
        if hidden_layer < 1:
            self.input_to_output = nn.Linear(input_size, output_size)
            self.repeat = -1
        else:
            self.input_to_hidden = nn.Linear(input_size, hidden_size)
            self.activation = nn.LeakyReLU()
            if hidden_layer > 1:
                self.hidden_to_hidden = dict()
                for i in range(hidden_layer - 1):
                    self.hidden_to_hidden[i] = nn.Linear(hidden_size, hidden_size)
            else:
                self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
            self.repeat = hidden_layer - 1
            self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
    def hyperparameters(self):
        '''Return the hyperparameters of the model'''
        if self.repeat < 0:
            return {'hidden_size': self.input_to_output.out_features, 'hidden_layer': 0}
        else:
            return {'hidden_size': self.input_to_hidden.out_features, 'hidden_layer': self.repeat + 1}
        
    def forward(self, x):
        '''Forward pass'''
        if self.repeat < 0:
            return self.input_to_output(x)
        x = self.input_to_hidden(x)
        x = self.activation(x)
        if self.repeat > 0:
            for i in range(self.repeat):
                x = self.hidden_to_hidden[i](x)
                x = self.activation(x)
        x = self.hidden_to_output(x)
        return x

def classify_validate(model, val_set):
    '''Validate the model'''
    accuracy = 0
    for data, target in val_set:
        output = run(model, data)
        output = t.argmax(output, dim=1)
        accuracy += t.sum(output == target).item()
    return accuracy / len(val_set.dataset) * val_set.batch_size

def regression_validate(model, val_set, criterion):
    '''Validate the model'''
    loss = 0
    for data, target in val_set:
        output = run(model, data)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        loss += criterion(output, target)
    return loss / len(val_set.dataset) * val_set.batch_size
    
def classify_train_model(model, train_set, val_set, criterion, optimizer, epochs, threshold=0.99):
    '''Train the model'''
    for epoch in range(1, epochs+1):
        for data, target in train_set:
            optimizer.zero_grad() 
            output = run(model, data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        accuracy = classify_validate(model, val_set)
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Accuracy: {:.4f}".format(accuracy), end = ' ')
        print("Loss: {:.4f}".format(loss.item()))
        if accuracy > threshold:
            break

def regression_train_model(model, train_set, val_set, criterion, optimizer, epochs, threshold=0.01):
    '''Train the model'''
    for epoch in range(1, epochs+1):
        for data, target in train_set:
            optimizer.zero_grad() 
            output = run(model, data)
            if target.dim() == 1:
                target = target.unsqueeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        valid_loss = regression_validate(model, val_set, criterion)
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Last Train Loss: {:.4f}".format(loss.item()), end = ' ')
        print("Validation Loss: {:.4f}".format(valid_loss.item()))
        if loss < threshold:
            break

def run(model, data):
    '''Test the model'''
    return model(data)

