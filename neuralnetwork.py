import torch as t
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import data as d

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        if hidden_layer > 1:
            self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        else:
            hidden_layer = 0
        self.repeat = hidden_layer - 1
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
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
    
def train_model(model, data, target, criterion, optimizer, epochs, threshold=0.01):
    '''Train the model'''
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        if loss.item() < threshold:
            break
        
def main():
    '''Main function'''
    # Load the data
    data_path = 'digit-recognizer/train.csv'
    data = d.extract_data(data_path)
    if data is None:
        return
    # Preprocess the data
    x = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    x = x.astype(np.float32)
    x = x / 255.0
    x = t.tensor(x, dtype=t.float32)
    # convert the labels to one-hot encoding
    y_data = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        y_data[i] = np.array([1 if j == y[i] else 0 for j in range(10)])
    y = t.tensor(y_data, dtype=t.float32)
    # Configure the model
    input_size = 784
    output_size = 10
    hidden_size = 300
    hidden_layer = 1
    model = NeuralNetwork(input_size, hidden_size, hidden_layer, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # Train the model
    train_model(model, x, y, criterion, optimizer, 5000)
    # Save the model
    t.save(model.state_dict(), 'model.pth')
    # Run the model on the test data
    test_data_path = 'digit-recognizer/test.csv'
    test_data = d.extract_data(test_data_path)
    if test_data is None:
        return
    x_test = test_data.values
    x_test = x_test.astype(np.float32)
    x_test = x_test / 255.0
    x_test = t.tensor(x_test, dtype=t.float32)
    y_test = model(x_test)
    y_test = t.argmax(y_test, dim=1)
    # Convert model output to dataframe format and then to file
    y_test = np.array([range(y_test.shape[0]), y_test.numpy()])
    predictions = pd.DataFrame(y_test.T, columns=['ImageId', 'Label'])
    d.write_prediction('predictions.csv', predictions)
    
main()