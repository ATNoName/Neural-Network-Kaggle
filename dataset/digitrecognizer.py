import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import sys
sys.path.append('.')
import ml.neuralnetwork as neural
import ml.data as d

def main():
    '''Main function'''
    # Load the data
    data_path = 'dataset/digit-recognizer/train.csv'
    data = d.extract_data(data_path)
    if data is None:
        print('No training data found')
        return
    # Preprocess the data
    x = data.loc[:, data.columns != 'label'].values
    x = x.astype(np.float32)
    x = x / 255.0
    y = data['label'].values
    # Create the dataset
    data = d.BasicDataset(x, y)
    split = int(0.9*len(data))
    train_set, val_set = random_split(data, [split, len(data) - split])
    train_set = DataLoader(train_set, batch_size=1000, shuffle=True)
    val_set = DataLoader(val_set, batch_size=1000, shuffle=False)
    # Configure the model
    input_size = 784
    output_size = 10
    hidden_size = 300
    hidden_layer = 1
    model = neural.BasicNN(input_size, hidden_size, hidden_layer, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0075)
    # Train the model
    neural.train_model(model, train_set, val_set, criterion, optimizer, 5000, 0.975)
    # Run the model on the test data
    test_data_path = 'dataset/digit-recognizer/test.csv'
    test_data = d.extract_data(test_data_path)
    if test_data is None:
        print('No test data found')
        return
    x_test = test_data.values
    x_test = x_test.astype(np.float32)
    x_test = x_test / 255.0
    x_test = t.tensor(x_test, dtype=t.float32)
    y_test = neural.run(model, x_test)
    y_test = t.argmax(y_test, dim=1)
    # Convert model output to dataframe format and then to file
    y_test = np.array([range(1, y_test.shape[0]+1), y_test.numpy()])
    predictions = pd.DataFrame(y_test.T, columns=['ImageId', 'Label'])
    predictions.to_csv('dataset/digit-recognizer/predictions.csv', index=False)
    
main()