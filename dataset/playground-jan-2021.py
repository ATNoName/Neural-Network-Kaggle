import sys
sys.path.append('.')
import ml.neuralnetwork as neural
import ml.data as d
import ml.visualization as vis
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

def main():
    # Load the data - 300000 samples
    data_path = 'dataset/tabular-playground-series-jan-2021/train.csv'
    data = d.extract_data(data_path)
    if data is None:
        print('No training data found')
        return
    # Convert into numpy arrays
    x = data.iloc[:, 1:15].values.astype(np.float32)
    y = data['target'].values.astype(np.float32)
    # Visualize some data
    vis.plot_scatter(x, y, 1)
    vis.plot_histo(x, 50)
    ''' 
    Problem: The direct relationship between the features and the target is not clear,
    also some features have different value ranges.
    Interesting fact: There are some distributions patterns in certain features,
    some are seen clearly in the scatter plot, most require histogram
     '''
    
    # preprocess the data
    dataset = d.BasicDataset(x,y)
    train_set, val_set = d.train_val_split(dataset, 0.95)
    train_set = DataLoader(train_set, batch_size=6000, shuffle=True)
    val_set = DataLoader(val_set, batch_size=6000, shuffle=False)
    # Configure the model
    input_size = 14
    output_size = 1
    hidden_size = 50
    hidden_layer = 3
    model = neural.BasicNN(input_size, hidden_size, hidden_layer, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Train the model
    neural.regression_train_model(model, train_set, val_set, criterion, optimizer, 5000)
    # Run the model on the test data
    test_data_path = 'dataset/tabular-playground-series-jan-2021/test.csv'
    test_data = d.extract_data(test_data_path)
    if test_data is None:
        print('No test data found')
        return
    id = d.get_test_ids(test_data, 'id')
    x_test = d.basic_preprocessing(test_data.iloc[:,1:], False)
    x_test = t.tensor(x_test.values, dtype=t.float32)
    y_test = neural.run(model, x_test)
    # Convert model output to dataframe format and then to file
    predictions = pd.DataFrame(y_test, columns=['target'])
    d.write_prediction('dataset/tabular-playground-series-jan-2021/predictions.csv', id, predictions)
    # Visualize the data
    vis.plot_nn_loss(model)

main()