import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import sys
sys.path.append('.')
import ml.decisiontree as dt
import ml.data as d

def main():
    '''Main function'''
    # Load the data
    data_path = 'dataset/titanic/train.csv'
    data = d.extract_data(data_path)
    if data is None:
        print('No training data found')
        return
    # Edit the data :TODO
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    # Create datasets
    train_set, val_set = d.train_val_split(data, 0.9)
    train_data, train_labels = d.label_split(train_set, 'Survived')
    # Figure out the types of the data
    types = []
    for column in train_data.columns:
        types.append(dt.data_type(train_data[column]))
    # Configure the model
    model = dt.DecisionTree()
    max_depth = 5
    # Train the model
    dt.ID3(model.root, types, train_data, train_labels, max_depth, 0)
    # Validate the model
    val_data, val_labels = d.label_split(val_set, 1)
    accuracy = dt.validate(model, val_data, val_labels)
    print('Validation accuracy: {:.4f}'.format(accuracy))
    # Run the model on the test data
    test_data_path = 'dataset/titanic/test.csv'
    test_data = d.extract_data(test_data_path)
    if test_data is None:
        print('No test data found')
        return
    predictions = []
    for index, row in test_data.iterrows():
        predictions.append(dt.run(model, row))
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    d.write_prediction('dataset/titanic/predictions.csv', predictions)
    
main()