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

def preprocess(data):
    '''Preprocess the data by converting it to a usable format'''
    # Copy any columns that won't be modified
    new_data = data[['Pclass']]
    # Convert any useful columns into numerical data
    data['Embarked'] = data['Embarked'].fillna('S') # Fill in missing values
    new_data = pd.concat([new_data, data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})], axis=1)
    data['Age'] = data['Age'].fillna(data['Age'].mean()) # Fill in missing values
    new_data = pd.concat([new_data, data['Age']], axis=1)
    new_data = pd.concat([new_data, data['Sex'].map({'male':0,'female':1})], axis=1)
    ticket_num = data['Ticket'].apply(lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0)
    new_data = pd.concat([new_data, ticket_num], axis=1)
    # Use the rest of the data to create new columns
    family_size = pd.DataFrame(data['SibSp'] + data['Parch'], columns=['FamilySize'])
    new_data = pd.concat([new_data, family_size], axis=1)
    num_cabin = pd.DataFrame(columns=['NumCabin'])
    num_cabin['NumCabin'] = data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    new_data = pd.concat([new_data, num_cabin], axis=1)
    fare_per_person = pd.DataFrame(data['Fare'] / (family_size['FamilySize'] + 1), columns=['FarePerPerson'])
    new_data = pd.concat([new_data, fare_per_person], axis=1)
    friends = pd.DataFrame(columns=['Friends'])
    friends['Friends'] = ticket_num.apply(lambda x: len(data[data['Ticket'].apply(lambda y: int(y.split()[-1]) if y.split()[-1].isdigit() else 0) == x]))
    new_data = pd.concat([new_data, friends], axis=1)
    # normalize the data
    new_data = d.basic_preprocessing(new_data, False)
    return new_data

def main():
    '''Main function'''
    # Load the data
    data_path = 'dataset/titanic/train.csv'
    data = d.extract_data(data_path)
    if data is None:
        print('No training data found')
        return
    # Edit the data
    labels = data['Survived'].values
    data = preprocess(data).values
    data = data.astype(np.float32)
    # Create the dataset
    data = d.BasicDataset(data, labels)
    split = int(0.9*len(data))
    train_set, val_set = random_split(data, [split, len(data) - split])
    train_set = DataLoader(data, batch_size=81, shuffle=True)
    val_set = DataLoader(val_set, batch_size=81, shuffle=False)
    # Configure the model
    input_size = 9
    output_size = 2
    hidden_size = 10
    hidden_layer = 1
    model = neural.BasicNN(input_size, hidden_size, hidden_layer, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Train the model
    neural.train_model(model, train_set, val_set, criterion, optimizer, 1000, 1)
    # Run the model on the test data
    test_data_path = 'dataset/titanic/test.csv'
    test_data = d.extract_data(test_data_path)
    if test_data is None:
        print('No test data found')
        return
    id = d.get_test_ids(test_data, 'PassengerId')
    test_data = preprocess(test_data)
    test_data = t.tensor(test_data.values, dtype=t.float32)
    predictions = neural.run(model, test_data)
    predictions = t.argmax(predictions, dim=1)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    d.write_prediction('dataset/titanic/predictions.csv', id, predictions)
    
main()