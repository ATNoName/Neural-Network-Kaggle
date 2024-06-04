import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

'''This file does any data handling'''

class BasicDataset(Dataset):
    '''Dataset class'''
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def extract_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print('File not found')
        return None
    return data

def basic_preprocessing(data, norm = True):
    '''Preprocess the data by either normalizing or standardizing it. Assumes the data is already structured'''
    if not norm:
        data = (data - data.mean()) / data.std()
    else:
        data = (data - data.min()) / (data.max() - data.min())
    return data.astype(np.float32)

def train_val_split(data = Dataset, val_percent = float()):
    '''Splits the data into training and validation sets'''
    n = int(len(data) * val_percent)
    train_set, val_set = random_split(data, [n, len(data) - n])
    return train_set, val_set

def label_split(data = Dataset, label = str()):
    '''Split the data into features and labels'''
    labels = data.dataset.loc[:, label]
    data = data.dataset.drop(label, axis=1)
    return data, labels

def get_test_ids(data = pd.DataFrame, id = 'Id'):
    '''Get the ids of the test data'''
    return data.loc[:, id]

def write_prediction(file_path, id, predictions):
    '''Writes the predictions to a file'''
    try:
        dt = pd.concat([id, predictions], axis=1)
        dt.to_csv(file_path, index=False)
    except:
        print('Error writing to file')