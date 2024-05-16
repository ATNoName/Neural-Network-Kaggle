import pandas as pd
from torch.utils.data import Dataset 

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

def write_prediction(file_path, predictions):
    '''Writes the predictions to a file'''
    try:
        predictions.to_csv(file_path, index=False)
    except:
        print('Error writing to file')