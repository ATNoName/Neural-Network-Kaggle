import pandas as pd

'''This file does any data handling'''

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