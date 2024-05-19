import numpy as np
from enum import Enum

class DataType(Enum):
    '''Enum for data types'''
    BINARY = 1
    MULTIVARIATE = 2
    CONTINUOUS_INT = 3
    CONTINUOUS_FLOAT = 4

def data_type(col):
    '''Determine the type of data'''
    if len(set(col)) == 2:
        return DataType.BINARY
    elif len(set(col)) > 2 and all(isinstance(x, int) for x in col):
        return DataType.MULTIVARIATE
    elif all(isinstance(x, int) for x in col):
        return DataType.CONTINUOUS_INT
    else:
        return DataType.CONTINUOUS_FLOAT

class Node:
    '''Node class for Decision Tree'''
    def __init__(self, type):
        self.type = type
        self.children = []
        self.continuous = type == DataType.CONTINUOUS_INT or type == DataType.CONTINUOUS_FLOAT
        if self.continuous:
            self.split = None
        else:
            self.left = []
        
    def add_child(self, node):
        self.children.append(node)
        
    def add_split(self, split):
        if self.continuous:
            self.split = split
        else:
            self.left.append(split)
    
    def next(self, value):
        '''Return the next node based on value'''
        if self.continuous:
            if value <= self.split:
                return self.children[0]
            else:
                return self.children[1]
        else:
            if value in self.left:
                return self.children[0]
            else:
                return self.children[1]

class DecisionTree:
    '''Decision Tree class using ID3 algorithm'''
    def __init__(self):
        self.root = None

def entropy(data):
    '''Calculate the entropy of the data'''
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, labels, type):
    '''Calculate the information gain of the data'''
    if type == DataType.BINARY:
        return entropy(labels) - entropy(labels[data == 0]) - entropy(labels[data == 1])
    elif type == DataType.MULTIVARIATE:
        gain = entropy(labels)
        for value in set(data):
            gain -= len(data[data == value]) / len(data) * entropy(labels[data == value])
        return gain
    elif type == DataType.CONTINUOUS_INT:
        gain = entropy(labels)
        split = np.mean(data)
        for value in [0, 1]:
            gain -= len(data[data <= split]) / len(data) * entropy(labels[data <= split])
        return gain
    else:
        gain = entropy(labels)
        split = np.mean(data)
        for value in [0, 1]:
            gain -= len(data[data <= split]) / len(data) * entropy(labels[data <= split])
        return gain
    
def ID3(node, types, data, labels, max_depth, depth):
    '''ID3 algorithm'''
    pass