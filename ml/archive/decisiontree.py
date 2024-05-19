import numpy as np
import pandas as pd
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
    elif col.dtype == np.int64 and set(col) == range(min(col), max(col)+1):
        return DataType.MULTIVARIATE
    elif col.dtype == np.int64:
        return DataType.CONTINUOUS_INT
    else:
        return DataType.CONTINUOUS_FLOAT

class Node:
    '''Node class for Decision Tree'''
    def __init__(self, arg1=None, arg2=None):
        # Leaf node only
        if arg2 == None:
            self.label = arg1
            self.leaf = True
        else:
            # Internal node only
            self.attribute = arg1
            self.split = arg2
            self.left = None
            self.right = None
            self.leaf = False
    
    def traverse(self, row):
        '''Traverse the tree to get the label'''
        if self.leaf:
            return self.label
        if type(self.split) == int or type(self.split) == float:
            if row[self.attribute] <= self.split:
                return self.left.traverse(row)
            else:
                return self.right.traverse(row)
        else:
            if row[self.attribute] in self.split:
                return self.left.traverse(row)
            else:
                return self.right.traverse(row)

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
    
def ID3(types, attributes, data, labels, max_depth, depth, default):
    '''ID3 algorithm'''
    if depth == max_depth or len(attributes) == 0:
        return Node(labels.mode())
    if len(set(labels)) == 1:
        return Node(labels[0])
    if len(data) == 0:
        return Node(default)
    # Find the best attribute to split on
    gains = []
    for i in range(len(attributes)):
        gains.append(information_gain(data.iloc[:,i], labels, types[i]))
    best_attribute = attributes[np.argmax(gains)]
    best_index = attributes.get_loc(best_attribute)
    attributes.delete(best_index)
    # Find the best split for the attribute
    if types[best_index] == DataType.BINARY:
        split = 0
    elif types[best_index] == DataType.MULTIVARIATE:
        split = set(data[best_attribute])
    else:
        split = np.mean(data.iloc[:,best_index])
    # Create the node
    node = Node(best_attribute, split)
    # Recurse
    if types[best_index] == DataType.BINARY:
        node.left = ID3(types, attributes, data[data.iloc[:, best_index] == 0], labels[data.iloc[:, best_index] == 0], max_depth, depth+1, default)
        node.right = ID3(types, attributes, data[data.iloc[:, best_index] == 1], labels[data.iloc[:, best_index] == 1], max_depth, depth+1, default)
    elif types[best_index] == DataType.MULTIVARIATE:
        for value in split:
            node.left = ID3(types, attributes, data[data.iloc[:, best_index] == value], labels[data.iloc[:, best_index] == value], max_depth, depth+1, default)
            node.right = ID3(types, attributes, data[data.iloc[:, best_index] != value], labels[data.iloc[:, best_index] != value], max_depth, depth+1, default)
    else:
        node.left = ID3(types, attributes, data[data.iloc[:, best_index] <= split], labels[data.iloc[:, best_index] <= split], max_depth, depth+1, default)
        node.right = ID3(types, attributes, data[data.iloc[:, best_index] > split], labels[data.iloc[:, best_index] > split], max_depth, depth+1, default)
    return node

def run(model = DecisionTree, data = pd.DataFrame):
    '''Run the model on the data'''
    model.root.traverse(data)
    pass