from os.path import join
import pandas as pd
import numpy as np

def readData(dataPath, test_trial=4, step=5):
    data = pd.read_csv(dataPath, header=None)
    '''
    data should be in pandas dataframe
    '''
    X = data.iloc[:, :data.shape[1]-1]
    y = data.iloc[:, data.shape[1]-1:]
    
    test_index = list(range(test_trial, X.shape[0], step))
    train_index = list(X.index)
    for i in test_index:
        train_index.remove(i)
    
    test_x = X.iloc[test_index, :]
    test_y = np.ravel(y.iloc[test_index, :])
    train_x = X.iloc[train_index, :]
    train_y = np.ravel(y.iloc[train_index, :])
    
    return train_x, train_y, test_x, test_y

def run():
    dir = "data"
    tasks = ["beach", "finger", "rest"]

    dataDict = {}
    for task in tasks:
        train_x, train_y, test_x, test_y = readData(join(dir, task + "ARP.csv"))
        dataDict[task] = [train_x, train_y, test_x, test_y]

    return dataDict
