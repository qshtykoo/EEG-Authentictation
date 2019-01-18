# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:05:21 2019

@author: Ramon Que
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import csv

beach_data = pd.read_csv("beach.csv", header=None)

train_x = beach_data.iloc[:, :402]
train_y = beach_data.iloc[:, 402:]
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(train_x, train_y)
y = ada.predict(train_x)
graph = confusion_matrix(train_y, y)

err = 0
for i in range(train_y.shape[0]):
    if y[i] != train_y.iloc[i,0]:
        err = err + 1
err_rate = err / train_y.shape[0]