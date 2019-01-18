# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:05:21 2019

@author: Ramon Que
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def error_rate(y_true, y):
    err = 0
    for i in range(y_true.shape[0]):
        if y[i] != y_true.iloc[i,0]:
            err = err + 1
    return err / y.shape[0]

beach_data = pd.read_csv("finger.csv", header=None)

X = beach_data.iloc[:, :402]
y = beach_data.iloc[:, 402:]

test_index = list(range(4,135,5))
train_index = list(X.index)
for i in test_index:
    train_index.remove(i)

test_x = X.iloc[test_index, :]
test_y = y.iloc[test_index, :]
train_x = X.iloc[train_index, :]
train_y = y.iloc[train_index, :]
    
gnb = GaussianNB()
y_gnb = gnb.fit(train_x, train_y).predict(test_x)
#graph = confusion_matrix(test_y, y_gnb)

ada = AdaBoostClassifier(n_estimators=7)
ada.fit(train_x, train_y)
y_ada = ada.predict(train_x)

svm_clf = SVC(gamma='auto')
y_svm = svm_clf.fit(train_x,train_y).predict(train_x)

err_rate = error_rate(train_y, y_svm)

graph = confusion_matrix(train_y, y_svm)