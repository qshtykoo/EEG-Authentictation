# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:05:21 2019

@author: Ramon Que
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def error_rate(y_true, y):
    if isinstance(y_true, pd.core.frame.DataFrame):
        y_true = np.ravel(y_true.values) #return a contiguous flattened array
    err = 0
    for i in range(len(y_true)):
        if y[i] != y_true[i]:
            err = err + 1
    return err / len(y_true)

def confusion_matrix(y_true, y_pred):
    if isinstance(y_true, pd.core.frame.DataFrame):
        y_true = np.ravel(y_true.values) #return a contiguous flattened array
    num_class = len(np.unique(y_true))
    c_matrix = np.zeros((num_class, num_class))
    for i in range(len(y_true)):
        x = int(y_true[i])
        y = int(y_pred[i])
        c_matrix[x-1, y-1] += 1
    return c_matrix

def error_list_ada(iteration_num, weak_learner=None):
    test_err_list = []
    train_err_list = []
    for T in iteration_num:
        ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=T)
        y_ada_test = ada.fit(train_x, train_y).predict(test_x)
        y_ada_train = ada.fit(train_x, train_y).predict(train_x)
        test_err_list.append(error_rate(test_y, y_ada_test))
        train_err_list.append(error_rate(train_y, y_ada_train))
    
    return train_err_list, test_err_list

beach_data = pd.read_csv("beach.csv", header=None)
finger_data = pd.read_csv("finger.csv", header=None)
rest_data = pd.read_csv("rest.csv", header=None)

X_total = np.concatenate((beach_data.iloc[:, :402], finger_data.iloc[:, :402], rest_data.iloc[:, :402]))
y_total = np.concatenate((beach_data.iloc[:, 402:], finger_data.iloc[:, 402:], rest_data.iloc[:, 402:]))

X = beach_data.iloc[:, :402]
y = beach_data.iloc[:, 402:]

test_index = list(range(4, X.shape[0], 5))
train_index = list(X.index)
for i in test_index:
    train_index.remove(i)

test_x = X.iloc[test_index, :]
test_y = np.ravel(y.iloc[test_index, :])
train_x = X.iloc[train_index, :]
train_y = np.ravel(y.iloc[train_index, :])

#pca = PCA(n_components=2)
#compressed = pca.fit_transform(X_total)


gnb = GaussianNB()
#y_gnb = gnb.fit(train_x, train_y).predict(test_x)
#graph = confusion_matrix(test_y, y_gnb)
iteration_num = [5, 10, 15, 20, 50, 100, 200, 500, 1000, 5000]
train_err_list, test_err_list = error_list_ada(iteration_num)

plt.plot(iteration_num, train_err_list)
plt.plot(iteration_num, test_err_list)
    


svm_clf = SVC(gamma='auto')
y_svm = svm_clf.fit(train_x,train_y).predict(test_x)


#err_rate = error_rate(test_y, y_ada)
#graph = confusion_matrix(test_y, y_ada)


