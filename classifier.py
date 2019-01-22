# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:05:21 2019

@author: Ramon Que
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn import tree

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import common_utils as utils

def error_rate(y_true, y, accuracy=False):
    if isinstance(y_true, pd.core.frame.DataFrame):
        y_true = np.ravel(y_true.values) #return a contiguous flattened array
    err = 0
    for i in range(len(y_true)):
        if y[i] != y_true[i]:
            err = err + 1
    if accuracy == True:
        return 1 - (err / len(y_true))
    else:
        return err / len(y_true)

def confusion_matrix(y_true, y_pred):
    if isinstance(y_true, pd.core.frame.DataFrame):
        y_true = np.ravel(y_true.values) #return a contiguous flattened array
    num_class = int(np.max(y_true))
    c_matrix = np.zeros((num_class, num_class))
    for i in range(len(y_true)):
        x = int(y_true[i])
        y = int(y_pred[i])
        c_matrix[x-1, y-1] += 1
    return c_matrix

def error_list_ada(iteration_num, weak_learner=None, accuracy=False):
    test_err_list = []
    #train_err_list = []
    for T in iteration_num:
        ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=T)
        y_ada_test = ada.fit(train_x, train_y).predict(test_x)
        #y_ada_train = ada.fit(train_x, train_y).predict(train_x)
        test_err_list.append(error_rate(test_y, y_ada_test, accuracy))
        #train_err_list.append(error_rate(train_y, y_ada_train, accuracy))
    if accuracy == False:
        ind = np.argmin(test_err_list)
    else:
        ind = np.argmax(test_err_list)
        
    targeted_T = iteration_num[ind]
    ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=targeted_T)
    targeted_y_ada = ada.fit(train_x, train_y).predict(test_x)
    
    return test_err_list, targeted_y_ada 

def data_separation(data, test_trial=4, step=5):
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

beach_data = pd.read_csv("beachARMP.csv", header=None)
finger_data = pd.read_csv("fingerARMP.csv", header=None)
rest_data = pd.read_csv("restARMP.csv", header=None)

total_data = [beach_data, finger_data, rest_data]

#train_x, train_y, test_x, test_y = data_separation(beach_data)

#X_total = np.concatenate((beach_data.iloc[:, :402], finger_data.iloc[:, :402], rest_data.iloc[:, :402]))
#y_total = np.concatenate((beach_data.iloc[:, 402:], finger_data.iloc[:, 402:], rest_data.iloc[:, 402:]))

#pca = PCA(n_components=2)
#compressed = pca.fit_transform(X_total)
plot_colors = ['red', 'blue', 'green']
line_labels = ['beach', 'finger', 'rest']
count = 0
for data in total_data:
    train_x, train_y, test_x, test_y = data_separation(data)
    #iteration_num = [1, 5, 10, 50, 100, 500, 1000]
    #test_err_list, polar_y_ada = error_list_ada(iteration_num, accuracy=True)
    clf_tree = tree.DecisionTreeClassifier()
    y_pred = clf_tree.fit(train_x, train_y).predict(test_x)
    
    svm_clf = SVC(gamma ='auto', kernel='poly', degree=3, coef0=0.015625)
    y_svm = svm_clf.fit(train_x,train_y).predict(test_x)
    
    graph = confusion_matrix(test_y, y_svm)
    utils.plotCM(graph, title="SVM - Polynomial: "+line_labels[count])
    utils.savePlots("dt", plt, "CM_SVM_ARMP_" + line_labels[count], "png")
    #plt.plot(iteration_num, test_err_list, color = plot_colors[count], label=line_labels[count])
    count += 1


plt.legend()
plt.xlabel('Number of estimators (T)')
plt.ylabel('Accuracy Score')
plt.title('AdaBoost with Decision Stump')
#graph = confusion_matrix(test_y, y_gnb)



'''
svm_clf = SVC(gamma='auto')
y_svm = svm_clf.fit(train_x,train_y).predict(test_x)

#gmm = GaussianMixture(n_components=20)
#y_gmm = gmm.fit(train_x,train_y).predict(test_x)


err_rate = error_rate(test_y, y_svm)
#graph = confusion_matrix(test_y, y_gnb)
'''

