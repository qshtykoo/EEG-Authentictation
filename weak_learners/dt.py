# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:59:23 2019

@author: Administrator
"""
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

import common_utils as utils

modelName = "dt"
T = [1, 5, 10, 50, 100, 500, 1000]

def error_list_ada(iteration_num, train_x, train_y, test_x, test_y, weak_learner=None, accuracy=False):
    test_err_list = []
    #train_err_list = []
    for T in iteration_num:
        ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=T)
        y_ada_test = ada.fit(train_x, train_y).predict(test_x)
        #y_ada_train = ada.fit(train_x, train_y).predict(train_x)
        test_err_list.append(utils.error_rate(test_y, y_ada_test, accuracy))
        #train_err_list.append(error_rate(train_y, y_ada_train, accuracy))
    if accuracy == False:
        ind = np.argmin(test_err_list)
    else:
        ind = np.argmax(test_err_list)
        
    targeted_T = iteration_num[ind]
    ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=targeted_T)
    targeted_y_ada = ada.fit(train_x, train_y).predict(test_x)
    
    return test_err_list, targeted_y_ada

def run(dataDict, ada=False):
    #Iterate through datasets
    for key in dataDict:
        print(">> Dataset: " + key.upper())

        [train_x, train_y, test_x, test_y] = dataDict[key]

        print("Train: X " + str(train_x.shape) + ", Y " + str(train_y.shape)
        + "\nTest: X " + str(test_x.shape) + ", Y " + str(test_y.shape) + "\n")
        
        if ada == False:
            clf_tree = tree.DecisionTreeClassifier()     
            pred_y = clf_tree.fit(train_x, train_y).predict(test_x)
        else:
            test_err_list, pred_y = error_list_ada(T, train_x, train_y, test_x, test_y, accuracy=True)
            #plot accuracy curve over iteration numbers
            plot_colors = {'beach':'red', 'finger':'blue', 'rest':'green'}
            plt.plot(T, test_err_list, color = plot_colors[key], label=key)
            plt.legend()
            plt.xlabel('Number of estimators (T)')
            plt.ylabel('Accuracy Score')
            plt.title('AdaBoost with Decision Stump')
            
        # Compute confusion matrix
        #cnf_matrix = utils.cal_confusion_matrix(test_y, pred_y)
        #np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        #plt.figure()
        
        


        #utils.plotCM(cnf_matrix, title=key.capitalize()+' Confusion Matrix')
        #utils.savePlots(modelName, plt, key)


        accuracy = metrics.accuracy_score(test_y, pred_y)
        print("accuracy ",accuracy)

        recall = metrics.recall_score(test_y, pred_y, average='macro')
        print("recall (macro) ", recall)

        precision = metrics.precision_score(test_y, pred_y, average='macro')
        print("precision (macro) ", precision)