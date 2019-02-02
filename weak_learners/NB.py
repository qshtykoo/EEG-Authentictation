# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:48:47 2019

@author: Administrator
"""

from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import common_utils as utils

modelName = "NB"

def run(dataDicts):


    clf = GaussianNB()
    a, p, r = utils.cross_val(clf, dataDicts)
    print(a)


    # #Iterate through datasets
    # for key in dataDict:
    #     print(">> Dataset: " + key.upper())
    #
    #     [train_x, train_y, test_x, test_y] = dataDict[key]
    #
    #     print("Train: X " + str(train_x.shape) + ", Y " + str(train_y.shape)
    #     + "\nTest: X " + str(test_x.shape) + ", Y " + str(test_y.shape) + "\n")
    #
    #     gnb = GaussianNB()
    #     gnb.fit(train_x, train_y)
    #     pred_y = gnb.predict(test_x)

        # # Compute confusion matrix
        # cnf_matrix = utils.cal_confusion_matrix(test_y, pred_y)
        # np.set_printoptions(precision=2)
        #
        # # Plot normalized confusion matrix
        # plt.figure()
        #
        #
        # utils.plotCM(cnf_matrix, title=key.capitalize()+' Confusion Matrix')
        # utils.savePlots(modelName, plt, key)
        #
        #
        # accuracy = metrics.accuracy_score(test_y, pred_y)
        # print("accuracy ",accuracy)
        #
        # recall = metrics.recall_score(test_y, pred_y, average='macro')
        # print("recall (macro) ", recall)
        #
        # precision = metrics.precision_score(test_y, pred_y, average='macro')
        # print("precision (macro) ", precision)