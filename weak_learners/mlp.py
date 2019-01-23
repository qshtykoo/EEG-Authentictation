"""
Created on Fri Jan 18 17:58:32 2019

@author: Yunfei Xue
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import common_utils as utils

modelName = "mlp"

def run(dataDict):
    #Iterate through datasets
    for key in dataDict:
        print(">> Dataset: " + key.upper())

        [train_x, train_y, test_x, test_y] = dataDict[key]

        print("Train: X " + str(train_x.shape) + ", Y " + str(train_y.shape)
        + "\nTest: X " + str(test_x.shape) + ", Y " + str(test_y.shape) + "\n")

        clf = MLPClassifier(solver='lbfgs', alpha=100, hidden_layer_sizes=(100,), random_state=1)
        clf.fit(train_x, train_y)

        pred_y = clf.predict(test_x)
        # print(pred_y.shape)
        proba_y = clf.predict_proba(test_x)
        # print(proba_y)

        # Compute confusion matrix

        cnf_matrix = confusion_matrix(test_y, pred_y)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure()


        utils.plotCM(cnf_matrix, title=key.capitalize()+' Confusion Matrix')
        utils.savePlots(modelName, plt, key)


        accuracy = metrics.accuracy_score(test_y, pred_y)
        print("accuracy ",accuracy)

        recall = metrics.recall_score(test_y, pred_y, average='macro')
        print("recall (macro) ", recall)

        precision = metrics.precision_score(test_y, pred_y, average='macro')
        print("precision (macro) ", precision)



