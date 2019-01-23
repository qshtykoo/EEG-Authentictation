"""
Created on Fri Jan 18 17:58:16 2019

@author: Viet Ba Hirvola
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
import common_utils as utils

modelName = "svm"
T = [1, 5, 10, 50, 100, 500, 1000]
#T=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
kernels = ['linear', 'rbf', 'poly']
colors = {'beach': 'red', 'finger': 'green', 'rest': 'blue'}

def buildSVM(train_x, train_y, test_x, kernel = 'linear'):
    svm = SVC(gamma='auto', kernel=kernel)
    svm.fit(train_x,train_y)
    y_svm = svm.predict(test_x)

    return y_svm

def buildAdaSVM(train_x, train_y, test_x, t = 5, kernel = 'linear'):
    ada = AdaBoostClassifier(SVC(gamma='auto', kernel=kernel), n_estimators=t, algorithm='SAMME')
    ada.fit(train_x, train_y)
    y_ada_svm = ada.predict(test_x)

    return y_ada_svm

def compareTs(train_x, train_y, test_x, test_y, kernel='linear'):
    accuracies = []

    for t in T:
        y_pred = buildAdaSVM(train_x, train_y, test_x, t=t, kernel=kernel)
        accuracies.append(metrics.accuracy_score(test_y, y_pred))

    maxScore = max(accuracies)
    maxT = T[accuracies.index(maxScore)]
    print("Highest accuracy obtained: " + str(maxScore) + "for t=" + str(maxT))

    return accuracies

def compareKernels(train_x, train_y, test_x, test_y, t=10):
    accuracies = []

    for kernel in kernels:
        y_pred = buildAdaSVM(train_x, train_y, test_x, kernel=kernel)
        accuracies.append(metrics.accuracy_score(test_y, y_pred))

    return accuracies

def plotTComparison(dataDict):
    plt.clf()

    for kernel in kernels:
        print(kernel.upper())
        for key in dataDict:
            print(">> Dataset: " + key.upper())

            [train_x, train_y, test_x, test_y] = dataDict[key]

            accuracies = compareTs(train_x, train_y, test_x, test_y, kernel)
            #taskAccuracies[key] = tAccuracies
            plt.plot(T, accuracies, color=colors[key], label=key.capitalize())

        plt.xlabel('Number of estimators (T)')
        plt.ylabel('Accuracy Score')
        plt.title('AdaBoost with SVM')
        plt.legend()
        utils.savePlots(modelName, plt, kernel + '_t_comparison', "png")

def plotKernelComparison(dataDict):
    plt.clf()

    for key in dataDict:
        [train_x, train_y, test_x, test_y] = dataDict[key]

        accuracies = compareKernels(train_x, train_y, test_x, test_y,)
        plt.plot(kernels, accuracies, color=colors[key], label=key.capitalize())

    plt.xlabel('SVM Kernel')
    plt.ylabel('Accuracy Score')
    plt.title('Comparison of SVM kernels in AdaBoost')
    plt.legend()
    utils.savePlots(modelName, plt, 'kernel_comparison', "png")

def run(dataDict):
    #plotTComparison(dataDict)
    #plotKernelComparison(dataDict)

    #Best: linear kernel, t doesn't change the accuracy
    for key in dataDict:
        print(">> Dataset: " + key.upper())
        [train_x, train_y, test_x, test_y] = dataDict[key]

        #Train AdaBoost with SVM
        y_ada_svm = buildAdaSVM(train_x, train_y, test_x)
        print("Accuracy: " + str(metrics.accuracy_score(test_y, y_ada_svm)) +
            ", Recall: " + str(metrics.recall_score(test_y, y_ada_svm, average='macro')) +
            ", Precision: " + str(metrics.precision_score(test_y, y_ada_svm, average='macro')))
        utils.plotCM(metrics.confusion_matrix(test_y, y_ada_svm), title= "Adaboost with SVM: " + key.capitalize())
        utils.savePlots(modelName, plt, "ada_" + key, "png")


        #Train plain SVM
        y_svm = buildSVM(train_x, train_y, test_x)
        print("Accuracy: " + str(metrics.accuracy_score(test_y, y_svm)) +
            ", Recall: " + str(metrics.recall_score(test_y, y_svm, average='macro')) +
            ", Precision: " + str(metrics.precision_score(test_y, y_svm, average='macro')))
        utils.plotCM(metrics.confusion_matrix(test_y, y_svm), title= "SVM: " + key.capitalize())
        utils.savePlots(modelName, plt, "svm_" + key, "png")
