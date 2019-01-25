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

def buildAdaSVM(train_x, train_y, test_x, T=1, kernel = 'linear'):
    ada = AdaBoostClassifier(SVC(gamma='auto', kernel=kernel), n_estimators=T, algorithm='SAMME')
    ada.fit(train_x, train_y)
    y_ada_svm = ada.predict(test_x)

    return y_ada_svm

def compareTs(train_x, train_y, test_x, test_y, kernel='linear'):
    accuracies = []

    for t in T:
        y_pred = buildAdaSVM(train_x, train_y, test_x, T=t, kernel=kernel)
        accuracies.append(metrics.accuracy_score(test_y, y_pred))

    maxScore = max(accuracies)
    maxT = T[accuracies.index(maxScore)]
    print("Highest accuracy obtained: " + str(round(maxScore,3)) + " for t=" + str(maxT))

    return accuracies

def compareKernels(train_x, train_y, test_x, test_y, t=10):
    accuracies = []

    for kernel in kernels:
        y_pred = buildAdaSVM(train_x, train_y, test_x, kernel=kernel)
        accuracies.append(metrics.accuracy_score(test_y, y_pred))

    return accuracies

def plotTComparison(dataDict, prefix=""):
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
        utils.savePlots(modelName, plt, prefix + kernel + '_t_comparison', "png")

def plotKernelComparison(dataDict, prefix=""):
    plt.clf()

    for key in dataDict:
        [train_x, train_y, test_x, test_y] = dataDict[key]

        accuracies = compareKernels(train_x, train_y, test_x, test_y,)
        plt.plot(kernels, accuracies, color=colors[key], label=key.capitalize())

    plt.xlabel('SVM Kernel')
    plt.ylabel('Accuracy Score')
    plt.title('Comparison of SVM kernels in AdaBoost')
    plt.legend()
    utils.savePlots(modelName, plt, prefix + 'kernel_comparison', "png")


def run(dataDicts):
    #plotTComparison(dataDict)
    #plotKernelComparison(dataDict)
    
    acc_ada = {"beach": 0, "finger": 0, "rest": 0}
    rec_ada = {"beach": 0, "finger": 0, "rest": 0}
    pre_ada = {"beach": 0, "finger": 0, "rest": 0}
    
    acc_ = {"beach": 0, "finger": 0, "rest": 0}
    rec_ = {"beach": 0, "finger": 0, "rest": 0}
    pre_ = {"beach": 0, "finger": 0, "rest": 0}

    #Best: linear kernel, t doesn't change the accuracy
    for dataDict in dataDicts:
        for key in dataDict:
            #print(">> Dataset: " + key.upper())
            [train_x, train_y, test_x, test_y] = dataDict[key]
    
            #Train AdaBoost with SVM
            y_ada_svm = buildAdaSVM(train_x, train_y, test_x)
            
            acc_ada[key] += metrics.accuracy_score(test_y, y_ada_svm)
            rec_ada[key] += metrics.recall_score(test_y, y_ada_svm, average='micro')
            pre_ada[key] += metrics.precision_score(test_y, y_ada_svm, average='micro')
            

            '''
            print("Accuracy: " + str(metrics.accuracy_score(test_y, y_ada_svm)) +
                ", Recall: " + str(metrics.recall_score(test_y, y_ada_svm, average='micro')) +
                ", Precision: " + str(metrics.precision_score(test_y, y_ada_svm, average='micro')))
            utils.plotCM(metrics.confusion_matrix(test_y, y_ada_svm), title= "Adaboost with SVM: " + key.capitalize())
            utils.savePlots(modelName, plt, "ada_" + key, "png")
            '''
    
            #Train plain SVM
            y_svm = buildSVM(train_x, train_y, test_x)
            
            acc_[key] += metrics.accuracy_score(test_y, y_svm)
            rec_[key] += metrics.recall_score(test_y, y_svm, average='micro')
            pre_[key] += metrics.precision_score(test_y, y_svm, average='micro')
            
            '''
            print("Accuracy: " + str(metrics.accuracy_score(test_y, y_svm)) +
                ", Recall: " + str(metrics.recall_score(test_y, y_svm, average='micro')) +
                ", Precision: " + str(metrics.precision_score(test_y, y_svm, average='micro')))
            utils.plotCM(metrics.confusion_matrix(test_y, y_svm), title= "SVM: " + key.capitalize())
            utils.savePlots(modelName, plt, "svm_" + key, "png")
            '''
    
    
    keys = ["beach", "finger", "rest"]
    
    for key in keys:
        
        acc_[key] /= len(dataDicts)
        rec_[key] /= len(dataDicts)
        pre_[key] /= len(dataDicts)
        
        acc_ada[key] /= len(dataDicts)
        rec_ada[key] /= len(dataDicts)
        pre_ada[key] /= len(dataDicts)
        
        print("dataset: " + key)
        
        print("SVM with AdaBoost")
        print("Accuracy: " + str(acc_ada[key]) + 
              ", Recall: " + str(rec_ada[key]) +
              ", Precision: " + str(pre_ada[key]))
        
        print("SVM without AdaBoost")
        print("Accuracy: " + str(acc_[key]) + 
              ", Recall: " + str(rec_[key]) +
              ", Precision: " + str(pre_[key]))
    
'''
def getPerformanceMeasures(dataDict, prefix="", T=5, kernel='linear'):
    for key in dataDict:
        print(">> Dataset: " + key.upper())
        [train_x, train_y, test_x, test_y] = dataDict[key]

        #Train AdaBoost with SVM
        y_ada_svm = buildAdaSVM(train_x, train_y, test_x, T=T, kernel=kernel)
        print(">> AdaSVM: \n")
        print("Accuracy: " + str(round(metrics.accuracy_score(test_y, y_ada_svm),3)) +
            ", Precision: " + str(round(metrics.precision_score(test_y, y_ada_svm, average='macro'),3)) +
            ", Recall: " + str(round(metrics.recall_score(test_y, y_ada_svm, average='macro'),3))
            )
        utils.plotCM(metrics.confusion_matrix(test_y, y_ada_svm), title= "Adaboost with SVM: " + key.capitalize())
        utils.savePlots(modelName, plt, prefix + "ada_" + key, "png")


        #Train plain SVM
        y_svm = buildSVM(train_x, train_y, test_x, kernel=kernel)
        print(">> SVM: \n")
        print("Accuracy: " + str(round(metrics.accuracy_score(test_y, y_svm),3)) +
            ", Precision: " + str(round(metrics.precision_score(test_y, y_svm, average='macro'),3)) +
            ", Recall: " + str(round(metrics.recall_score(test_y, y_svm, average='macro'),3))
            )
        utils.plotCM(metrics.confusion_matrix(test_y, y_svm), title= "SVM: " + key.capitalize())
        utils.savePlots(modelName, plt, prefix + "svm_" + key, "png")

def run(dataDict, inData=""):
    prefix = inData if inData == "" else inData + "_"

    plotTComparison(dataDict, prefix)
    #plotKernelComparison(dataDict, prefix)
    #getPerformanceMeasures(dataDict, prefix)

    #Best: linear kernel, t doesn't change the accuracy
'''