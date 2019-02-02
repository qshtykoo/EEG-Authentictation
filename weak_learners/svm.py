"""
Created on Fri Jan 18 17:58:16 2019

@author: Viet Ba Hirvola
"""
#import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from matplotlib import pyplot as plt
import common_utils as utils
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.decomposition import KernelPCA
from matplotlib.colors import ListedColormap


modelName = "svm"
T = [1, 10, 15, 25, 50, 75, 100, 250, 500, 750, 1000]

kernels = ['linear', 'rbf', 'poly']
colors = {'beach': 'red', 'finger': 'green', 'rest': 'blue'}



def getSVMScoreByParameters(parameters, train_x, train_y, test_x, test_y):
    #dictionary with Ada parameters
    if 'n_estimators' in parameters:
        clf = AdaBoostClassifier(SVC(kernel=parameters['base_estimator__kernel'], C=parameters['base_estimator__C'], gamma=parameters['base_estimator__gamma'], degree=parameters['base_estimator__degree']), n_estimators=parameters['n_estimators'], algorithm='SAMME')
    #dictionary with plain SVM parameters
    elif 'kernel' in parameters:
        clf = SVC(kernel=parameters['kernel'], C=parameters['C'], gamma=parameters['gamma'], degree=parameters['degree'])
    else:
        print("Error! Dictionary does not contain correct parameters.")
        return


    clf.fit(train_x,train_y)

    start = time.time()
    y_pred = clf.predict(test_x)
    dur = time.time() - start

    acc = round(metrics.accuracy_score(test_y, y_pred), 3)
    print(acc)

    return acc, dur * 1000


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

def compareCV(dataDict):

    svm = SVC(kernel='linear')
    acc_, _, _ = utils.cross_val(svm, dataDict)

    print(acc_)

    ada = AdaBoostClassifier(svm, algorithm='SAMME')
    acc_, _, _ = utils.cross_val(ada, dataDict)
    print(acc_)



def runGridSearch(dataDict, prefix, ada=False):
    c_list = [0.001, 0.01, 0.1, 1, 10, 50, 100]
    gamma_list = [0.001, 0.01, 0.1, 1]
    degree_list = [2, 3, 4, 5]
    suffix = ""
    if ada:
        print("\nGRID SEARCH: AdaSVM\n")
        parameters = {'n_estimators': T, 'algorithm': ['SAMME'], 'base_estimator__kernel': kernels, 'base_estimator__C': c_list, 'base_estimator__gamma': gamma_list, 'base_estimator__degree': degree_list}
        suffix = "ada_"
        estimator = AdaBoostClassifier(SVC())
        n = 25

    else:
        print("\nGRID SEARCH: plain SVM\n")
        parameters = {'kernel': kernels, 'C': c_list, 'gamma': gamma_list, 'degree': degree_list}
        estimator = SVC()
        n = 50

    for key in dataDict:
        print(">> Dataset: " + key.upper())
        [train_x, train_y, test_x, test_y] = dataDict[key]

        #Run RandomizedSearchCV for the original data due to large feature set in it
        svm_grid = GridSearchCV(estimator, parameters, cv=4, n_jobs=-1) if prefix != "" else RandomizedSearchCV(estimator, parameters, cv=4, n_jobs=-1,
                                   n_iter=n)

        svm_grid.fit(train_x,train_y)

        #print(svm_grid.best_score_)
        #utils.printToTxt(modelName, prefix + "svm_" + key + "_best_score", svm_grid.best_score_)
        best = svm_grid.best_params_
        print(best)
        utils.printToTxt(modelName, prefix + "svm_" + suffix + key + "_best", best)
        utils.printToTxt(modelName, prefix + "svm_" + suffix + key + "_cv_results", svm_grid.cv_results_)
        utils.printToTxt(modelName, prefix + "svm_" + suffix + key + "_best", "\n CV Score: " + str(svm_grid.best_score_), "a")

        acc, dur = getSVMScoreByParameters(best, train_x, train_y, test_x, test_y)

        utils.printToTxt(modelName, prefix + "svm_" + suffix + key + "_best", "\n Accuracy score: " + str(acc), "a")
        utils.printToTxt(modelName, prefix + "svm_" + suffix + key + "_best", "\n Duration [ms]: " + str(dur), "a")

    return best, acc, dur

##########################################################################################################
# BoundaryLine source: https://www.kaggle.com/jsultan/visualizing-classifier-boundaries-using-kernel-pca #
##########################################################################################################

def BoundaryLine(kernel, algo, algo_name, x_train, y_train, x_test, y_test):
    reduction = KernelPCA(n_components=2, kernel = kernel)
    x_train_reduced = reduction.fit_transform(x_train)
    x_test_reduced = reduction.transform(x_test)

    classifier = algo
    classifier.fit(x_train_reduced, y_train)

    y_pred = classifier.predict(x_test_reduced)


    #Boundary Line
    X_set, y_set = np.concatenate([x_train_reduced, x_test_reduced], axis = 0), np.concatenate([y_train, y_test], axis = 0)
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.5, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('{} Boundary Line with {} PCA' .format(algo_name, kernel))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.xticks(fontsize = 3)
    plt.yticks(fontsize = 3)

    plt.show()


def run(dataDict, inData=""):

    prefix = inData if inData == "" else inData + "_"
    #compareCV(dataDict)
    #plotTComparison(dataDict, prefix)
    #plotKernelComparison(dataDict, prefix)
    #getPerformanceMeasures(dataDict, prefix)

    #Run whole parameter tuning
    for isAda in [False, True]:
        runGridSearch(dataDict, prefix, isAda)

    #Checking single combinations
    #[train_x, train_y, test_x, test_y] = dataDict["beach"]
    #best = {'C': 0.001, 'degree': 3, 'gamma': 0.1, 'kernel': 'poly'}
    #acc, dur = getSVMScoreByParameters(best, train_x, train_y, test_x, test_y)
    #print(acc)
    #print(dur)
    #BoundaryLine('poly', SVC(kernel=best['kernel'], C=best['C'], gamma=best['gamma'], degree=best['degree']), "SVM - poly", train_x, train_y, test_x, test_y)
