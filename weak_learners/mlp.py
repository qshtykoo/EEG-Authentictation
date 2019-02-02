"""
Created on Fri Jan 18 17:58:32 2019

@author: Yunfei Xue
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
import common_utils as utils

modelName = "mlp"
Alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
HiddenLayerSize = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MaxIter = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
Activation = {'identity', 'logistic', 'tanh', 'relu'}
LearningRate = [ 0.001, 0.002, 0.005, 0.007, 0.01, 0.015, 0.0165, 0.017, 0.018, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1]

colors = {'beach': 'red', 'finger': 'green', 'rest': 'blue'}

def run(dataDicts):

    clf = MLPClassifier(solver='lbfgs', alpha=0.01, random_state=1)
    A, B, C = utils.cross_val(clf, dataDicts)
    print(A)

    # for a in Alpha:
    #     print(a)
    #     clf = MLPClassifier(solver='lbfgs', alpha=a, random_state=1)
    #     A, B, C = utils.cross_val(clf, dataDicts)
    #     print(A)

    # for layer in HiddenLayerSize:
    #     print(layer)
    #     clf = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes = (layer,), random_state=1)
    #     A, B, C = utils.cross_val(clf, dataDicts)
    #     print(A)

    # for iter in MaxIter:
    #     print(iter)
    #     clf = MLPClassifier(solver='lbfgs', alpha=0.01, max_iter=iter, random_state=1)
    #     A, B, C = utils.cross_val(clf, dataDicts)
    #     print(A)

    # for act in Activation:
    #     print(act)
    #     clf = MLPClassifier(solver='lbfgs', alpha = 0.01, activation = act, random_state=1)
    #     A, B, C = utils.cross_val(clf, dataDicts)
    #     print(A)

    # for learn in LearningRate:
    #     print(learn)
    #     clf = MLPClassifier(solver='sgd', alpha=0.01, learning_rate_init = learn, random_state=1)
    #     A, B, C = utils.cross_val(clf, dataDicts)
    #     print(A)

    warnings.filterwarnings('ignore')


    for key in dataDicts[4]:
        print(">> Dataset: " + key.upper())
        [train_x, train_y, test_x, test_y] = dataDicts[4][key]

        param_g = {"alpha": [0.001, 0.01, 0.1, 1],
                     "hidden_layer_sizes": [(50, 50, 50), (100, 50), (100)],
                     "max_iter": [100, 250, 500, 1000],
                     "activation": ['identity', 'logistic', 'relu'],
                     "solver": ['adam'],
                     "random_state":[0, 1, 2, 3, 4],
                     "learning_rate_init": [0.001, 0.005, 0.01]}

        # inData = ""
        # gs = RandomizedSearchCV(MLPClassifier(), param_distributions=param_g, scoring="accuracy", n_jobs=-1, n_iter=100)

        # inData = "ARM"
        # gs = GridSearchCV(MLPClassifier(random_state=1), param_grid=param_g, scoring="accuracy", n_jobs=-1)

        # inData = "FARM"
        # gs = GridSearchCV(MLPClassifier(random_state=1), param_grid=param_g, scoring="accuracy", n_jobs=-1)

        # inData = "WARM"
        # gs = GridSearchCV(MLPClassifier(random_state=1), param_grid=param_g, scoring="accuracy", n_jobs=-1)

        # inData = "WARM" - test best random seed
        gs = GridSearchCV(MLPClassifier(), param_grid=param_g, scoring="accuracy", n_jobs=-1)

        # inData = "NWARM"
        # gs = GridSearchCV(MLPClassifier(random_state=1), param_grid=param_g, scoring="accuracy", n_jobs=-1)

        gs.fit(train_x, train_y)

        bestParam = gs.best_params_
        print(bestParam)

        bestScore = gs.best_score_
        print(bestScore)

        utils.printToTxt(modelName, "mlp_" + key + "_best", bestParam)
        utils.printToTxt(modelName, "mlp_" + key + "_best_score", bestScore)
        utils.printToTxt(modelName, "mlp_" + key + "_cv_results", gs.cv_results_)


        # # Compute confusion matrix
        #
        # cnf_matrix = confusion_matrix(test_y, pred_y)
        # np.set_printoptions(precision=2)
        #
        # # Plot normalized confusion matrix
        # plt.figure()
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
        # print("precision (macro) ", precision, "\n")





def compareAlpha(train_x, train_y, test_x, test_y):
    accuracies = []
    for a in Alpha:
        clf = MLPClassifier(solver='lbfgs', alpha=a, hidden_layer_sizes=(100,), random_state=1)
        clf.fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, pred_y)
        accuracies.append(accuracy)

    print("highest accuracy = ", max(accuracies))
    print("alpha = ", Alpha[accuracies.index(max(accuracies))], "\n")
    return accuracies


def plotAlphaComparison(dataDict):
    plt.clf()

    for key in dataDict:
        print(">> Dataset: " + key.upper())
        [train_x, train_y, test_x, test_y] = dataDict[key]

        accuracies = compareAlpha(train_x, train_y, test_x, test_y)
        plt.plot( accuracies, color=colors[key], label=key.capitalize())

    plt.xlabel('Value of alpha (a)')
    plt.ylabel('Accuracy Score')
    plt.title('MLP')
    plt.legend()
    utils.savePlots(modelName, plt,'alpha_comparison', "png")
