"""
Created on Sat Jan 19 13:48:05 2019

@author: Viet Ba Hirvola
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import warnings


def plotCM(cm, normalize = False, title='Confusion matrix', cmap=plt.cm.Blues):
    #clean plt in case there's some previous plot
    plt.clf()
    plt.cla()
    plt.close()

    #create labels
    labels = np.arange(27)
    labels += 1

    plot_confusion_matrix(cm, classes = labels, normalize=normalize, title = title, cmap=cmap)

##########################################################################################################################
# plot_confusion_matrix source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html #
##########################################################################################################################
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:

        #print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def printToTxt(model, outName, data, operation="w"):
    dir = join("plots", model, "txt")
    with open(join(dir, outName + ".txt"), operation) as txt_file:
        print(data, file=txt_file)
        #txt_file.write(data)

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

def savePlots(model, plot, name, type = "png"):
    '''
    model: a string - SVM, Decision Tree, etc.
    '''
    dir = join("plots", model)
    plot.savefig(join(dir, name + "." + type))
    plot.clf()
    plot.cla()
    plot.close()

def cal_confusion_matrix(y_true, y_pred):
    if isinstance(y_true, pd.core.frame.DataFrame):
        y_true = np.ravel(y_true.values) #return a contiguous flattened array
    num_class = int(np.max(y_true))
    c_matrix = np.zeros((num_class, num_class))
    for i in range(len(y_true)):
        x = int(y_true[i])
        y = int(y_pred[i])
        c_matrix[x-1, y-1] += 1
    return c_matrix

def cross_val_1key(clf, dataDicts, key):
    '''
    5-fold cross validation for one key (beach, finger or rest)
    clf: classifier, an object
    dataDicts: a list of data dictionaries
    '''
    warnings.filterwarnings('ignore')
    acc_ = 0
    rec_ = 0
    pre_ = 0

    for dataDict in dataDicts:
        [train_x, train_y, test_x, test_y] = dataDict[key]


        pred_y = clf.fit(train_x, train_y).predict(test_x)

        acc_ += metrics.accuracy_score(test_y, pred_y)
        rec_ += metrics.recall_score(test_y, pred_y, average='macro')
        pre_ += metrics.precision_score(test_y, pred_y, average='macro')


    acc_ /= len(dataDicts)
    rec_ /= len(dataDicts)
    pre_ /= len(dataDicts)

    return round(acc_,3), round(pre_,3), round(rec_,3)

def cross_val(clf, dataDicts, ada=False):
    '''
    4-fold cross validation
    clf: classifier, an object
    dataDicts: a list of data dictionaries
    output: average accuracy, recall and precision of all validation datasets
    '''
    warnings.filterwarnings('ignore')
    acc_ = {"beach": 0, "finger": 0, "rest": 0}
    rec_ = {"beach": 0, "finger": 0, "rest": 0}
    pre_ = {"beach": 0, "finger": 0, "rest": 0}

    for dataDict in dataDicts:
        for key in dataDict:
            #print(">> Dataset: " + key.upper())
            [train_x, train_y, test_x, test_y] = dataDict[key]

            if ada == False:
                pred_y = clf.fit(train_x, train_y).predict(test_x)
            else:
                T = [1, 5, 10, 50, 100, 500, 1000]
                _, pred_y = error_list_ada(T, train_x, train_y, test_x, test_y, weak_learner=clf)

            acc_[key] += metrics.accuracy_score(test_y, pred_y)
            rec_[key] += metrics.recall_score(test_y, pred_y, average='macro')
            pre_[key] += metrics.precision_score(test_y, pred_y, average='macro')

    keys = ["beach", "finger", "rest"]

    for key in keys:
        acc_[key] /= len(dataDicts)
        rec_[key] /= len(dataDicts)
        pre_[key] /= len(dataDicts)


    #Round to 3 decimal points
    for k, v in acc_.items():
            acc_[k] = round(v, 3)

    for k, v in acc_.items():
            pre_[k] = round(v, 3)

    for k, v in acc_.items():
            rec_[k] = round(v, 3)

    return acc_, pre_, rec_



def error_list_ada(iteration_num, train_x, train_y, test_x, test_y, weak_learner=None, accuracy=False):
    test_err_list = []
    #train_err_list = []
    for T in iteration_num:
        ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=T, random_state=0)
        y_ada_test = ada.fit(train_x, train_y).predict(test_x)
        #y_ada_train = ada.fit(train_x, train_y).predict(train_x)
        test_err_list.append(error_rate(test_y, y_ada_test, accuracy))
        #train_err_list.append(error_rate(train_y, y_ada_train, accuracy))
    if accuracy == False:
        ind = np.argmin(test_err_list)
    else:
        ind = np.argmax(test_err_list)

    targeted_T = iteration_num[ind]
    ada = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=targeted_T, random_state=0)
    targeted_y_ada = ada.fit(train_x, train_y).predict(test_x)

    return test_err_list, targeted_y_ada
