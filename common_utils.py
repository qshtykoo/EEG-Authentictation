"""
Created on Sat Jan 19 13:48:05 2019

@author: Viet Ba Hirvola
"""
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd

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

