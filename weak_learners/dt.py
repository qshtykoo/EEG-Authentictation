# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:59:23 2019

@author: Administrator
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

from sklearn import tree

from sklearn.tree import export_graphviz
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

from sklearn.ensemble import AdaBoostClassifier

import common_utils as utils

modelName = "dt"

def run_grid_search(dataDicts, ada=False):
    depths = [1, 2, 3, 5, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 25, 27, 29, 30, 35]
    accuracies_beach = []
    accuracies_finger = []
    accuracies_rest = []
    
    for depth in depths:
        if ada == False:
            clf_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
            accuracy, _, _ = utils.cross_val(clf_tree, dataDicts)
            accuracies_beach.append(accuracy["beach"])
            accuracies_finger.append(accuracy["finger"])
            accuracies_rest.append(accuracy["rest"])
        else:
            if depth > 15:
                accuracies_beach.append(0)
                accuracies_finger.append(0)
                accuracies_rest.append(0)
            else:
                clf_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
                accuracy, _, _ = utils.cross_val(clf_tree, dataDicts, ada=True)
                accuracies_beach.append(accuracy["beach"])
                accuracies_finger.append(accuracy["finger"])
                accuracies_rest.append(accuracy["rest"])
    
    keys = ["beach", "finger", "rest"]
    accuracies = {"beach": accuracies_beach, "finger": accuracies_finger, "rest": accuracies_rest, "depths": depths}
    for key in keys:
        data = accuracies[key]
        ind = np.argmax(data)
        print("dataset: " + key)   
        if ada == False:
            print("optimal maximum depth value for Decision Tree: ")
            print("depth value: " + str(depths[ind]) + ", accuracy: " + str(data[ind]))
        else:
            print("optimal maximum depth value for Decision Tree with AdaBoost: ")
            print("depth value: " + str(depths[ind]) + ", accuracy: " + str(data[ind]))
    print("depths")
    print(depths)
    print("beach data")
    print(accuracies_beach)
    print("finger data")
    print(accuracies_finger)
    print("rest data")
    print(accuracies_rest)
    
    
    #return accuracies

def plot_tree(key, tree, feature_names, labels):
    '''
    T: iteration number
    key: beach, finger or rest
    labels: supposedly it is a list of integers
    '''
    
    estimator = tree
    class_names = []
    
    img_name = 'tree' + key + '.dot'

    for label in labels:
        class_names.append(str(label))
    
    export_graphviz(estimator, out_file=img_name, 
                feature_names = feature_names,
                class_names = class_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    '''
    #dispaly in python
    graph = Source(export_graphviz(estimator, out_file=None, 
                feature_names = feature_names,
                class_names = class_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True))
    
    display(SVG(graph.pipe(format='svg')))
    '''

def plot_feature_importances(clf, feature_names, clf_name = '', top_n = 10, figsize=(8,8), T=None, save_fig=True):
    '''
    clf: fitted classifier
    feature_names = train_x.columns
    train_x is pandas dataFrame
    '''
    
    title="Feature Importances of " + clf_name
    
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = feature_names
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    if save_fig == True:
        if T == None:
            utils.savePlots('dt', plt, name=clf_name)
        else:
            utils.savePlots('dt', plt, name=clf_name + str(T))
    else:
        plt.close()
        print("not saving the plot of " + clf_name)
    #plt.show()
    
    return feat_imp

def select_best_features(feat_imp_list, ada=True):
    '''
    Select the top 3 features
    '''
    print("start selecting the best features..")
    feature_list = []
    if type(feat_imp_list) == list:
        for feat_imp in feat_imp_list:
            feat_imp = feat_imp.sort_values(by='importance', ascending=False)
            feat_imp = feat_imp.iloc[:3]
            feature_list = np.concatenate((feature_list, feat_imp.index.values))
    
    print("feature list: ")
    print(feature_list)
    
    unique_nums = np.unique(feature_list)
    feat_dic = {}
    for num in unique_nums:
        feat_dic[str(num)] = 0
    for num in feature_list:
        feat_dic[str(num)] += 1
    
    print("feature counts: ")
    print(feat_dic)
    
    return feat_dic

def plot_best_features(dataDict, task=None):
    
    keys = ['beach', 'finger', 'rest']
    '''
    top features for Decision Tree
    '''
    features = {'beach': [2, 27], 'finger': [23, 30], 'rest': [38,1]}
    feature_dict = {}

    for key in keys:
        feature_set = []
        X, Y, _, _ = dataDict[key]
        top_features = features[key]
        for fea in top_features:
            feature_set.append(X.iloc[:, fea])
        feature_dict[key] = feature_set
        feature_dict[key+'label'] = Y
    
    if task == None:
        print("Error: no key specified")
        return
    else:
        feats = feature_dict[task]
        fe1 = feats[0]
        fe2 = feats[1]
        plt.scatter(fe1, fe2, c=feature_dict[task+'label'])
        plt.colorbar()
        plt.xlabel("feature " + str(features[task][0]))
        plt.ylabel("feature " + str(features[task][1]))
        plt.title("Feature Space of " + task + "Data")
        plt.savefig(task + ".png")
        plt.show()
    
        
        
    

def test_accuracy(dataDict, depth, key, ada=False, T=None):
    
    train_x, train_y, test_x, test_y = dataDict[key]
    if ada == False:
        clf = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(train_x, train_y)
        plot_feature_importances(clf, train_x.columns, clf_name = key + " Decision Tree", save_fig=False)
    else:
        clf_tree = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
        if T == None:
            print("Error: iteration number T cannot be None")
            return
        else:
            if type(T) == list:
                feat_imp_list = []
                for t in T:
                    clf = AdaBoostClassifier(base_estimator = clf_tree, n_estimators=t, random_state=0)
                    clf.fit(train_x, train_y)
                    feat_imp = plot_feature_importances(clf, train_x.columns, clf_name = key + " AdaBoost", T=t, save_fig=False)
                    feat_imp_list.append(feat_imp)
            else:
                clf = AdaBoostClassifier(base_estimator = clf_tree, n_estimators=T, random_state=0)
                clf.fit(train_x, train_y)
                print("WARNING: the return feature importance list is only one single dataFrame")
    
    
    '''
    if ada == False:
        plot_tree(key, clf, list(train_x.columns.values), np.unique(train_y))
    '''
        
    #select top 3 features
    if ada == True and type(T) == list:
        feature_list = select_best_features(feat_imp_list)
    
    start = time.time()
    
    y_pred = clf.predict(test_x)
    
    end = time.time()
    
    accuracy = metrics.accuracy_score(test_y, y_pred)
    
    print("prediction time: ", 1000*(end-start))
    print("The test accuracy is: ", accuracy)
    
    
    
        

def run(dataDicts, ada=False, max_depth=1):
    
    
    if ada == False:
        clf_tree = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        accuracy, precision, recall = utils.cross_val(clf_tree, dataDicts)
    else:
        clf_tree = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        accuracy, precision, recall = utils.cross_val(clf_tree, dataDicts, ada=True)
    
    keys = ["beach", "finger", "rest"]
    for key in keys:   
        print("dataset: " + key)   
        if ada == False:
            print("Decision Tree")
            print("Accuracy: " + str(accuracy[key]) + 
              ", Recall: " + str(recall[key]) +
              ", Precision: " + str(precision[key]))
        else:
            print("Decision Tree with AdaBoost")
            print("Accuracy: " + str(accuracy[key]) + 
              ", Recall: " + str(recall[key]) +
              ", Precision: " + str(precision[key]))
            
            
        
            
        # Compute confusion matrix
        #cnf_matrix = utils.cal_confusion_matrix(test_y, pred_y)
        #np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        #plt.figure()
        #utils.plotCM(cnf_matrix, title=key.capitalize()+' Confusion Matrix')
        #utils.savePlots(modelName, plt, key)

