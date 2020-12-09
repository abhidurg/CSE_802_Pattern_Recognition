# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:35:27 2020

@author: abhir
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statistics import mean
from sklearn.preprocessing import Normalizer
import statistics 
from sklearn.metrics import confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data  =  pd.read_csv('winequality-red.csv', sep=';')
data.columns = data.columns.str.replace(' ','_')
X = data.drop(['quality'], axis = 1)
Y = data['quality']

#n_comp_results = []
#for n in range(1,6):
val_scores = []
test_scores = []
test_std_dev = []
parameter_list = []
confusionmat_list = []
subsets_list = []
for f in range(1,6): #for every seed or repartitioning of the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10+f, stratify=Y)
    #min_max_scaler = MinMaxScaler()
    #x_train_minmax = min_max_scaler.fit_transform(x_train)
    #x_test_minmax = min_max_scaler.transform(x_test)
    
    #normalizer = preprocessing.Normalizer().fit(x_train)
    #x_train_normalized = normalizer.transform(x_train)
    #x_test_normalized = normalizer.transform(x_test)
    

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
       
#    clf = GridSearchCV(svm.SVC(), {
#            'C': [0.001, 0.01, 0.1, 1, 10, 100],
#            'kernel': ['rbf', 'linear'],
#            'gamma': ['auto', 'scale', 0.001, 0.01, 0.1, 1, 10, 100]       
#            }, cv=3, return_train_score=False)
    feat_cols = [0,1,2,4,5,6,8,9,10]
#    clf = GridSearchCV(tree.DecisionTreeClassifier(), {
#            'max_depth': [5, 10, 50, 100, 200, 300, 400, 500],
#            'max_features': [1,2,3,4,5,6,7,8,9],
#            }, cv=3, return_train_score=False)
#    clf = GridSearchCV(RandomForestClassifier(), {
#            'n_estimators': [5, 10, 50, 100, 200, 300, 400, 500],
#            'max_features': [1,2,3,4,5,6,7,8,9],
#            }, cv=3, return_train_score=False)    
    clf = GridSearchCV(KNeighborsClassifier(), {
            'n_neighbors': list(range(1,500,5)),
            'weights': ['uniform','distance'],
            'p': [1,2]
            }, cv=3, return_train_score=False)  

    clf.fit(x_train_scaled[:,feat_cols], y_train)
    #df = pd.DataFrame(clf.cv_results_)
    y_pred = clf.predict(x_test_scaled[:,feat_cols]) #use the best average validation score across 3 folds (their tuned parameters), to predict true test results
    test_accuracy = accuracy_score(y_test, y_pred)*100
    confusionmat = confusion_matrix(y_test, y_pred)
    val_scores.append(clf.best_score_*100)
    test_scores.append(test_accuracy)
    best_params = clf.best_params_
    parameter_list.append(best_params)
    confusionmat_list.append(confusionmat)
#mean_val_score = mean(val_scores)

mean_test_score = mean(test_scores)    
test_var = statistics.variance(test_scores)

#results = [mean_val_score, mean_test_score, test_std]
    #n_comp_results.append(results)