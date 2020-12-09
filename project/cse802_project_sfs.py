# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:22:54 2020

@author: abhir
"""

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
feature_list = []

for f in range(1,6): #for every seed or repartitioning of the dataset
    print("-------------f is---------------:", f)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10+f, stratify=Y)
#    
#    
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
#    
#     
    clf = svm.SVC(C=10, gamma='scale', kernel='rbf');
    sfs1 = sfs(clf, k_features=10, forward=False, floating=False, verbose=2, scoring='accuracy', cv=3)
    sfs1 = sfs1.fit(x_train_scaled, y_train)
    feat_cols = list(sfs1.k_feature_idx_)
    feature_list.append(feat_cols)
    
#    feat_cols = [0,1,2,4,5,6,8,9,10]
#    clf.fit(x_train_scaled[:,feat_cols], y_train)
#    y_pred = clf.predict(x_test_scaled[:,feat_cols]) #use the best average validation score across 3 folds (their tuned parameters), to predict true test results
#    test_accuracy = accuracy_score(y_test, y_pred)*100
#    test_scores.append(test_accuracy)
#
#
#mean_test_score = mean(test_scores)    
#test_var = statistics.variance(test_scores)
