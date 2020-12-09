# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:12:23 2020

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
from scipy.stats import multivariate_normal

data  =  pd.read_csv('winequality-red.csv', sep=';')
data.columns = data.columns.str.replace(' ','_')
X = data.drop(['quality'], axis = 1)
Y = data['quality']
f =3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10+f, stratify=Y)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
h_n = 10 #window width
parzen_prediction = np.zeros(shape=(x_test.shape[0],1)) #label prediction for test data

for i in range(np.shape(x_test_scaled)[0]): #for every test sample
    pn_3 = 0
    pn_4 = 0
    pn_5 = 0
    pn_6 = 0
    pn_7 = 9
    pn_8 = 0
    for j in range(np.shape(x_train_scaled)[0]): #for every training sample
        if y_train.values[j] == 3:
            pn_3 = pn_3 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
        elif y_train.values[j] == 4:
            pn_4 = pn_4 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
        elif y_train.values[j] == 5:
            pn_5 = pn_4 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
        elif y_train.values[j] == 6:
            pn_6 = pn_4 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
        elif y_train.values[j] == 7:
            pn_7 = pn_4 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
        elif y_train.values[j] == 8:
            pn_8 = pn_4 + (1/h_n)* multivariate_normal.pdf((x_test_scaled[i]- x_train_scaled[j])/h_n,mean=[0,0,0,0,0,0,0,0,0,0,0], cov=np.identity(11))
    pn_3 = pn_3/h_n
    pn_4 = pn_4/h_n
    pn_5 = pn_5/h_n
    pn_6 = pn_6/h_n
    pn_7 = pn_7/h_n
    pn_8 = pn_8/h_n
    pred_label = np.argmax([pn_3, pn_4, pn_5, pn_6, pn_7, pn_8])
    pred_label += 3
    parzen_prediction[i] = pred_label
test_accuracy = accuracy_score(y_test, parzen_prediction)*100
confusionmat = confusion_matrix(y_test, parzen_prediction) 