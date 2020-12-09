# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:08:56 2020

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
#X = data.drop(['residual_sugar'], axis=1)
#X = data.drop(['density'], axis=1)
f =4
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10+f, stratify=Y)

#scaler = preprocessing.StandardScaler().fit(x_train)
#x_train_scaled = scaler.transform(x_train)
#x_test_scaled = scaler.transform(x_test)

idx = y_train[:] == 3
mu_3 = np.mean(x_train.values[idx], axis=0)
cov_3 = np.cov(x_train.values[idx], bias=True, rowvar=False)
cov_3 = np.diag(np.diag(cov_3))

idx = y_train[:] == 4
mu_4 = np.mean(x_train.values[idx], axis=0)
cov_4 = np.cov(x_train.values[idx], bias=True, rowvar=False)
cov_4 = np.diag(np.diag(cov_4))


idx = y_train[:] == 5
mu_5 = np.mean(x_train.values[idx], axis=0)
cov_5 = np.cov(x_train.values[idx], bias=True, rowvar=False)
cov_5 = np.diag(np.diag(cov_5))


idx = y_train[:] == 6
mu_6 = np.mean(x_train.values[idx], axis=0)
cov_6 = np.cov(x_train.values[idx], bias='True', rowvar=False)
cov_6 = np.diag(np.diag(cov_6))


idx = y_train[:] == 7
mu_7 = np.mean(x_train.values[idx], axis=0)
cov_7 = np.cov(x_train.values[idx], bias='True', rowvar=False)
cov_7 = np.diag(np.diag(cov_7))


idx = y_train[:] == 8
mu_8 = np.mean(x_train.values[idx], axis=0)
cov_8 = np.cov(x_train.values[idx], bias='True', rowvar=False)
cov_8 = np.diag(np.diag(cov_8))


y_pred = np.zeros(shape=(x_test.shape[0],1))
for i in range(np.shape(x_test)[0]):
    p3 = multivariate_normal.pdf(x_test.values[i], mean=mu_3, cov=cov_3, allow_singular=True)
    p4 = multivariate_normal.pdf(x_test.values[i], mean=mu_4, cov=cov_4, allow_singular=True)
    p5 = multivariate_normal.pdf(x_test.values[i], mean=mu_5, cov=cov_5, allow_singular=True)
    p6 = multivariate_normal.pdf(x_test.values[i], mean=mu_6, cov=cov_6, allow_singular=True)
    p7 = multivariate_normal.pdf(x_test.values[i], mean=mu_7, cov=cov_7, allow_singular=True)
    p8 = multivariate_normal.pdf(x_test.values[i], mean=mu_8, cov=cov_8, allow_singular=True)
    pred_label = np.argmax([p3, p4, p5, p6, p7, p8])
    pred_label += 3
    y_pred[i] = pred_label

test_accuracy = accuracy_score(y_test, y_pred)*100
confusionmat = confusion_matrix(y_test, y_pred)  
    