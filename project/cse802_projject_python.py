import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

data  =  pd.read_csv('winequality-red.csv', sep=';')
data.columns = data.columns.str.replace(' ','_')
#data.info()
#data.describe()
X = data.drop(['quality'], axis = 1)
Y = data['quality']
#X_normalized = preprocessing.normalize(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 10, stratify=Y)


#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf,x_train,y_train,cv=5)
#print(scores)
scores = []
for f in range(5):
    x_true_train, x_valid, y_true_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = f+1, stratify=y_train)
    x_true_train_scaled = preprocessing.scale(x_true_train)
    scaler = preprocessing.StandardScaler().fit(x_true_train)
    x_valid_transormed = scaler.transform(x_valid)
    #file_name1 = "Validation_data"
    #file_name1 = file_name1 + str(f) + ".csv"
    #pd.DataFrame(x_valid).to_csv(file_name1)
    clf = svm.SVC(kernel='rbf', C=1, gamma=0.5).fit(x_true_train_scaled, y_true_train) #rbf kernel best was 67%
    #clf = LinearSVC().fit(x_true_train_scaled, y_true_train)
    #clf = KNeighborsClassifier(n_neighbors=1).fit(x_true_train, y_true_train) 
    
    y_valid_pred = clf.predict(x_valid_transormed)
    print(y_valid.shape)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred) 
    print('Validation accuracy:', valid_accuracy)
    scores.append(valid_accuracy)
















