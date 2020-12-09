# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:32:13 2020

@author: abhir
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


data  =  pd.read_csv('winequality-red.csv', sep=';')
#data.columns = data.columns.str.replace(' ','_')
X = data.drop(['quality'], axis = 1)
Y = data['quality']

data.corr() 
#f, axes = plt.subplots(figsize = (10,10))
#sns.heatmap(data.corr(), annot = True, linewidths=.8, fmt = ".3f", ax=axes, cmap="YlGnBu")
#plt.show()

#fig, axes = plt.subplots(11,11, figsize=(50,50))
#for i in range(11):
#    for j in range(11):
#        axes[i, j].scatter(data.iloc[:,i], data.iloc[:,j], c = data.quality, cmap="YlGnBu")
#        axes[i,j].set_xlabel(data.columns[i])
#        axes[i,j].set_ylabel(data.columns[j])
#        axes[i,j].legend(data.quality)
#plt.show()


#fig, ax1 = plt.subplots(4,3, figsize=(22,16))
#k = 0
#for i in range(4):
#    for j in range(3):
#        if k != 11:
#            sns.barplot('quality',data.iloc[:,k], data=data, ax = ax1[i][j])
#            k += 1
#plt.show()
sns.barplot('quality',data.iloc[:,10], data=data)
plt.xlabel("Wine Quality rating")
plt.ylabel("Alcohol")
plt.title("Distribution of Red Wine Quality Ratings")

#data.iloc[:, 1].min();