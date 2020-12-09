# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:29:07 2020

@author: abhir
"""

# Importing the Rotton Tomatoes Movie review dataset
from datasets import load_dataset

dataset = load_dataset('rotten_tomatoes')

print('--------------------------------------------')
print('Dataset consists of:')
print('--------------------------------------------')
print('Training data (n=' , len(dataset['train'])      , '),',
      'Validataion data (n=' , len(dataset['validation']) , '), and', 
      'Teting data (n=' , len(dataset['test']) , ')')

# Extracting the first training set point
print('\n--------------------------------------------')
print('Example sentence from Training set:')
print('--------------------------------------------')
print(dataset['train']['text'][0])

print('\n--------------------------------------------')
print('Example Label from training set')
print('--------------------------------------------')
print(dataset['test']['label'][0])