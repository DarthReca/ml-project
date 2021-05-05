# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:49:20 2021

@author: Hossein.JvdZ
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# def NBG(data: pd.DataFrame):
#     dataset = data['Class']

dataset = pd.read_csv('Test.csv')
    
    # last column not included, we are putting X axis
X= dataset.iloc[:,:-1]
    
    # last column is y 
y= dataset.iloc[:,8] 
    
    # split dataset
X_train, X_test,y_train,y_test=train_test_split(X, y, test_size=0.40, random_state=0)
    
    #use Naive Bayes Gaussian and show the computation and then store that into new variable 
NBModel=GaussianNB()
    
    #model to training dataset
NBModel.fit(X_train, y_train)
    
    #prediction function
y_predicted=NBModel.predict(X_test)

accuracy_score(y_test, y_predicted)*100
print(metrics.classification_report(y_test, y_predicted))
print(metrics.confusion_matrix(y_test, y_predicted))


np.array([])
