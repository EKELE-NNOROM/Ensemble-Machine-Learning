#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:08:51 2018

@author: ekele
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, RandomForestClassifier, BaggingClassifier
from util import BaggedTreeRegressor, BaggedTreeClassifier
import matplotlib.pyplot as plt

from rf_classification import get_data
X, Y = get_data()
Ntrain = int(0.8*len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

class NotAsRandomForest:
    def __init__(self,n_estimators):
        self.B = n_estimators
        
    def fit(self, X, Y, M=None):
        N, D = X.shape
        if M is None:
            M = int(np.sqrt(D))
            
        self.models = []
        self.features = []
        for b in range(self.B):
            tree = DecisionTreeClassifier()

            # sample features
            features = np.random.choice(D, size=M, replace=False)

            # sample training samples
            idx = np.random.choice(N, size=N, replace=True)
            Xb = X[idx]
            Yb = Y[idx]

            tree.fit(Xb[:, features], Yb)
            self.features.append(features)
            self.models.append(tree)
            
    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for features, tree in zip(self.features, self.models):
            P += tree.predict(X[:, features])
        return np.round(P / self.B)
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
T = 500
test_err_prf = np.empty(T)
test_err_rf = np.empty(T)
test_err_bag = np.empty(T)

for num_trees in range(T):
    if num_trees == 0:
        test_err_prf[num_trees] = None
        test_err_rf[num_trees] = None
        test_err_bag[num_trees] = None
    else:
        prf = NotAsRandomForest(n_estimators=num_trees)
        prf.fit(Xtrain,Ytrain)
        test_err_prf[num_trees] = prf.score(Xtest,Ytest)
        
        rf = RandomForestClassifier(n_estimators=num_trees)
        rf.fit(Xtrain,Ytrain)
        test_err_rf[num_trees] = rf.score(Xtest,Ytest)
        
        bag = BaggedTreeClassifier(n_estimators=num_trees)
        bag.fit(Xtrain,Ytrain)
        test_err_bag[num_trees] = bag.score(Xtest, Ytest)
        
    if num_trees % 10 == 0:
        print('num_trees: ', num_trees)
        
plt.plot(test_err_rf, label='rf')
plt.plot(test_err_prf, label='prf')
plt.plot(test_err_bag, label='bag')
plt.legend()
plt.show()