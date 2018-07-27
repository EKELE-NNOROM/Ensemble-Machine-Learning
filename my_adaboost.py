#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:32:16 2018

@author: ekele
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from rf_classification import get_data

class AdaBoost:
    def __init__(self, M):
        self.M = M
    
    def fit(self, X, Y):
        self.models = []
        self.alphas = []
        
        N, _ = X.shape
        W = np.ones(N) / N
        
        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=1)
            tree.fit(X, Y, sample_weight=W)
            P = tree.predict(X)
            
            err = W.dot(P != Y)
            
            alpha = 0.5 * (np.log(1 - err) - np.log(err))
            
            W = W*np.exp(-alpha*Y*P)
            
            W = W/W.sum()
            
            self.models.append(tree)
            self.alphas.append(alpha)
            
    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha*tree.predict(X)
        return np.sign(FX), FX
    
    def score(self,X,Y):
        P, FX = self.predict(X)
        L = np.exp(-Y*FX).mean()
        return np.mean(P == Y), L
    
if __name__ == '__main__':
    X, Y = get_data()
    Y[Y == 0] = -1
    Ntrain = int(0.8*(len(X)))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    T = 200
    train_errors = np.empty(T)
    test_errors = np.empty(T)
    test_losses = np.empty(T)
    
    for num_trees in range(T):
        if num_trees == 0:
            train_errors[num_trees] = None
            test_errors[num_trees] = None
            test_losses[num_trees] = None
        if num_trees % 20 == 0:
            print(num_trees)
            
        model = AdaBoost(num_trees)
        model.fit(Xtrain,Ytrain)
        acc, loss = model.score(Xtest, Ytest)
        acc_train, _ = model.score(Xtrain, Ytrain)
        test_errors[num_trees] = 1 - acc
        train_errors[num_trees] = 1 - acc_train
        test_losses[num_trees] = loss
        
        if num_trees == T - 1:
            print('final train error', 1 - acc_train)
            print('final test error', 1 - acc)
            
        plt.plot(test_errors, label = 'Test errors', color = 'g')
        plt.plot(train_errors, label = 'Train errors', color = 'r')
        plt.legend()
        plt.show()
        
        plt.plot(test_errors, label = 'Test errors')
        plt.plot(test_losses, label = 'Test losses')
        plt.legend()
        plt.show()
            