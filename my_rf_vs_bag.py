#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:52:34 2018

@author: ekele
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, RandomForestClassifier, BaggingClassifier
from util import BaggedTreeRegressor, BaggedTreeClassifier

N = 15
D = 100
X = (np.random.random((N, D)) - 0.5)*10
Y = X.sum(axis=1)**2 + 0.5*np.random.randn(N)
Ntrain = 8*N//10
Xtrain = X[:Ntrain]
Ytrain = Y[:Ntrain]
Xtest = X[Ntrain:]
Ytest = Y[Ntrain:]

T = 300

test_err_rf = np.empty(T)
test_err_bag = np.empty(T)

for num_trees in range(T):
    if num_trees == 0:
        test_err_rf[num_trees] = None
        test_err_bag[num_trees] = None
    else:
        rf = RandomForestRegressor(n_estimators=num_trees)
        rf.fit(Xtrain,Ytrain)
        test_err_rf[num_trees] = rf.score(Xtest,Ytest)
        
        bg = BaggedTreeRegressor(n_estimators=num_trees)
        bg.fit(Xtrain,Ytrain)
        test_err_bag[num_trees] = bg.score(Xtest,Ytest)
        
        if num_trees % 10 == 0:
            print('num_trees: ', num_trees)
            
plt.plot(test_err_rf, label='rf')
plt.plot(test_err_bag, label='bag')
plt.legend()
plt.show()
    