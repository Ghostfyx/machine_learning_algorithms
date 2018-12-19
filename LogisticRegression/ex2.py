#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/18 下午11:12
@Author  : fanyuexiang
@Site    : 非正则化Logistic Regression
@File    : ex2.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import pandas as pd
import numpy as np
from LogisticRegression.costFunction import costFunction
from LogisticRegression.gradientFunction import gradientFunction
from LogisticRegression.ml import plotData, plotSigmoid, plotDecisionBoundary
import scipy.optimize as op

print('begin read data')
data = pd.read_csv('ex2data1', header=None)
X = data.iloc[:, :-1]
m,n = X.shape
y = data.iloc[:, 2]
print('Visualizing data and sigmoid function')
plotData(X, y)
plotSigmoid()
print('compute costfunction')
theta = np.zeros(n+1)
X = np.concatenate((np.ones((m, 1)), X), axis=1)
print(costFunction(theta, X, y))
print('begin gradientFunction')
print(gradientFunction(theta, X, y))
print('begin use fminunc function')
# matlab的fminunc与scipy的
min_cost_result = op.minimize(costFunction, theta, (X,y), 'BFGS', jac=gradientFunction)
theta = min_cost_result.x
print(theta)
print('begin Evaluating Logistic Regression')
plotDecisionBoundary(theta, X, y)