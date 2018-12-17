#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 16:35
# @Author  : fanyuexiang
# @Site    : 单变量线性回归
# @File    : ex1.py
# @Software: PyCharm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from linerRegression.featureNormalize import featureNormalize
from linerRegression.gradientDescent import gradientDescent
from linerRegression.plotData import plotData

print("read data")
ex1data1 = pd.read_csv('ex1data1', header=None)
X_ex1data1 = ex1data1.iloc[:, 0]
y_ex1data1 = ex1data1.iloc[:, 1]
print("begin feature normalize")
X_normal,mu,sigma = featureNormalize(X_ex1data1)
print("begin data plot")
plotData(X_ex1data1, y_ex1data1)
print("begin BGD algorithms")
theta = np.zeros((2,))
a = np.zeros((X_ex1data1.shape[0], 1))
# np.vstack在列上进行拼接
X_train = np.vstack((np.ones(X_ex1data1.shape[0]), X_normal)).T
alpha = 0.1
max_iters = 1000
theta, J_history = gradientDescent(X=X_train, y=y_ex1data1, theta=theta, alpha=alpha, num_iters=max_iters)
print('beigin Visualizing cost function J')
plt.figure(figsize=(10, 6))
plt.plot(range(len(J_history)),J_history,'bo')
plt.grid(True)
plt.title("Convergence of Cost Function")
plt.xlabel("Iteration number")
plt.ylabel("Cost function")
print('Theta found by gradient descent: ', theta)
print('begin plot linerModel')
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 1], y_ex1data1, label='Training data')
plt.plot(X_train[:, 1], X_train.dot(theta), color='r', linestyle='-', label='Linear regression')
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')
plt.legend(loc='upper right')
plt.show()
print('predict data')
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = np.array([1, 7.0]).dot(theta)
print('For population = 70,000, we predict a profit of ', predict2*10000)
