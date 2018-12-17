#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/18 上午12:13
@Author  : fanyuexiang
@Site    : 多变量梯度下降算法
@File    : gradientDescentMulti.py
@Software: PyCharm
@version: 1.0
@describe:
'''
from linerRegression.computeCostMult import computeCostMult
from matplotlib import pyplot as plt

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples
    for i in range(num_iters):
        theta = theta - alpha * (X.T.dot(X.dot(theta) - y)) / m
        J_history.append(computeCostMult(X, y, theta))
    plt.plot(range(len(J_history)), J_history, 'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    plt.show()
    return theta, J_history