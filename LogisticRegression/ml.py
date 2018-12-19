#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/18 下午11:16
@Author  : fanyuexiang
@Site    : 数据可视化
@File    : ml.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import numpy as np
from matplotlib import pyplot as plt
from pandas import Series
from mpl_toolkits.mplot3d import axes3d
from scipy.special import expit
from LogisticRegression.sigmoid import sigmoid


def plotData(X,y):
    pos = X[np.where(y==1,True,False)]
    neg = X[np.where(y==0,True,False)]
    p1 = plt.scatter(pos.iloc[:, 0], pos.iloc[:, 1], marker='+', s=30, color='b')
    p2 = plt.scatter(neg.iloc[:, 0], neg.iloc[:, 1], marker='o', s=30, color='y')
    # 显示图例
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

def plotSigmoid():
    myx = np.arange(-10, 10, .1)
    plt.plot(myx, sigmoid(myx))
    plt.title("this is sigmoid function")
    # 显示网格
    plt.grid(True)
    plt.show()


def plotDecisionBoundary(theta, X, y):
    """
    绘制二项分类Logistic决策边界
    Plotting the decision boundary: two points, draw a line between
    Decision boundary occurs when h = 0, or when
    theta0 + theta1*x1 + theta2*x2 = 0
    y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)
    """
    pos = X[np.where(y == 1, True, False)]
    neg = X[np.where(y == 0, True, False)]
    p1 = plt.scatter(pos[:, 1], pos[:, 2], marker='+', s=30, color='b', label ='Admitted')
    p2 = plt.scatter(neg[:, 1], neg[:, 2], marker='o', s=30, color='y', label='Not admitted')
    boundary_xs = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    boundary_ys = (-1/theta[2])*(theta[0] + theta[1]*boundary_xs)
    p3 = plt.plot(boundary_xs, boundary_ys, color='r', label='Decision Boundary')
    # 显示图例
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()