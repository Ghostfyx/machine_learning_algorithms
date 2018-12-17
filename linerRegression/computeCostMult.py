#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 18:07
# @Author  : fanyuexiang
# @Site    : 求解多特征数据集损失函数
# @File    : computeCostMult.py
# @Software: PyCharm
import numpy as np
def computeCostMult(X, y, theta):
    """
       Compute cost for linear regression with multiple variables
         J = computeCost(X, y, theta) computes the cost of using theta as the
         parameter for linear regression to fit the data points in X and y
      """
    m = y.size
    J = 0
    h_theta = np.dot(X, theta)
    J = (h_theta - y).dot(h_theta-y) /(2*m)
    return J
