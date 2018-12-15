#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 17:29
# @Author  : fanyuexiang
# @Site    : 单变量梯度下降算法
# @File    : gradientDescent.py
# @Software: PyCharm
from linerRegression.computeCost import computeCost
import numpy as np

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    :param X:
    :param y:
    :param theta: 模型参数矩阵
    :param alpha: 学习速率
    :param num_iters: 最大迭代次数
    """
    J_history = []
    m = len(y)
    theta_n = len(theta)
    for i in range(num_iters):
        # 对代价函数J求偏导数
        deltaJ = X.T.dot(X.dot(theta)-y)/m
        theta = theta - alpha*deltaJ
        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
