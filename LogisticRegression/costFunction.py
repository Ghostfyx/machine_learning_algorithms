#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/19 上午12:00
@Author  : fanyuexiang
@Site    : LR损失函数
@File    : costFunction.py
@Software: PyCharm
@version: 1.0
@describe:
'''
from numpy import log

from LogisticRegression.sigmoid import sigmoid


def costFunction(theta, X, y):
    m = y.size
    z = X.dot(theta)
    g = sigmoid(z)
    J = -1/m*(y.T.dot(log(g)) + (1-y).T.dot(log((1-g))))
    return float(J)