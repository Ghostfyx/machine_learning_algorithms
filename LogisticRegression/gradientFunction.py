#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/19 下午11:44
@Author  : fanyuexiang
@Site    : 
@File    : gradientFunction.py
@Software: PyCharm
@version: 1.0
@describe:
'''
from LogisticRegression.sigmoid import sigmoid


def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization
    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    grad = 1/m*(X.T.dot(sigmoid(X.dot(theta)) - y))
    return grad