#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/18 下午11:40
@Author  : fanyuexiang
@Site    : sigmoid函数
@File    : sigmoid.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import numpy as np
def sigmoid(z):
    """
    computes the sigmoid of z.
    z : ndarray
        The ndarray to apply expit to element-wise.
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).

    # =============================================================
    g = 1/(1 + np.exp(-z))
    return g