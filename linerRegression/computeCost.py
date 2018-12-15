#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 17:18
# @Author  : fanyuexiang
# @Site    : 计算线性回归的损失函数
# @File    : computeCost.py
# @Software: PyCharm
import numpy as np

def computeCost(X, y, theta):
    """
    :param X:
    :param y:
    :param theta: 参数矩阵
    """
    m = len(y)
    h_theta = np.dot(X, theta)
    J_theta = (h_theta-y).dot(h_theta-y)/(2*m)
    return J_theta
