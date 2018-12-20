#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/20 23:47
@Author  : fanyuexiang
@Site    : 将训练集的特征向高维扩展，即为多项式，更好的拟合特征
@File    : mapFeature.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import numpy as np

def mapFeature(X, degree=6):
    """
    :param X: Array like
    :param degree: feature polynomial degree
    """
