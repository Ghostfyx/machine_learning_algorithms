#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/15 16:41
# @Author  : fanyuexiang
# @Site    : 进行数据展示
# @File    : plotData.py
# @Software: PyCharm
from matplotlib import pyplot as plt
import pandas as pd

from linerRegression import computeCost


def plotData(X, y):
    """
       绘制2D线图
       plots the data points and gives the figure axes labels of
       population and profit.
    """
    plt.scatter(X, y)
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()