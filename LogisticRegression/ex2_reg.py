#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/20 23:22
@Author  : fanyuexiang
@Site    : 对数据集2进行正则化Logistic Regression分类
@File    : ex2_reg.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import pandas as pd

from LogisticRegression.ml import plotData

print('begin read data2')
data = pd.read_csv('ex2data2', header=None)
X = data.iloc[:, :-1]
m,n = X.shape
y = data.iloc[:, 2]
print('begin Visualizing data and sigmoid function')
plotData(X, y)
print('begin map feature')
