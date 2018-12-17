#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/12/18 上午12:18
@Author  : fanyuexiang
@Site    : 进行多变量线性回归练习
@File    : ext2.py
@Software: PyCharm
@version: 1.0
@describe:
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linerRegression.featureNormalize import featureNormalize
from linerRegression.gradientDescentMulti import gradientDescentMulti

print("read data")
ex1data1 = pd.read_csv('ex1data2', header=None)
X_ex1data1 = ex1data1.iloc[:, 0:2]
y_ex1data1 = ex1data1.iloc[:, 2]
print('begin featureNormalize')
X_norm, mu, sigma = featureNormalize(X_ex1data1)
print('begin analysis data')
# 绘制三维散点图
ax = plt.figure(figsize=(15, 10)).add_subplot(111, projection = '3d')
ax.scatter(ex1data1.iloc[:, 0], ex1data1.iloc[:, 1], ex1data1.iloc[:, 2], c = 'r', marker = '^') #点为红色三角形
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
print('minimize costFunction by BGD')
alpha = 0.1
num_iters = 1000
X_norm = np.concatenate((np.ones((X_norm.shape[0],1)), X_norm), axis=1)
theta = np.zeros(X_norm.shape[1])
theta, J_history = gradientDescentMulti(X_norm, y_ex1data1, theta, alpha, num_iters)
