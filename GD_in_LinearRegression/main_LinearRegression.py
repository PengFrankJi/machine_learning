#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:09:42 2021

@author: jipeng
"""
import os
print(os.getcwd())#获得当前目录
os.chdir("/Users/jipeng/Documents/Study/Study_myself/maching_learning/my_work")

from GradientDescent import GradientDescent
from read_data import read_data
from normalize_data import rescaleNormalize
from split_data import split_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data and normalize it.
dataDf = read_data("sat.csv", [1, 2, 4])
data = rescaleNormalize(dataDf)
data = data.values

# split data
X = np.ones((data.shape[0], data.shape[1]))
X[:, 1: 3] = data[:, :2]
Y = data[:, 2]
X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.66)

# final parameters:
ALPHA = 0.05
ITERATIONS = 200

# call GradientDescent function to train the model
theta = np.zeros(X.shape[1])
theta, costList = GradientDescent(X_train, Y_train, theta, ITERATIONS, ALPHA)

# visualsize the convergence curve
plt.plot(range(0,len(costList)), costList);
plt.xlabel('iteration')
plt.ylabel('cost')
plt.title('alpha = {}  theta = {}'.format(ALPHA, theta))
plt.show()

# test
Y_test_model = X_test @ theta

# evaluate the model
R_square = 1 - (np.transpose(Y_test_model - Y_test) @ \
                (Y_test_model - Y_test)) / \
    (np.transpose(np.mean(Y_test) - Y_test) @ (np.mean(Y_test) - Y_test))
print(R_square)