#!/usr/bin/env python
# coding: utf-8

"""
Created on Mon Mar  1 16:12:12 2021

@author: jipeng
"""

# In this file, I define a function gradientDescent to compute parameters 
# through Gradient Descent method. This method can be used in linear 
# regression and logistic regression. 

# In[ ]:


import numpy as np

def GradientDescent(X, Y, theta, iteration = 5000, alpha = 0.01):
    """

    Parameters
    ----------
    X : TYPE, np.array
        DESCRIPTION. data of fetures. Size: n * c
    Y : TYPE, np.array 
        DESCRIPTION. data of labels. Size: n * 1
    theta : TYPE, np.array
        DESCRIPTION. data of initial parameters. Size: c * 1
    iteration : TYPE, int
        DESCRIPTION. number of iterations. The default is 5000.
    alpha : TYPE, float
        DESCRIPTION. learning rate. The default is 0.01.

    Returns
    -------
    theta : TYPE, np.array.
        DESCRIPTION. The parameters after iterations.
    costList: TYPE, list. 
        DESCRIPTION. The cost values of each iteration.
    
    """

    X_transpose = np.transpose(X)
    n = X.shape[0]
    costList = []
    
    for i in range(iteration):
        Z = X @ theta - Y # n * 1
        theta -= alpha * (X_transpose @ Z) / n
        costList.append( (np.transpose(Z) @ Z) / n / 2 )
    
    return theta, costList

