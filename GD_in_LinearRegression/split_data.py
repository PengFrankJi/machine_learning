#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:28:48 2021

@author: jipeng
"""

import numpy as np

def split_data(X, Y, ratio_train = 0.66):
    """

    Parameters
    ----------
    X : TYPE, np.array
        DESCRIPTION. all feature data
    Y : TYPE, np.array
        DESCRIPTION. all label data
    ratio_train : TYPE, float
        DESCRIPTION. The ratio of train to all data. in the range of (0, 1)

    Returns
    -------
    X_train : TYPE, np.array
        DESCRIPTION. all feature data used to train model
    X_test : TYPE, np.array
        DESCRIPTION. all feature data used to test model
    Y_train : TYPE, np.array
        DESCRIPTION. all label data used to train model
    Y_test : TYPE, np.array
        DESCRIPTION. all label data used to test model

    """
    n = X.shape[0]
    index = np.random.permutation(n)
    n_train = int(n * ratio_train)
    index_train = index[: n_train]
    index_test = index[n_train: ]
    
    X_train = X[index_train]
    X_test = X[index_test]
    Y_train = Y[index_train]
    Y_test = Y[index_test]
    
    return X_train, X_test, Y_train, Y_test
    