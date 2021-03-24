#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:42:43 2021

@author: jipeng
"""

import pandas as pd

def rescaleNormalize(dataDf):
    """
    
    Parameters
    ----------
    dataDf : TYPE, pd.DataFrame
        DESCRIPTION. All data

    Returns
    -------
    data : TYPE, pd.DataFrame
        DESCRIPTION. all data after normalization

    """
    
    min_vals = dataDf.apply(lambda x: x.min(), axis = 0)
    max_vals = dataDf.apply(lambda x: x.max(), axis = 0)
    domin = max_vals - min_vals
    data = dataDf.apply(lambda x: (x - min_vals) / domin, axis = 1)
    return data