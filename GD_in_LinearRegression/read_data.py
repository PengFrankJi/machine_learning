#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:12:12 2021

@author: jipeng
"""

# In this file, I define a function to read data.

# In[ ]:

import pandas as pd

def read_data(fileLocation, fields):
    """

    Parameters
    ----------
    fileLocation : TYPE, string
        DESCRIPTION. Location of the file to be read
    fileds : TYPE, list
        DESCRIPTION. The needed columns of the file

    Returns
    -------
    frame : TYPE, pd.DataFrame
        DESCRIPTION. All needed data

    """
    
    frame = pd.read_table(
        fileLocation,
        
        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,

        # Generate column headers row from each column number
        header=0,          # use the first line as headers

        usecols=fields
    )

    # Return the entire frame
    return frame