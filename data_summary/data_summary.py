# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:20:01 2017

@author: LK
"""

import pandas as pd
import numpy as np

import functools as ft
def SummaryDesc(Data):
    Summary = Data.describe(include = 'all')
    Summary = Summary.transpose().reset_index()
    Summary.columns.values[0] = 'index'
    Summary['MissingPercent'] = Summary['count'].apply(lambda x: 1 - (float(x)/len(Data.index)))
    Summary = Summary.sort_values(by = (['MissingPercent']), ascending = False)
    Columns = Data.columns
    Unique = pd.Series()
    for i in Columns:
        Unique[i] = Data[i].nunique()
    Unique = Unique.to_frame()
    Unique.columns = ['NUnique']
    Unique.reset_index(inplace = True)
    Unique.columns.values[0] = 'index'
    Dtypes = Data.dtypes
    Dtypes = Dtypes.to_frame()
    Dtypes.columns = ['DataType']
    Dtypes.reset_index(inplace = True)
    Dtypes.columns.values[0] = 'index'
    Dfs = [Summary, Unique, Dtypes]
    Summary = ft.reduce(lambda left,right: pd.merge(left,right,on='index'), Dfs)
    return Summary[['index', 'MissingPercent','DataType', 'mean', 'min', 'max']]
