# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:14:37 2018

@author: Mohammadsuzan.Shaikh
"""

'''Outlier detection function'''
def mean_replacement(dat):
    print('This function calculates outlier which are 3 standard deviation from mean and replace them by mean of remaining data')
    for col in dat:
        if np.issubdtype(dat[col].dtype, np.number) & len(dat[col].value_counts())>5:
            dat[col]=np.where(abs(dat[col]-np.mean(dat[col]))>3*np.std(dat[col]),
               np.mean(dat[col][dat[col]-np.mean(dat[col])<3*np.std(dat[col])]),
                      dat[col])
    return dat

#---------------------------------------------------------------------------------------------------------------#

def mad_replacement(dat):
    print('This function calculates outlier which are more than 3 median absolute deviation and replace them by median of data')
    for col in dat:
        if np.issubdtype(dat[col].dtype, np.number) & len(dat[col].value_counts())>5:
            dat[col]=np.where(abs(dat[col]-np.median(dat[col]))/mad>3,np.median(dat[col]),dat[col])
    return dat

#---------------------------------------------------------------------------------------------------------------#

def iqr_replacement(dat):
    print('This function calculates outlier which are more than 1.5 IQR away from Q1 and Q3 and replace them by median of data')
    for col in dat:
        if np.issubdtype(dat[col].dtype, np.number) & len(dat[col].value_counts())>5:
            q1=np.percentile(dat[col],25)
            q3=np.percentile(dat[col],75)
            iqr=q3-q1
            dat[col]=np.where((dat[col]<q1-(1.5*iqr)) | (dat[col]>q3+(1.5*iqr)),np.median(dat[col]),dat[col])
    return dat