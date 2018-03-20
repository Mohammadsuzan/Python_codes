# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 11:05:15 2017

@author: LK
"""

import pandas as pd
import numpy as np
from data_summary import data_summary as ds

def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))

import math
def miss_summary(data, col, dv):
    col_list = ['var', 'bin', 'bin_cuts','miss_flag', 'num_ids', 'bad_ids']
    miss_summary = pd.DataFrame(index = np.arange(1), columns = col_list)
    miss_summary['var'] = col
    miss_summary['bin'], miss_summary['bin_cuts'] = np.nan, np.nan
    miss_summary['miss_flag'] = 1
    miss_data = data[pd.isnull(data[col])]
    miss_summary['num_ids'] = len(miss_data.index)
    miss_summary['bad_ids'] = miss_data[dv].sum()
    return miss_summary

def woe(per_goods, per_bads):
            if per_bads > 0:
                if per_goods > 0:
                    return math.log(per_goods/per_bads)*100
                else:
                    return math.log(0.00005/per_bads)*100
            else:
                if per_goods > 0:
                    return math.log(per_goods/0.00005)*100
                else:
                    return 0

def woe_binning(X, y, num_bins):
    summary_X = ds.SummaryDesc(X.drop(y, axis = 1))
    obj_cols = summary_X['index'][summary_X['DataType'] == 'object']
    obj_cols_binning = pd.DataFrame()
    for col in obj_cols:
        missing_summary = miss_summary(X, col, y)
        missing_summary['dtype'] = 'object'
        summary = X[~pd.isnull(X[col])].groupby([col]).agg({col:np.size, y:np.sum})
        summary = summary.rename(columns = {col: 'num_ids', y: 'bad_ids'})
        summary = summary.reset_index()
        summary = summary.rename(columns = {col:'bin'})
        summary['var'], summary['bin_cuts'], summary['miss_flag'] = col, np.nan, 0
        summary['dtype'] = 'object'
        summary = summary[missing_summary.columns.tolist()]
        obj_cols_binning = obj_cols_binning.append([summary, missing_summary])
        tot_goods = len(X.index) - X[y].sum()
        tot_bads = X[y].sum()
        obj_cols_binning['per_goods'] = (obj_cols_binning['num_ids'] - obj_cols_binning['bad_ids'])/tot_goods
        obj_cols_binning['per_bads'] = obj_cols_binning['bad_ids']/tot_bads
        #print obj_cols_binning['per_goods']
        obj_cols_binning['woe'] = obj_cols_binning.apply(lambda row: woe(row['per_goods'], row['per_bads']), axis = 1)
        obj_cols_binning['iv_bin'] = (obj_cols_binning['per_goods'] - obj_cols_binning['per_bads'])*(obj_cols_binning['woe']/100)
        
    num_cols = summary_X['index'][summary_X['DataType'] != 'object']
    num_cols_binning = pd.DataFrame()
    for col in num_cols:
        print col
        missing_summary = miss_summary(X, col, y)
        missing_summary['dtype'] = 'numeric'
        X_nonmiss = X[~pd.isnull(X[col])]
        X_nonmiss['bin'] = pd.qcut(X_nonmiss[col] + jitter(X_nonmiss[col]) , num_bins, labels = False)
        summary = X_nonmiss.groupby(['bin']).agg({'bin': np.size, col: np.max, y:np.sum})
        summary = summary.rename(columns = {'bin': 'num_ids', y: 'bad_ids', col: 'bin_cuts'})
        summary = summary.reset_index()
        summary = summary.groupby(['bin_cuts']).agg({'num_ids': np.sum, 'bad_ids': np.sum}).reset_index()
        summary['var'], summary['miss_flag'] = col, 0
        summary['bin'] = summary.groupby(['var'])['bin_cuts'].rank(ascending = True) - 1
        summary['dtype'] = 'numeric'
        summary = summary[missing_summary.columns.tolist()]
        num_cols_binning = num_cols_binning.append([summary, missing_summary])
        tot_goods = len(X.index) - X[y].sum()
        tot_bads = X[y].sum()
        num_cols_binning['per_goods'] = (num_cols_binning['num_ids'] - num_cols_binning['bad_ids'])/tot_goods
        num_cols_binning['per_bads'] = num_cols_binning['bad_ids']/tot_bads
        num_cols_binning['woe'] = num_cols_binning.apply(lambda row: woe(row['per_goods'], row['per_bads']), axis = 1)
        num_cols_binning['iv_bin'] = (num_cols_binning['per_goods'] - num_cols_binning['per_bads'])*(num_cols_binning['woe']/100)
    
    combined_binning = obj_cols_binning.append(num_cols_binning)
    combined_binning['iv_final'] = combined_binning.groupby(['var'])['iv_bin'].transform('sum')
    combined_binning.sort_values(by = ['iv_final', 'bin_cuts'], ascending = [False, True], inplace = True)
    combined_binning = combined_binning[combined_binning['num_ids'] > 0]
    return combined_binning

def clubbing(bins, decile, forward_flag, i, dv):
    row = bins[bins['bin'] == i + forward_flag]
    club = row.append(decile)
    club = club.groupby(['var']).agg({'num_ids': np.sum, 'bad_ids': np.sum, 'bin_cuts': np.max, 'miss_flag': np.mean, 'per_bads': np.sum, 'per_goods': np.sum, 'bin': np.min}).reset_index()
    club['flag'] = 0
    club['woe'] = club.apply(lambda row: woe(row['per_goods'], row['per_bads']), axis = 1)
    club['iv_bin'] = (club['per_goods'] - club['per_bads'])*(club['woe']/100)
    bins.drop(bins[bins['bin'] == i+forward_flag].index, inplace = True)
    bins.drop(bins[bins['bin'] == i].index, inplace = True)
    club = club[bins.columns.tolist()]
    bins = bins.append(club)
    bins['bin'] = bins.groupby(['var'])['bin_cuts'].rank(ascending = True)
    bins = bins.reset_index(drop = True)
    return bins

def smoothen_1(woe_bins, asc_flag, dv):
    woe_bins['bin'] = woe_bins.groupby(['var'])['bin_cuts'].rank(ascending = True) - 1
    woe_bins.sort_values(by = 'bin', ascending = asc_flag, inplace = True)
    woe_bins['woe_prev'] = woe_bins['woe'].shift(1)
    woe_bins['flag'] = np.where(pd.isnull(woe_bins['woe_prev']), 0, 
                					np.where(woe_bins['woe_prev'] > woe_bins['woe'], 1, 0))
    while woe_bins['flag'].sum() > 0:
        del woe_bins['woe_prev']
        for i in range(woe_bins['bin'].size):
            if woe_bins['flag'][woe_bins['bin'] == i].any() == 1:
                woe_prev = woe_bins['woe'][woe_bins['bin'] == i-1].mean()
                woe_next = woe_bins['woe'][woe_bins['bin'] == i+1].mean()
                woe_curr = woe_bins['woe'][woe_bins['bin'] == i].mean()
                if ((abs(woe_curr - woe_next) > abs(woe_curr - woe_prev)) | (pd.isnull(woe_next))):
                    woe_bins = clubbing(woe_bins, woe_bins[woe_bins['bin'] == i], -1, i, dv)
                else:
                    woe_bins = clubbing(woe_bins, woe_bins[woe_bins['bin'] == i], 1, i, dv)
        woe_bins['bin'] = woe_bins.groupby(['var'])['bin_cuts'].rank(ascending = True) - 1
        woe_bins.sort_values(by = 'bin', ascending = asc_flag, inplace = True)
        woe_bins['woe_prev'] = woe_bins['woe'].shift(1)
        woe_bins['flag'] = np.where(pd.isnull(woe_bins['woe_prev']), 0, 
                    					np.where(woe_bins['woe_prev'] > woe_bins['woe'], 1, 0))
    return woe_bins.drop(['woe_prev'], axis = 1)

def smoothen_bins(woe_bins, trend, dv):
    smoothened_bins = pd.DataFrame()
    num_woe_bins, obj_woe_bins = woe_bins[woe_bins['dtype'] == 'numeric'], woe_bins[woe_bins['dtype'] == 'object']
    num_woe_bins = pd.merge(num_woe_bins, trend, on = 'var')
    for var in num_woe_bins['var'].unique():
        print var
        var_bins = num_woe_bins[num_woe_bins['var'] == var]
        var_bins_nmiss, var_bins_miss = var_bins[~(pd.isnull(var_bins['bin']))], var_bins[(pd.isnull(var_bins['bin']))]
        var_bins_nmiss.drop(['trend', 'iv_final', 'dtype'], axis = 1, inplace = True), var_bins_miss.drop(['trend', 'iv_final', 'dtype'], axis = 1, inplace = True)
        if var_bins['trend'].unique().any() == 'V':
            break_bin = var_bins_nmiss['bin'][var_bins_nmiss['woe'] == var_bins_nmiss['woe'].min()].iloc[0]
            var_bins_nmiss_p1, var_bins_nmiss_p2 = var_bins_nmiss[var_bins_nmiss['bin'] <= break_bin], var_bins_nmiss[var_bins_nmiss['bin'] > break_bin]
            var_bins_nmiss_p1 = smoothen_1(var_bins_nmiss_p1, False, dv)
            var_bins_nmiss_p2 = smoothen_1(var_bins_nmiss_p2, True, dv)
            smoothened_bins = smoothened_bins.append([var_bins_nmiss_p1, var_bins_nmiss_p2, var_bins_miss])
        elif var_bins['trend'].unique().any() == 'inward-V':
            break_bin = var_bins_nmiss['bin'][var_bins_nmiss['woe'] == var_bins_nmiss['woe'].max()].iloc[0]
            var_bins_nmiss_p1, var_bins_nmiss_p2 = var_bins_nmiss[var_bins_nmiss['bin'] <= break_bin], var_bins_nmiss[var_bins_nmiss['bin'] > break_bin]
            var_bins_nmiss_p1 = smoothen_1(var_bins_nmiss_p1, True, dv)
            var_bins_nmiss_p2 = smoothen_1(var_bins_nmiss_p2, False, dv)
            var_bins = var_bins_nmiss_p1.append(var_bins_nmiss_p2)
            smoothened_bins = smoothened_bins.append([var_bins_nmiss_p1, var_bins_nmiss_p2, var_bins_miss])
        else:
            asc_flag = var_bins[['trend']].drop_duplicates()['trend'].values[0]
            var_bins_nmiss = smoothen_1(var_bins_nmiss, asc_flag, dv)
            smoothened_bins = smoothened_bins.append([var_bins_nmiss, var_bins_miss])
    smoothened_bins['iv_final'] = smoothened_bins.groupby(['var'])['iv_bin'].transform('sum')
    smoothened_bins['dtype'] = 'numeric'
    smoothened_bins = smoothened_bins[obj_woe_bins.columns.tolist()]
    smoothened_bins = smoothened_bins.append(obj_woe_bins)
    smoothened_bins.sort_values(by = ['iv_final', 'bin_cuts'], ascending = [False, True], inplace = True)
    return smoothened_bins

def woe_transformation(X, woe_bins):
    vars_miss_flag = woe_bins.groupby(['var']).agg({'miss_flag':np.sum}).reset_index()
    obj_cols = woe_bins[['var']][woe_bins['dtype'] == 'object'].drop_duplicates()
    obj_cols = pd.merge(obj_cols, vars_miss_flag, on = 'var')
    woe_data = pd.DataFrame(index = X.index)
    for col in obj_cols['var']:
        print col
        if obj_cols['miss_flag'][(obj_cols['var'] == col)].any() == 1:
            miss_woe = woe_bins['woe'][(woe_bins['miss_flag'] == 1) & (woe_bins['var'] == col)].iloc[0]
            var_miss_data = X[col][pd.isnull(X[col])]
            var_miss_data.replace(np.nan, miss_woe, inplace = True)
        non_miss_woe = woe_bins[['woe', 'bin']][(woe_bins['miss_flag'] == 0) & (woe_bins['var'] == col)]
        var_nmiss_data = X[col][~pd.isnull(X[col])]
        non_miss_woe = non_miss_woe.set_index('bin')
        non_miss_woe = non_miss_woe.T.squeeze()
        var_nmiss_data = var_nmiss_data.map(non_miss_woe)
        if obj_cols['miss_flag'][(obj_cols['var'] == col)].any() == 1:
            var_woe = var_miss_data.append(var_nmiss_data)
        else:
            var_woe = var_nmiss_data.copy()
        var_woe.sort_index(inplace = True)
        woe_data = woe_data.join(var_woe)
    
    num_cols = woe_bins[['var']][woe_bins['dtype'] != 'object'].drop_duplicates()
    num_cols = pd.merge(num_cols, vars_miss_flag, on = 'var')
    for col in num_cols['var']:
        print col
        if num_cols['miss_flag'][(num_cols['var'] == col)].any() == 1:
            miss_woe = woe_bins['woe'][(woe_bins['miss_flag'] == 1) & (woe_bins['var'] == col)].iloc[0]
            var_miss_data = X[col][pd.isnull(X[col])]
            var_miss_data.replace(np.nan, miss_woe, inplace = True)
        non_miss_woe = woe_bins[['woe', 'bin']][(woe_bins['miss_flag'] == 0) & (woe_bins['var'] == col)]
        var_nmiss_data = X[col][~pd.isnull(X[col])]
        non_miss_woe = non_miss_woe[['woe']].reset_index(drop = True)
        non_miss_woe['bin'] = non_miss_woe.index 
        non_miss_woe = non_miss_woe.set_index('bin')
        non_miss_woe = non_miss_woe.iloc[:,0]
        bin_cuts = woe_bins['bin_cuts'][woe_bins['var'] == col].dropna()
        if float(bin_cuts.min()) < float(var_nmiss_data.min()):
            bin_cuts = np.append((bin_cuts.min() -1), bin_cuts)
        else:
            bin_cuts = np.append(var_nmiss_data.min() - 1, bin_cuts)
        if bin_cuts.max() < var_nmiss_data.max():
            bin_cuts[len(bin_cuts)-1] = var_nmiss_data.max()
        bins = pd.cut(var_nmiss_data, bin_cuts, labels = non_miss_woe.index)
        var_nmiss_data = bins.map(non_miss_woe)
        if num_cols['miss_flag'][(num_cols['var'] == col)].any() == 1:
            var_woe = var_miss_data.append(var_nmiss_data)
        else:
            var_woe = var_nmiss_data.copy()
        var_woe.sort_index(inplace = True)
        woe_data = woe_data.join(var_woe)
    return woe_data