# Copyright (C) 2021 #
# @Time    : 2023/6/28 15:48
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true, task):

    pred_mean = np.mean(pred, axis=0)
    true_mean = np.mean(true, axis=0)

    pred_dev = pred - pred_mean
    true_dev = true - true_mean

    numerator = np.sum(pred_dev * true_dev, axis=0)
    denominator = np.sqrt(np.sum(pred_dev**2, axis=0) * np.sum(true_dev**2, axis=0))
    return np.mean(numerator / (denominator + 1e-6))
        

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    # return mean_squared_error(pred,true)
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred : np.array, true : np.array, true_mask : np.array = None):

    zero_mask = ~np.isclose(true, 0, atol=1e-5)

    mape = np.abs((pred - true) / (true))

    mape *= zero_mask

    mape = np.nan_to_num(mape)
    

    return np.mean(mape)

def MSPE(pred, true):

    zero_mask = ~np.isclose(true, 0, atol=1e-5)

    mspe = np.square((pred - true) / (true))

    mspe *= zero_mask

    return np.mean(mspe)

def WAPE(pred, true):
    


    wape = np.sum(np.abs(pred - true), axis=1) / (np.sum(np.abs(true), axis=1) + 5e-5)

    return np.mean(wape)

def R2(pred, true):
    
    mean = np.mean(true, axis=0)
    sse = np.sum(np.pow(pred - true, 2), axis=0)
    sst = np.sum(np.pow(true - mean, 2), axis=0)

    r2 = 1 - sse/(sst + 1e-6)

    return np.mean(r2)

def metric(pred, true, task):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true, task)
    wape = WAPE(pred, true)
    
    if task == 'soft_sensor':
        r2 = R2(pred, true)
        return mae,mse,rmse,mape,mspe,r2,corr
    elif task == 'short_term_forecasting':
        return mae,mse,rmse,mape,mspe,wape,corr
    
