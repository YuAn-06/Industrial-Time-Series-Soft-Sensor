# Copyright (C) 2021 #
# @Time    : 2023/6/28 15:48
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

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

def R2(pred, true):
    
    return r2_score(pred, true)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae,mse,rmse,mape,mspe
