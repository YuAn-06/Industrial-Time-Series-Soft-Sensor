# Copyright (C) 2021 #
# @Time    : 2023/6/28 15:48
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
import torch

def RSE(pred: np.array, true: np.array) -> float:
    """
    Root Squared Error
    pred (np.array): predicted values
    true (np.array): true values
    """
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred: np.ndarray, true: np.ndarray, task: str) -> float:
    """
    Pearson Correlation Coefficient Metrics
    pred (np.ndarray): predicted values
    true (np.ndarray): true values
    task (str): task type
    """
    pred_mean = np.mean(pred, axis=0)
    true_mean = np.mean(true, axis=0)

    pred_dev = pred - pred_mean
    true_dev = true - true_mean

    numerator = np.sum(pred_dev * true_dev, axis=0)
    denominator = np.sqrt(np.sum(pred_dev**2, axis=0) * np.sum(true_dev**2, axis=0))
    return np.mean(numerator / (denominator + 1e-6))
        

def MAE(pred: np.array, true: np.array) -> float:
    """
    Mean Absolute Error
    pred (np.array): predicted values
    true (np.array): true values
    """
    return np.mean(np.abs(pred-true))

def MSE(pred: np.array, true: np.array) -> float:
    """
    Mean Squared Error
    pred (np.array): predicted values
    true (np.array): true values
    """
    return np.mean((pred-true)**2)

def RMSE(pred: np.array, true: np.array) -> float:
    """
    Root Mean Squared Error
    pred (np.array): predicted values
    true (np.array): true values
    """
    return np.sqrt(MSE(pred, true))

def MAPE(pred : np.array, true : np.array, true_mask : np.array = None) -> float:
    """
    Mean Absolute Percentage Error
    pred (np.array): predicted values
    true (np.array): true values
    true_mask (np.array): true mask
    """

    zero_mask = ~np.isclose(true, 0, atol=1e-5)

    mape = np.abs((pred - true) / (true))

    mape *= zero_mask

    mape = np.nan_to_num(mape)
    

    return np.mean(mape)

def MSPE(pred: np.array, true: np.array) -> float:
    """
    Mean Squared Percentage Error
    pred (np.array): predicted values
    true (np.array): true values
    """

    zero_mask = ~np.isclose(true, 0, atol=1e-5)

    mspe = np.square((pred - true) / (true))

    mspe *= zero_mask

    return np.mean(mspe)

def WAPE(pred: np.array, true: np.array) -> float:
    """
    Weighted Absolute Percentage Error
    pred (np.array): predicted values
    true (np.array): true values
    """

    wape = np.sum(np.abs(pred - true), axis=1) / (np.sum(np.abs(true), axis=1) + 5e-5)

    return np.mean(wape)

def R2(pred: np.array, true: np.array) -> float:
    """
    R-squared Metrics
    pred (np.array): predicted values
    true (np.array): true values
    """
    
    mean = np.mean(true, axis=0)
    sse = np.sum(np.pow(pred - true, 2), axis=0)
    sst = np.sum(np.pow(true - mean, 2), axis=0)

    r2 = 1 - sse/(sst + 1e-6)

    return np.mean(r2)

def metric(pred: np.array, true: np.array, task: str) -> tuple:
    """
    Calculate the metrics for the given predicted and true values. 
    The metrics include MAE, MSE, RMSE, MAPE, MSPE, R2, and correlation coefficient. 
    The function returns a tuple containing the calculated metrics.
    pred (np.array): predicted values
    true (np.array): true values
    task (str): task type
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true, task)
    wape = WAPE(pred, true)
    
    if task == 'soft_sensor':
        r2 = R2(pred, true)
        return mae, mse, rmse, mape, mspe, wape, corr
    elif task == 'short_term_forecasting':
        return mae, mse, rmse, mape, mspe, wape, corr
    
