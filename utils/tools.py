"""
Copyright (C) 2023
@ Name: tools.py
@ Time: 2023/10/31 22:48
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

import math
import torch
import numpy as np
import os
import random
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def mean_interpolate_zeros(arr):
    """
    对数组中的 0 值进行均值插值
    :param arr: 输入数组，形状为 [C, H, W]
    :return: 插值后的数组
    """
    # 生成 Mask，标记 0 值的位置
    mask = (arr == 0)
    
    # 如果数组中没有 0 值，直接返回
    if not np.any(mask):
        return arr
    
    # 创建一个副本用于存储插值结果
    interpolated_arr = arr.copy()
    
    # 获取数组的形状
    c, h, w = arr.shape
    
    # 遍历所有 0 值的位置
    zero_indices = np.argwhere(mask)
    for idx in zero_indices:
        c_idx, h_idx, w_idx = idx
        
        # 获取周围非零值的均值
        neighbors = []
        for dh in [-1, 0, 1]:
            for dw in [-1, 0, 1]:
                if (dh == 0 and dw == 0) or (h_idx + dh < 0) or (w_idx + dw < 0) or (h_idx + dh >= h) or (w_idx + dw >= w):
                    continue  # 跳过自身和越界的位置
                if not mask[c_idx, h_idx + dh, w_idx + dw]:  # 如果邻居不是 0 值
                    neighbors.append(arr[c_idx, h_idx + dh, w_idx + dw])
        if neighbors:  # 如果有非零邻居
            interpolated_arr[c_idx, h_idx, w_idx] = np.mean(neighbors)
    
    return interpolated_arr


def roll_data(df_raw, cols: list, roll=0):
    if cols is None or roll == 0:
        return df_raw, None
    else:
        data = df_raw.copy()
        roll_cols = []
        for c in cols:
            for i in roll:
                data[f'{c}_{i}'] = df_raw[c].shift(i)
                roll_cols.append(f'{c}_{i}')
        return data[roll_cols], roll_cols

def visual(preds, trues):
    """

    :param preds: predition values
    :param trues: trues value
    :param if_re: if reconstruction task?
    :param name: model name
    :return:
    """

    assert trues.shape == preds.shape, f"The 'true' shape: {trues.shape} is not identical with 'preds': {preds.shape} "

    shape = preds.shape
    if len(shape) == 3:
        # chose the first sample of time sereis
        preds = preds[:,0,:]
        trues = trues[:,0,:]
    num_features = preds.shape[-1]
    num_rows = (num_features + 2) // 3  # 每行3个子图

    # 创建图形
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 12))
    axes = axes.ravel()  # 将二维数组展平为一维

    # 绘制每个特征
    for i in range(num_features):
        axes[i].plot(preds[:,i],label='preds')
        axes[i].plot(trues[:,i],label='trues')
        axes[i].set_title(f'Feature {i+1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
        axes[i].legend()
    # 隐藏多余的子图
    for j in range(num_features, num_rows*3):
        axes[j].axis('off')
   
    plt.tight_layout()
    plt.show()


def visual_feautures(features,model_name,data_name,decomposition='PCA'):
    if features.shape[-1] != 2:
        if decomposition =='PCA':
            decom = PCA(n_components=2)
        else:
            decom = TSNE(n_components=2)
        features = decom.fit_transform(features)

    plt.figure()
    plt.title(label='{}_{}'.format(model_name,data_name))
    plt.scatter(features[:, 0], features[:, 1])





class EarlyStopping:
    def __init__(self,patience = 8, verbose = False, delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta: # val loss 不小于best score
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
         
        else:   # val_loss 下降
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
           
        torch.save(model.state_dict(),path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            20: 5e-5, 40: 1e-5, 60: 5e-6, 80: 1e-6,
            100: 5e-7, 150: 1e-7, 200: 5e-8
        }
        # lr_adjust = {100: 5e-4, 200: 1e-4, 300: 5e-5, 500: 1e-5, 700: 5e-6}

    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.epoch* math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def plot_ss(trues, preds):
    assert trues.shape == preds.shape, f"The 'true' shape: {trues.shape} is not identical with 'preds': {preds.shape} "
    if trues.ndim > 2:
        bs, sl, c = trues.shape
        plt.figure(figsize=(20,10))
        for i in range(c):
            plt.subplot(c//2 ,2 +1 ,i+1)
            plt.plot(trues[:,0,i],label='trues')
            plt.plot(preds[:, 0, i],label='preds')
    else:
        bs, c = trues.shape
        plt.figure(figsize=(20, 10))
        for i in range(c):
            plt.subplot(c // 2, 2 + 1, i + 1)
            plt.plot(trues[:,i], label='trues')
            plt.plot(preds[:,i], label='preds')
    plt.legend()
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def select_tensorboard_hparams(args):
    """
    选择用于TensorBoard超参数记录的参数
    
    Args:
        args: 包含模型配置的参数对象
        
    Returns:
        dict: 包含选定超参数的字典
    """
    hparams = {
        # 学习率
        'lr': getattr(args, 'learning_rate', None),
        # 批次大小
        'batch_size': getattr(args, 'batch_size', None),
        # 模型名称
        'model': getattr(args, 'model', None),
        # 训练轮数
        'epochs': getattr(args, 'epoch', None),
        # 序列长度
        'seq_len': getattr(args, 'seq_len', None),
        # 预测长度
        'pred_len': getattr(args, 'pred_len', None),
        # 标签长度
        'label_len': getattr(args, 'label_len', None),
        # 模型维度
        'd_model': getattr(args, 'd_model', None),
        # 注意力头数
        'n_heads': getattr(args, 'n_heads', None),
        # 编码器层数
        'e_layers': getattr(args, 'e_layers', None),
        # 解码器层数
        'd_layers': getattr(args, 'd_layers', None),
        # 前馈网络维度
        'd_ff': getattr(args, 'd_ff', None),
        # Dropout率
        'dropout': getattr(args, 'dropout', None),
        # 成分数量（用于多模态模型）
        'n_components': getattr(args, 'n_components', None),
    }
    
    # 移除值为None的项
    hparams = {k: v for k, v in hparams.items() if v is not None}
    
    return hparams


def DC_preprocess(df_raw, target):
    
    """
    Debutainer columns (DC), Enhance the sequence data with rolling window data
    """
    
    data_df = df_raw.copy()
    target_df = df_raw[target]

    
    data_df = data_df.drop(columns=['x_6','x_7','y_1'])
    
    
    
    x5_roll, x5_roll_cols = roll_data(df_raw, ['x_5'], roll=[1,2,3])

    
    y1_roll, y1_roll_cols = roll_data(df_raw, [target], roll=[1,2,3,4])
    
    x_78_mean = df_raw[['x_6','x_7']].mean(axis=1).rename('x_78_mean')
    
    cols = data_df.columns.tolist() + [x_78_mean.name] + x5_roll_cols + y1_roll_cols
    
    data_df = pd.concat([data_df, x_78_mean, x5_roll, y1_roll, target_df], axis=1)

    data_df = data_df.iloc[4:, :]
    
    
    print('DC Preprocess Done')
    print('Enhance Sequence columns', data_df.columns)
    
    return data_df, cols
    
def SRU_preprocess(df_raw, target):  
    """
    For SRU, Enhance the sequence data with rolling window data
    """
    data_df = df_raw.copy()
    target_df = df_raw[target]

    

    data_df = data_df.drop(columns=['SO2','H2S'])

    x1_roll, x1_roll_cols = roll_data(data_df, ['x_1'], roll=[5,7,9])

    x2_roll, x2_roll_cols = roll_data(data_df, ['x_2'], roll=[5,7,9])

    x3_roll, x3_roll_cols = roll_data(data_df, ['x_3'], roll=[5,7,9])

    x4_roll, x4_roll_cols = roll_data(data_df, ['x_4'], roll=[5,7,9])

    x5_roll, x5_roll_cols = roll_data(data_df, ['x_5'], roll=[5,7,9])


    cols = data_df.columns.tolist() + x1_roll_cols + x2_roll_cols + x3_roll_cols + x4_roll_cols + x5_roll_cols
    
    data_df = pd.concat([data_df, x1_roll, x2_roll, x3_roll, x4_roll, x5_roll, target_df], axis=1)

    data_df = data_df.iloc[9:, :]

    print('SRU Preprocess Done')
    print('Enhance Sequence columns', data_df.columns)

    return data_df, cols