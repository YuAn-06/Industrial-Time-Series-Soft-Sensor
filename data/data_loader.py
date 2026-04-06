"""
Copyright (C) 2024
@ Name: data_loader.py
@ Time: 2024/12/10 17:24
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

import os
import numpy as np
import pandas as pd
from sympy.polys.specialpolys import dup_from_raw_dict
import torch

from utils.scaler import StandardScaler, ZeroMaskStandardScaler, MinMaxScaler
from utils.tools import  DC_preprocess, SRU_preprocess
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features
from torch import nn

preprocess_data_dict = {
    'DC': DC_preprocess,
    'SRU': SRU_preprocess
}


class Dataset_Custom(Dataset):
    def __init__(self, args, flag, timeenc):
        self.args = args
        self.timeenc = timeenc
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.target = args.target

        self.df_raw = pd.read_csv(self.data_path)
        self.data = self.df_raw.values
        
        self.flag = flag
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.process()
        

        
    
    def process(self):

        if self.args.if_missing:
            self.scaler = ZeroMaskStandardScaler()
        else:
            if self.args.model in ['HSAM_dGRUs','ARDNN']:
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()  # 
        
        TRAIN_SIZE = int(self.data.shape[0] * 0.7)
        TEST_SIZE = int(self.data.shape[0] * 0.2)
        VALID_SIZE = self.data.shape[0] - TRAIN_SIZE - TEST_SIZE

        border1s = [0, TRAIN_SIZE - self.args.seq_len, self.data.shape[0] - TEST_SIZE - self.args.seq_len]
        border2s = [TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE, self.data.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode"
            ]

        if self.data_name in ['DC', 'SRU'] and self.args.if_data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        

        data_x = self.df_raw[columns_with_x + [self.target]].values
        data_y = self.df_raw[columns_with_x + [self.target]].values
    
        # Sequence to sequence 训练方式 D TO D
        
        train_data = data_x[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)

        data_x = self.scaler.transform(data_x)
        data_y = self.scaler.transform(data_y)

        if "date" in self.df_raw.columns:
            df_stamp = pd.DataFrame(self.df_raw["date"])[border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(labels=["date"], axis=1).values

            elif self.args.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_stamp = data_stamp
            self.use_stamp = True


        
        else:
            self.data_stamp = torch.empty((0, 0), dtype=torch.int)
            self.use_stamp = False
        
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_end - self.args.label_len
        r_end = r_begin + self.args.label_len + self.args.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        dec_inp = np.zeros_like(seq_y[ -self.args.pred_len:, :])
        dec_inp = np.concatenate([seq_y[ :self.args.label_len, :], dec_inp], axis=0)
        if self.flag in ['train', 'valid']:
            dec_inp = seq_y if self.args.model == 'TCVAE' else dec_inp
        

        if self.use_stamp:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': torch.Tensor(seq_x_mark).float(),
                'x_mark_dec': torch.Tensor(seq_y_mark).float(),
                'batch_y': torch.Tensor(seq_y).float(),
            }
        else:
            seq_x_mark = torch.empty((0, 0), dtype=torch.int)
                
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }


    def __len__(self):
        return len(self.data_x) - self.args.seq_len - self.args.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_4_Soft_Sensor(Dataset):
    ## Soft Sensor Forecasting Task: For single step forecasting, we only use the last point of the sequence as the target value.
    
    def __init__(self, args, flag, timeenc):
        self.args = args
        self.timeenc = timeenc
        self.data_path = args.data_path
 
        
        self.data_name = args.data_name
        self.target = args.target

        self.df_raw = pd.read_csv(self.data_path)
        
        self.data = self.df_raw.values
        
        print(np.isnan(self.data).any())
        
        self.flag = flag

        assert flag in ['train', 'test', 'valid']
        
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.process()
    

    def process(self):

        if self.args.if_missing:
            self.scaler_x = ZeroMaskStandardScaler()
            self.scaler_y = ZeroMaskStandardScaler()
        else:
            if self.args.model == 'HSAM_dGRUs':
                self.scaler_x = MinMaxScaler()
                self.scaler_y = MinMaxScaler()
            else:
                self.scaler_x = StandardScaler() 
                self.scaler_y = StandardScaler()

        TRAIN_SIZE = int(self.data.shape[0] * 0.7)
        TEST_SIZE = int(self.data.shape[0] * 0.2)
        VALID_SIZE = self.data.shape[0] - TRAIN_SIZE - TEST_SIZE

        border1s = [0, TRAIN_SIZE - self.args.seq_len, self.data.shape[0] - TEST_SIZE - self.args.seq_len]
        border2s = [TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE, self.data.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode"
            ]

        
        
        if self.data_name in ['DC', 'SRU'] and self.args.if_data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)


            
        if self.args.model not in ['DAGRU','HSAM_dGRUs']:
            data_x = self.df_raw[columns_with_x].values
            
        else:
            data_x = self.df_raw[columns_with_x + [self.target]].values # DAGRU and HSAM_dGRUs needs last point of sequence as input

            
        data_y = self.df_raw[self.target].values.reshape(-1,1)
  
        train_data = data_x[border1s[0]:border2s[0]]
        self.scaler_x.fit(train_data)
        train_data = data_y[border1s[0]:border2s[0]]
        self.scaler_y.fit(train_data)


        data_x = self.scaler_x.transform(data_x)
        data_y = self.scaler_y.transform(data_y)

        
        
        if "date" in self.df_raw.columns:
            df_stamp = pd.DataFrame(self.df_raw["date"])[border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(labels=["date"], axis=1).values

            elif self.args.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_stamp = data_stamp
            self.use_stamp = True
        
        else:
            self.data_stamp = torch.empty((0, 0), dtype=torch.int)
            self.use_stamp = False
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        
        
        if self.args.if_missing:
            self.mask_label = pd.read_csv(self.args.missing_path).values
            self.mask_label = self.mask_label[border1:border2]
        

    def __getitem__(self, index):
        """ 
        input: process variable x [T: T + seq_len, C_in]
        output: quality variable y [T + seq_len - 1 : T + seq_len, C_out]
        """
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_begin + self.args.seq_len - 1
        r_end = r_begin + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
     


        dec_inp = np.zeros_like(seq_y[ -self.args.pred_len:, :])
        dec_inp = np.concatenate([seq_y[ :self.args.label_len, :], dec_inp], axis=0)
        if self.flag in ['train', 'valid']:
            dec_inp = seq_y if self.args.model == 'TCVAE' else dec_inp
        
    
        if self.use_stamp:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': torch.Tensor(seq_x_mark).float(),
                'x_mark_dec': torch.Tensor(seq_y_mark).float(),
                'batch_y': torch.Tensor(seq_y).float(),
            }
        else:
            seq_x_mark = torch.empty((0, 0), dtype=torch.int)
                
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }

    def __len__(self):
        return len(self.data_x) - self.args.seq_len  + 1

    def inverse_transform(self, data):
        # only y need inverse transform
        return self.scaler_y.inverse_transform(data)


class Dataset_MultiMode(Dataset):
    """ Multi-Mode Dataset for Multi-Step Forecasting Task
    """
    def __init__(self, args, flag, timeenc):
        self.args = args
        self.timeenc = timeenc
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.target = args.target

        self.df_raw = pd.read_csv(self.data_path)
        self.data = self.df_raw.values
        

        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.process()
    
    def process(self):

        if self.args.if_missing:
            self.scaler_x = ZeroMaskStandardScaler()
            self.scaler_y = ZeroMaskStandardScaler()
        else:
            if self.args.model == 'HSAM_dGRUs':
                self.scaler_x = MinMaxScaler()
                self.scaler_y = MinMaxScaler()
            else:
                self.scaler_x = StandardScaler() 
                self.scaler_y = StandardScaler() 

        TRAIN_SIZE = int(self.data.shape[0] * 0.7)
        TEST_SIZE = int(self.data.shape[0] * 0.2)
        VALID_SIZE = self.data.shape[0] - TRAIN_SIZE - TEST_SIZE

        border1s = [0, TRAIN_SIZE - self.args.seq_len, self.data.shape[0] - TEST_SIZE - self.args.seq_len]
        border2s = [TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE, self.data.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        

        
        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode"
            ]
        
        if self.data_name in ['DC', 'SRU'] and self.args.if_data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        
        if 'mode' in self.df_raw.columns:
            self.label_mode = self.df_raw["mode"].values
        else:
            if self.data_name == 'DC':
                self.label_mode = np.loadtxt('./data/DC/mode_labels.txt')
            else:
                raise ValueError("Data mode is not defined")

        self.data = self.df_raw[columns_with_x].values
            
        
        if self.args.model not in ['DMRIFormer']:
            data_x = self.df_raw[columns_with_x + [self.target]].values
            data_y = self.df_raw[columns_with_x + [self.target]].values.reshape(-1,1)
        
        else:
            # DMRI use process variables to predict quality variables (multi-step forecasting) without using labels
            data_x = self.df_raw[columns_with_x].values
            data_y = self.df_raw[columns_with_x + [self.target]].values
       
        train_data_x = data_x[border1s[0]:border2s[0]]
        train_data_y = data_y[border1s[0]:border2s[0]]
        self.scaler_x.fit(train_data_x)
        self.scaler_y.fit(train_data_y)
        data_x = self.scaler_x.transform(data_x)
        data_y = self.scaler_y.transform(data_y)


        train_data = data_x[border1s[0]:border2s[0]]
        train_data_y = data_y[border1s[0]:border2s[0]]
        
        self.scaler_x.fit(train_data)
        self.scaler_y.fit(train_data_y)

        if "date" in self.df_raw.columns:
            df_stamp = pd.DataFrame(self.df_raw["date"])[border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(labels=["date"], axis=1).values

            elif self.args.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_stamp = data_stamp
            self.use_stamp = True
        
        else:
            self.data_stamp = torch.empty((0, 0), dtype=torch.int)
            self.use_stamp = False

        
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        # self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_end - self.args.label_len
        r_end = r_begin + self.args.label_len + self.args.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_c = self.label_mode[s_begin:r_end]
    

        dec_inp = np.zeros_like(seq_y[ -self.args.pred_len:, :])
        dec_inp = np.concatenate([seq_y[ :self.args.label_len, :], dec_inp], axis=0)
        

        # Mode Label
        c_enc = torch.Tensor(seq_c).long()
        
        c_enc = nn.functional.one_hot(c_enc, self.args.n_components)
        if self.use_stamp:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'c_enc': c_enc.float(),
                'x_mark_enc': torch.Tensor(seq_x_mark).float(),
                'x_mark_dec': torch.Tensor(seq_y_mark).float(),
                'batch_y': torch.Tensor(seq_y).float(),
            }
        else:
            seq_x_mark = torch.empty((0, 0), dtype=torch.int)
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'c_enc': c_enc.float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }

    def __len__(self):
        return len(self.data_x) - self.args.seq_len - self.args.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler_y.inverse_transform(data)
    

class Dataset_MultiMode_4_Soft_Sensor(Dataset):
    def __init__(self, args, flag, timeenc):
        self.args = args
        self.timeenc = timeenc
        self.data_path = args.data_path
 
        
        self.data_name = args.data_name
        self.target = args.target

        self.df_raw = pd.read_csv(self.data_path)
        self.data = self.df_raw.values
        

        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.process()
    
    def process(self):

        if self.args.if_missing:
            self.scaler_x = ZeroMaskStandardScaler()
            self.scaler_y = ZeroMaskStandardScaler()
        else:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        
        TRAIN_SIZE = int(self.data.shape[0] * 0.7)
        TEST_SIZE = int(self.data.shape[0] * 0.2)
        VALID_SIZE = self.data.shape[0] - TRAIN_SIZE - TEST_SIZE

        border1s = [0, TRAIN_SIZE - self.args.seq_len, self.data.shape[0] - TEST_SIZE - self.args.seq_len]
        border2s = [TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE, self.data.shape[0]]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode"
            ]
        
        if self.data_name in ['DC', 'SRU'] and self.args.if_data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        if 'mode' in self.df_raw.columns:
            self.label_mode = self.df_raw["mode"].values
        else:
            if self.data_name == 'DC':
                self.label_mode = np.loadtxt('./data/DC/mode_labels.txt')
            else:
                raise ValueError("Data mode is not defined")

        self.data = self.df_raw[columns_with_x].values
               
        
        if self.args.model != 'DAGRU':
            data_x = self.df_raw[columns_with_x].values
            
        else:
            data_x = self.df_raw[columns_with_x + [self.target]].values
            
        data_y = self.df_raw[self.target].values.reshape(-1,1)
  
        train_data = data_x[border1s[0]:border2s[0]]
        self.scaler_x.fit(train_data)
        train_data = data_y[border1s[0]:border2s[0]]
        self.scaler_y.fit(train_data)


        data_x = self.scaler_x.transform(data_x)
        data_y = self.scaler_y.transform(data_y)

        

        df_stamp = pd.DataFrame(self.df_raw["date"])[border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(labels=["date"], axis=1).values

        elif self.args.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.args.freq)
            data_stamp = data_stamp.transpose(1, 0)



        
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.data_stamp = data_stamp
        self.label_mode = self.label_mode[border1:border2]
        if self.args.if_missing:
            self.mask_label = pd.read_csv(self.args.missing_path).values
            self.mask_label = self.mask_label[border1:border2]
        

    def __getitem__(self, index):
        """ 
        input: process variable x [T: T + seq_len, C_in]
        output: quality variable y [T + seq_len - 1 : T + seq_len, C_out]
        """
        s_begin = index
        s_end = s_begin + self.args.seq_len
        r_begin = s_begin + self.args.seq_len - 1
        r_end = r_begin + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_c = self.label_mode[s_begin: s_end]

        c_enc = torch.Tensor(seq_c).long()
        
        c_enc = nn.functional.one_hot(c_enc, self.args.n_components)
        
        if self.use_stamp:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'c_enc': c_enc.float(),
                'x_mark_enc': torch.Tensor(seq_x_mark).float(),
                'x_mark_dec': torch.Tensor(seq_y_mark).float(),
                'batch_y': torch.Tensor(seq_y).float(),
            }
        else:
            seq_x_mark = torch.empty((0, 0), dtype=torch.int)
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'c_enc': c_enc.float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }

    def __len__(self):
        return len(self.data_x) - self.args.seq_len  + 1

    def inverse_transform(self, data):
        # only y need inverse transform
        return self.scaler_y.inverse_transform(data)
    

