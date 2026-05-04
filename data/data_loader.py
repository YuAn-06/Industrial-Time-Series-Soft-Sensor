"""
Copyright (C) 2024
@ Name: data_loader.py
@ Time: 2024/12/10 17:24
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""


import numpy as np
import pandas as pd
import torch


from utils import  *
from torch.utils.data import Dataset
from torch import nn

import matplotlib.pyplot as plt

preprocess_data_dict = {
    'DC': DC_preprocess,
    'SRU': SRU_preprocess
}


class Dataset_Custom(Dataset):
    """
    A specialized PyTorch Dataset class designed for Custom Time-Series Soft Sensor Forecasting tasks. 
    It handles CSV data loading, automated dataset splitting (Train/Valid/Test), 
    feature scaling, and time-feature encoding specifically formatted for 
    Encoder-Decoder architectures.
    Parameters:
    :param args: Configuration object containing:
        - data_path: Path to the CSV file.
        - data_name: Name of the dataset for specific preprocessing logic.
        - target: The name of the column to be predicted.
        - if_missing: Boolean flag for handling missing values.
        - model: Model name used to determine scaling strategy (e.g., MinMaxScaler for certain models).
        - seq_len: Input sequence length (look-back window).
        - label_len: Start token length for the decoder.
        - pred_len: Prediction horizon (forecast length).
        - data_aug: Boolean flag to enable data augmentation.
        - C_out: Number of output channels.
        - freq: Frequency for time feature encoding (e.g., 'h', 't', 's').
    :param flag: Dataset split identifier, must be in ['train', 'test', 'valid'].
    :param timeenc: Integer (0 or 1) indicating the type of time feature encoding to use.
    """
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
            if self.args.model in ['HSAM_dGRUs','ARDNN', 'GTFTS', 'GCT']:
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


        del_col = del_columns(self.data_name, self.target)
        
        if columns_with_x == []:
            columns_with_x = [col for col in self.df_raw.columns if col != del_col and col != "date" and col!="mode"]

        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

      

        data_x = self.df_raw[columns_with_x + [self.target]].values
        data_y = self.df_raw[columns_with_x + [self.target]].values
    
        
        train_data = data_x[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)

        if self.flag == 'test':
            self.scaler_y = StandardScaler()
            train_data_y = data_y[border1s[0]:border2s[0],-self.args.C_out:]
            self.scaler_y.fit(train_data_y)

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

        """
        Retrieves a single data sample (a sliding window) for the model.
        
        Logic Overview:
        1. Defines the sliding window boundaries for both Encoder and Decoder 
        2. x_enc: Historical data window used as input for the Model. # Size: [B, T, C_in]
        3. x_dec: Input for the Transformer based-Decoder, consisting of a 'start token' (label_len) followed 
           by zeros representing the period to be predicted (pred_len). # Size: [B, T + P - L, C_in]
        4. x_mark: Temporal features (e.g., hour, day) corresponding to the data windows. # Size: [B, T, C]
        5. batch_y: The target values for the prediction values. # Size: [B, P, C_out]
        """

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
        return self.scaler_y.inverse_transform(data)

class Dataset_Custom_4_Soft_Sensor(Dataset):
    """
    A specialized PyTorch Dataset class designed for Custom Time-Series Soft Sensor Regression and Sequential Estimation tasks. 
    It handles CSV data loading, automated dataset splitting (Train/Valid/Test), 
    feature scaling, and time-feature encoding specifically formatted for 
    Encoder-Decoder architectures.

    Parameters:
    :param args: Configuration object containing:
        - data_path: Path to the CSV file.
        - data_name: Name of the dataset for specific preprocessing logic.
        - target: The name of the column to be predicted.
        - if_missing: Boolean flag for handling missing values.
        - model: Model name used to determine scaling strategy (e.g., MinMaxScaler for certain models).
        - seq_len: Input sequence length (look-back window).
        - label_len: Start token length for the decoder.
        - pred_len: Prediction horizon (forecast length).
        - data_aug: Boolean flag to enable data augmentation.
        - C_out: Number of output channels.
        - freq: Frequency for time feature encoding (e.g., 'h', 't', 's').
    :param flag: Dataset split identifier, must be in ['train', 'test', 'valid'].
    :param timeenc: Integer (0 or 1) indicating the type of time feature encoding to use.
    """
    
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
            self.scaler_x = ZeroMaskStandardScaler()
            self.scaler_y = ZeroMaskStandardScaler()
        else:
            if self.args.model in ['HSAM_dGRUs','GCT','ARDNN', 'GTFTS']:
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


        del_col = del_columns(self.data_name, self.target)

        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode" and col!=del_col
            ]

        
        
        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        """
        For certain models, we need the target variable as input. Otherwise, we only use the input variables.
        """       
        if self.args.model in ['DAGRU','HSAM_dGRUs','GCT','STALSTM']:
            data_x = self.df_raw[columns_with_x + [self.target]].values 
        else:
             data_x = self.df_raw[columns_with_x].values

            
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
        Retrieves a single data sample (a sliding window) for the model.
        
        Logic Overview:
        1. Defines the sliding window boundaries for both Encoder and Decoder. 
        2. x_enc: Historical data window used as input for the Model. Size: [B, T, C_in]
        3. x_dec: Input for the Transformer based-Decoder, consisting of a 'start token' (label_len) 
           followed by zeros representing the period to be predicted (pred_len). Size: [B, T + P - L, C_in].
        4. x_mark: Temporal features (e.g., hour, day) corresponding to the data windows. Size: [B, T, C]
        5. batch_y: The target values for the prediction values at last time step T. Size: [B, 1, C_out]
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
    """
    A specialized PyTorch Dataset class designed for Custom Time-Series Multi-mode Soft Sensor forecasting tasks . 
    It handles CSV data loading, automated dataset splitting (Train/Valid/Test), 
    feature scaling, and time-feature encoding specifically formatted for 
    Encoder-Decoder architectures.

    Parameters:
    :param args: Configuration object containing:
        - data_path: Path to the CSV file.
        - data_name: Name of the dataset for specific preprocessing logic.
        - target: The name of the column to be predicted.
        - if_missing: Boolean flag for handling missing values.
        - model: Model name used to determine scaling strategy (e.g., MinMaxScaler for certain models).
        - seq_len: Input sequence length (look-back window).
        - label_len: Start token length for the decoder.
        - pred_len: Prediction horizon (forecast length).
        - data_aug: Boolean flag to enable data augmentation.
        - C_out: Number of output channels.
        - freq: Frequency for time feature encoding (e.g., 'h', 't', 's').
    :param flag: Dataset split identifier, must be in ['train', 'test', 'valid'].
    :param timeenc: Integer (0 or 1) indicating the type of time feature encoding to use.
    """
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

        del_col = del_columns(self.data_name, self.target)
        
        columns_with_x = [col for col in self.df_raw.columns if col.startswith("x_")]

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode" and col!=del_col
            ]
        
        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
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
            data_y = self.df_raw[columns_with_x + [self.target]].values
        
        else:
            # DMRI use process variables to predict quality variables (multi-step forecasting) without using labels
            data_x = self.df_raw[columns_with_x].values
            data_y = self.df_raw[columns_with_x + [self.target]].values
       
        train_data_x = data_x[border1s[0]:border2s[0]]
        train_data_y = data_y[border1s[0]:border2s[0]]

        if self.flag == 'test':
            self.scaler_only_y = StandardScaler()

            self.scaler_only_y.fit(train_data_y[:,-self.args.C_out:])
        
        self.scaler_x.fit(train_data_x)
        self.scaler_y.fit(train_data_y)

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
  
        # self.data_stamp = data_stamp


    def __getitem__(self, index):
        """
        Retrieves a single data sample (a sliding window) for the model.
        
        Logic Overview:
        1. Defines the sliding window boundaries for both Encoder and Decoder 
        2. x_enc: Historical data window used as input for the Model. # Size: [B, T, C_in]
        3. x_dec: Input for the Transformer based-Decoder, consisting of a 'start token' (label_len) followed 
           by zeros representing the period to be predicted (pred_len). # Size: [B, T + P - L, C_in]
        4. x_mark: Temporal features (e.g., hour, day) corresponding to the data windows. # Size: [B, T, C]
        5. c_enc: Mode label for the input sequence. # Size: [B, T, M]
        6. batch_y: The target values for the prediction values. # Size: [B, P, C_out]
        """

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
        return self.scaler_only_y.inverse_transform(data)
    

class Dataset_MultiMode_4_Soft_Sensor(Dataset):
    """
    A specialized PyTorch Dataset class designed for Custom Time-Series Multi-mode Soft Sensor Regression and Sequential Estimation tasks. 
    It handles CSV data loading, automated dataset splitting (Train/Valid/Test), 
    feature scaling, and time-feature encoding specifically formatted for 
    Encoder-Decoder architectures.

    Parameters:
    :param args: Configuration object containing:
        - data_path: Path to the CSV file.
        - data_name: Name of the dataset for specific preprocessing logic.
        - target: The name of the column to be predicted.
        - if_missing: Boolean flag for handling missing values.
        - model: Model name used to determine scaling strategy (e.g., MinMaxScaler for certain models).
        - seq_len: Input sequence length (look-back window).
        - label_len: Start token length for the decoder.
        - pred_len: Prediction horizon (forecast length).
        - data_aug: Boolean flag to enable data augmentation.
        - C_out: Number of output channels.
        - freq: Frequency for time feature encoding (e.g., 'h', 't', 's').
    :param flag: Dataset split identifier, must be in ['train', 'test', 'valid'].
    :param timeenc: Integer (0 or 1) indicating the type of time feature encoding to use.
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

        del_col = del_columns(self.data_name, self.target)

        if columns_with_x == []:
            columns_with_x = [
                col for col in self.df_raw.columns if col != self.target and col != "date" and col!="mode" and col!=del_col
            ]
        
        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        """
        Load mode labels for multi-mode tasks. If 'mode' column is not present, load from file.
        """
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
        self.label_mode = self.label_mode[border1:border2]
        # if self.args.if_missing:
        #     self.mask_label = pd.read_csv(self.args.missing_path).values
        #     self.mask_label = self.mask_label[border1:border2]
        

    def __getitem__(self, index):
        
        """
        Retrieves a single data sample (a sliding window) for the model.
        
        Logic Overview:
        1. Defines the sliding window boundaries for both Encoder and Decoder 
        2. x_enc: Historical data window used as input for the Model. # Size: [B, T, C_in]
        3. x_dec: Input for the Transformer based-Decoder, consisting of a 'start token' (label_len) followed 
           by zeros representing the period to be predicted (pred_len). # Size: [B, T + P - L, C_in]
        4. x_mark: Temporal features (e.g., hour, day) corresponding to the data windows. # Size: [B, T, C]
        5. c_enc: Mode label for the input sequence. # Size: [B, T, M]
        6. batch_y: The target values for the prediction values. # Size: [B, 1, C_out]
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

        dec_inp = np.zeros_like(seq_y[ -self.args.pred_len:, :])
        dec_inp = np.concatenate([seq_y[ :self.args.label_len, :], dec_inp], axis=0)
        
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
    

