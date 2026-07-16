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

preprocess_data_dict = {
    'DC': DC_preprocess,
    'SRU': SRU_preprocess
}

MINMAX_MODELS = [
    'HSAM_dGRUs',
    'ARDNN',
    'GTFTS',
    'GCT',
    'STDTAEm',
    'GraphSAGE_IMATCN',
    'FASConvAELSTM',
]


def _get_borders(data_len: int, seq_len: int, set_type: int):
    train_size = int(data_len * 0.7)
    test_size = int(data_len * 0.2)
    valid_size = data_len - train_size - test_size

    border1s = [0, train_size - seq_len, data_len - test_size - seq_len]
    border2s = [train_size, train_size + valid_size, data_len]

    border1 = border1s[set_type]
    border2 = border2s[set_type]
    if border1 < 0 or border2 <= border1:
        raise ValueError(
            f"Invalid data split borders: border1={border1}, border2={border2}. "
            f"Please check data length ({data_len}) and seq_len ({seq_len})."
        )

    return border1s, border2s, border1, border2


def _get_feature_columns(df_raw: pd.DataFrame, data_name: str, target: str, include_target_when_no_x: bool):
    excluded = {"date", "mode", target}
    return [col for col in df_raw.columns if col not in excluded]


def _build_time_features(df_raw: pd.DataFrame, border1: int, border2: int, timeenc: int, freq: str):
    if "date" not in df_raw.columns:
        return torch.empty((0, 0), dtype=torch.float32), False

    df_stamp = pd.DataFrame(df_raw["date"])[border1:border2].copy()
    dates = pd.to_datetime(df_stamp["date"])
    if timeenc == 0:
        df_stamp["month"] = dates.dt.month
        df_stamp["day"] = dates.dt.day
        df_stamp["weekday"] = dates.dt.weekday
        df_stamp["hour"] = dates.dt.hour
        data_stamp = df_stamp.drop(labels=["date"], axis=1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    else:
        raise ValueError(f"Unsupported timeenc: {timeenc}")

    return data_stamp, True


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
    def __init__(self, args, flag: str, timeenc: int):
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

        if self.args.model in MINMAX_MODELS:
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        border1s, border2s, border1, border2 = _get_borders(
            self.data.shape[0], self.args.seq_len, self.set_type
        )

        columns_with_x = _get_feature_columns(
            self.df_raw, self.data_name, self.target, include_target_when_no_x=True
        )

        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

      

        data_x = self.df_raw[columns_with_x + [self.target]].values
        data_y = self.df_raw[columns_with_x + [self.target]].values
    
        
        train_data = data_x[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)

        if self.flag == 'test':
            self.scaler_y = MinMaxScaler() if self.args.model in MINMAX_MODELS else StandardScaler()
            train_data_y = data_y[border1s[0]:border2s[0],-self.args.C_out:]
            self.scaler_y.fit(train_data_y)

        data_x = self.scaler.transform(data_x)
        data_y = self.scaler.transform(data_y)

        self.data_stamp, self.use_stamp = _build_time_features(
            self.df_raw, border1, border2, self.timeenc, self.args.freq
        )
        
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
            seq_x_mark = torch.empty((0, 0), dtype=torch.float32)
                
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }


    def __len__(self):
        return len(self.data_x) - self.args.seq_len - self.args.pred_len + 1
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scaler_y"):
            raise RuntimeError("inverse_transform is only available after initializing the test split.")
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
    
    def __init__(self, args, flag: str, timeenc: int):
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

        if self.args.model in MINMAX_MODELS:
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()

        border1s, border2s, border1, border2 = _get_borders(
            self.data.shape[0], self.args.seq_len, self.set_type
        )

        columns_with_x = _get_feature_columns(
            self.df_raw, self.data_name, self.target, include_target_when_no_x=False
        )

        
        
        if self.data_name in ['DC', 'SRU'] and self.args.data_aug:
            self.df_raw, columns_with_x = preprocess_data_dict[self.data_name](self.df_raw, self.target)

        """
        For certain models, we need the target variable as input. Otherwise, we only use the input variables.
        """       
        if self.args.model in ['DAGRU','HSAM_dGRUs','GCT','STALSTM','TSLambdaGRU']:
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

        
        
        self.data_stamp, self.use_stamp = _build_time_features(
            self.df_raw, border1, border2, self.timeenc, self.args.freq
        )
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        
        
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
        if self.args.model == 'STDTAEm':
            r_begin = s_begin
            r_end = s_end
        else:
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
            seq_x_mark = torch.empty((0, 0), dtype=torch.float32)
                
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': seq_x_mark,
                'x_mark_dec': seq_x_mark,
                'batch_y': torch.Tensor(seq_y).float(),
            }

    def __len__(self):
        return len(self.data_x) - self.args.seq_len  + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # only y need inverse transform
        shape = data.shape
        if data.ndim == 3:
            data = data.reshape(-1, shape[-1])
            return self.scaler_y.inverse_transform(data).reshape(shape)
        return self.scaler_y.inverse_transform(data)


class Dataset_LaggedMatrix_4_Soft_Sensor(Dataset_Custom_4_Soft_Sensor):
    """
    Soft-sensor dataset that builds a two-dimensional lag matrix for CNN-style
    models. For each sample at time t, x_enc rows are selected by lag offsets
    such as [0, 3, 5, 9], producing [num_lags, num_variables].
    """

    def process(self):
        if self.args.model in MINMAX_MODELS:
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()

        self.lag_offsets = [int(lag) for lag in getattr(self.args, "fa_lags", [0, 3, 5, 9])]
        if not self.lag_offsets or min(self.lag_offsets) < 0:
            raise ValueError("fa_lags must contain non-negative integer offsets.")
        self.max_lag = max(self.lag_offsets)

        border1s, border2s, border1, border2 = _get_borders(
            self.data.shape[0], self.max_lag + 1, self.set_type
        )

        columns_with_x = _get_feature_columns(
            self.df_raw, self.data_name, self.target, include_target_when_no_x=False
        )
        if self.data_name == "SRU":
            columns_with_x = [col for col in columns_with_x if col != "H2S"]
        elif self.data_name != "DC":
            raise ValueError(
                "Dataset_LaggedMatrix_4_Soft_Sensor currently supports SRU and DC."
            )

        data_x = self.df_raw[columns_with_x].values
        data_y = self.df_raw[self.target].values.reshape(-1, 1)

        self.scaler_x.fit(data_x[border1s[0]:border2s[0]])
        self.scaler_y.fit(data_y[border1s[0]:border2s[0]])

        data_x = self.scaler_x.transform(data_x)
        data_y = self.scaler_y.transform(data_y)

        self.data_stamp, self.use_stamp = _build_time_features(
            self.df_raw, border1, border2, self.timeenc, self.args.freq
        )
        self.data_x = data_x
        self.data_y = data_y
        self.border1 = border1
        self.border2 = border2
        self.feature_columns = columns_with_x

    def __getitem__(self, index):
        center = self.border1 + self.max_lag + index
        # LSTM expects time to move from the oldest observation to the newest.
        # For fa_lags=[0, 3, 5, 9], feed [t-9, t-5, t-3, t] without changing
        # the configured lag points themselves.
        lag_indices = [
            center - lag
            for lag in sorted(self.lag_offsets, reverse=True)
        ]
        seq_x = self.data_x[lag_indices]
        seq_y = self.data_y[center:center + 1]

        dec_inp = np.zeros_like(seq_y)
        if self.use_stamp:
            seq_x_mark = self.data_stamp[index + self.max_lag:index + self.max_lag + 1]
            seq_x_mark = np.repeat(seq_x_mark, len(self.lag_offsets), axis=0)
            seq_y_mark = self.data_stamp[index + self.max_lag:index + self.max_lag + 1]
            return {
                'x_enc': torch.Tensor(seq_x).float(),
                'x_dec': torch.Tensor(dec_inp).float(),
                'x_mark_enc': torch.Tensor(seq_x_mark).float(),
                'x_mark_dec': torch.Tensor(seq_y_mark).float(),
                'batch_y': torch.Tensor(seq_y).float(),
            }

        seq_x_mark = torch.empty((0, 0), dtype=torch.float32)
        return {
            'x_enc': torch.Tensor(seq_x).float(),
            'x_dec': torch.Tensor(dec_inp).float(),
            'x_mark_enc': seq_x_mark,
            'x_mark_dec': seq_x_mark,
            'batch_y': torch.Tensor(seq_y).float(),
        }

    def __len__(self):
        return self.border2 - self.border1 - self.max_lag


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
    def __init__(self, args, flag: str, timeenc: int):
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

        if self.args.model == 'HSAM_dGRUs':
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()

        border1s, border2s, border1, border2 = _get_borders(
            self.data.shape[0], self.args.seq_len, self.set_type
        )

        columns_with_x = _get_feature_columns(
            self.df_raw, self.data_name, self.target, include_target_when_no_x=False
        )
        
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
            if self.args.model in MINMAX_MODELS:
                self.scaler_only_y = MinMaxScaler()
            else:
                self.scaler_only_y = StandardScaler()
            self.scaler_only_y.fit(train_data_y[:,-self.args.C_out:])
        
        self.scaler_x.fit(train_data_x)
        self.scaler_y.fit(train_data_y)

        data_x = self.scaler_x.transform(data_x)
        data_y = self.scaler_y.transform(data_y)


        self.data_stamp, self.use_stamp = _build_time_features(
            self.df_raw, border1, border2, self.timeenc, self.args.freq
        )

        
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
            seq_x_mark = torch.empty((0, 0), dtype=torch.float32)
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
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scaler_only_y"):
            raise RuntimeError("inverse_transform is only available after initializing the test split.")
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
    def __init__(self, args, flag: str, timeenc: int):
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

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        border1s, border2s, border1, border2 = _get_borders(
            self.data.shape[0], self.args.seq_len, self.set_type
        )

        columns_with_x = _get_feature_columns(
            self.df_raw, self.data_name, self.target, include_target_when_no_x=False
        )
        
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

        

        self.data_stamp, self.use_stamp = _build_time_features(
            self.df_raw, border1, border2, self.timeenc, self.args.freq
        )



        
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        self.label_mode = self.label_mode[border1:border2]

        

    def __getitem__(self, index: int):
        
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
            seq_x_mark = torch.empty((0, 0), dtype=torch.float32)
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

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        # only y need inverse transform
        return self.scaler_y.inverse_transform(data)
    

