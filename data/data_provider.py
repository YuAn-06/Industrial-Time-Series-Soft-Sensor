"""
Copyright (C) 2024
@ Name: data_provider.py
@ Time: 2024/12/10 17:42
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""
from data import Dataset_Custom, Dataset_Custom_4_Soft_Sensor, Dataset_LaggedMatrix_4_Soft_Sensor, Dataset_MultiMode, Dataset_MultiMode_4_Soft_Sensor
from torch.utils.data import DataLoader, Dataset
from typing import Union

data_dict = {
    # DC
    'DC': Dataset_Custom, 
    'DC_MultiMode': Dataset_MultiMode,
    'DC_Soft_Sensor': Dataset_Custom_4_Soft_Sensor,
    'DC_MultiMode_Soft_Sensor': Dataset_MultiMode_4_Soft_Sensor,
    # SRU
    'SRU': Dataset_Custom, 
    'SRU_Soft_Sensor': Dataset_Custom_4_Soft_Sensor,
    # PPGAS
    'PPGAS': Dataset_Custom,
    'PPGAS_Soft_Sensor': Dataset_Custom_4_Soft_Sensor,
    'PPGAS_MultiMode': Dataset_MultiMode,
    'PPGAS_MultiMode_Soft_Sensor': Dataset_MultiMode_4_Soft_Sensor,
    # Ironmaking
    'Ironmaking': Dataset_Custom,
    'Ironmaking_Soft_Sensor': Dataset_Custom_4_Soft_Sensor,
    'MP': Dataset_Custom,
    'MP_Soft_Sensor': Dataset_Custom_4_Soft_Sensor,
}   

LAGGED_MATRIX_MODELS = ['FASConvAELSTM']


def data_provider(args, flag: str)-> Union[Dataset, DataLoader]:
    data_name = args.data_name
    if 'PPGAS' in data_name:
        data_name = 'PPGAS'

    if args.model in ['DMVAER', 'DMRIFormer'] or args.use_condition_label:
        data_name = data_name + '_MultiMode'
        if args.task == 'soft_sensor':
            data_name = data_name + '_Soft_Sensor'
        elif args.task == 'short_term_forecasting':
            pass
        else:
            raise ValueError("Invalid task name: {}".format(args.task))
    else:
        if args.task == 'soft_sensor':
            data_name = data_name + '_Soft_Sensor'
        elif args.task == 'short_term_forecasting':
            pass
        else:
            raise ValueError("Invalid task name: {}".format(args.task))



    
    if args.model in LAGGED_MATRIX_MODELS and args.task == 'soft_sensor' and args.data_name == 'SRU':
        Data = Dataset_LaggedMatrix_4_Soft_Sensor
    else:
        Data = data_dict[data_name]

    timeenc = 0 if args.embed != "timeF" else 1
    if flag in ['valid', 'test']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

    dataset = Data(args, flag, timeenc = timeenc)


    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        )
    return dataset, data_loader
