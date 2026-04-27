# Copyright (C) 2021 #
# @Time    : 2023/6/26 10:37
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : exp_basic.py
# @Software: PyCharm


import os
import torch
from models import (
    Nystromformer,
    DAGRU,
    DMVAER,
    VRNN,
    T_CVAE,
    iTransformer,
    Transformer,
    EnvFormer,
    Fredformer,
    HSAM_dGRUs,
    PatchTST,
    Autoformer,
    DLinear,
    ARDNN,
    MSACNN,
    CVAESMC,
    LDCNN,
    Nonstationary_Transformer,
    DMRIFormer,
    Informer,
    VALSTM,
    LSTM,
    Crossformer,
    TimeMixer,
    TimesNet,
    GTFTS,
    SparseTSF,
    
)

class Exp_basic(object):
    def __init__(self,args):

        self.args = args
        self.device = self._acquire_device()
        
        self.model_dict = {
            'Nystromformer': Nystromformer,
            'DAGRU': DAGRU,
            'DMVAER': DMVAER,
            'VRNN': VRNN,
            'TCVAE': T_CVAE,
            'iTransformer': iTransformer,
            'Transformer': Transformer,
            'EnvFormer': EnvFormer,
            'Fredformer': Fredformer,
            'HSAM_dGRUs': HSAM_dGRUs,
            'PatchTST': PatchTST,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'ARDNN': ARDNN,
            'MSACNN': MSACNN,
            'CVAESMC': CVAESMC,
            'LDCNN':LDCNN,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DMRIFormer': DMRIFormer,
            'Informer': Informer,
            'LSTM': LSTM,
            'VALSTM': VALSTM,
            'Crossformer': Crossformer,
            'TimeMixer': TimeMixer,
            'TimesNet': TimesNet,
            'GTFTS': GTFTS,
            'SparseTSF': SparseTSF,
        }
    

        self.model = self._build_model().to(self.device)


    def _build_model(self):

        raise  NotImplementedError

        return None

    def _acquire_device(self):

        if self.args.use_cuda:

            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('====use gpu=====')

        else:
            device = torch.device('cpu')
            print('====use cpu=====')

        return device


    def _get_data(self):
        pass



    def train(self,setting):
        pass

    def test(self,setting):
        pass

    def valid(self,dataloader):
        pass