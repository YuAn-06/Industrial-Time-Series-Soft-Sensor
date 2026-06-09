# Copyright (C) 2021 #
# @Time    : 2023/6/26 10:37
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : exp_basic.py
# @Software: PyCharm


import os
import torch
from typing import Dict, Union
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
    TCN,
    TimeFilter,
    STALSTM,
    Koopa,
    TimeKAN,
    MSGNet,
    DLSTM,
    GCT,
    SOFTS,
    FEDformer,
    STDTAEm,
    GraphSAGE_IMATCN,
)


MODEL_REGISTRY = {
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
    'LDCNN': LDCNN,
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
    'TCN': TCN,
    'TimeFilter': TimeFilter,
    'STALSTM': STALSTM,
    'Koopa': Koopa,
    'TimeKAN': TimeKAN,
    'MSGNet': MSGNet,
    'DLSTM': DLSTM,
    'GCT': GCT,
    'SOFTS': SOFTS,
    'FEDformer': FEDformer,
    'STDTAEm': STDTAEm,
    'GraphSAGE_IMATCN': GraphSAGE_IMATCN,
}


class Exp_basic(object):
    def __init__(self,args):

        self.args = args
        self.device = self._acquire_device()
        
        self.model_dict = MODEL_REGISTRY
    

        self.model = self._build_model().to(self.device)


    def _build_model(self):

        raise  NotImplementedError

        return None

    def _get_model_module(self, model_name):
        try:
            model_module = self.model_dict[model_name]
        except KeyError:
            supported_models = ', '.join(sorted(self.model_dict.keys()))
            raise ValueError(
                f"Unsupported model: {model_name}. Supported models: {supported_models}"
            )

        if not hasattr(model_module, "Model"):
            raise AttributeError(f"Model module '{model_name}' must define 'Model'.")

        return model_module

    def _normalize_flag(self, flag: str) -> str:
        return flag.lower()

    def _select_forecast_target(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, -self.args.pred_len:, -self.args.C_out:]

    def _select_last_time_target(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, -1, :]

    def _squeeze_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze(1)

    def _require_output_key(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        if not isinstance(outputs, dict) or key not in outputs:
            raise KeyError(f"Model '{self.args.model}' output must contain key '{key}'.")
        return outputs[key]

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
