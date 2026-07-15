# Copyright (C) 2021 #
# @Time    : 2023/6/26 10:37
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : exp_basic.py
# @Software: PyCharm


import os
import importlib
import torch
from types import ModuleType
from typing import Dict, Union


MODEL_REGISTRY = {
    'Nystromformer': 'models.Nystromformer',
    'DAGRU': 'models.DAGRU',
    'DMVAER': 'models.DMVAER',
    'VRNN': 'models.VRNN',
    'TCVAE': 'models.T_CVAE',
    'iTransformer': 'models.iTransformer',
    'Transformer': 'models.Transformer',
    'EnvFormer': 'models.EnvFormer',
    'Fredformer': 'models.Fredformer',
    'HSAM_dGRUs': 'models.HSAM_dGRUs',
    'PatchTST': 'models.PatchTST',
    'Autoformer': 'models.Autoformer',
    'DLinear': 'models.DLinear',
    'ARDNN': 'models.ARDNN',
    'MSACNN': 'models.MSACNN',
    'CVAESMC': 'models.CVAESMC',
    'LDCNN': 'models.LDCNN',
    'Nonstationary_Transformer': 'models.Nonstationary_Transformer',
    'DMRIFormer': 'models.DMRIFormer',
    'Informer': 'models.Informer',
    'LSTM': 'models.LSTM',
    'VALSTM': 'models.VALSTM',
    'Crossformer': 'models.Crossformer',
    'TimeMixer': 'models.TimeMixer',
    'TimesNet': 'models.TimesNet',
    'GTFTS': 'models.GTFTS',
    'SparseTSF': 'models.SparseTSF',
    'TCN': 'models.TCN',
    'TimeFilter': 'models.TimeFilter',
    'STALSTM': 'models.STALSTM',
    'Koopa': 'models.Koopa',
    'TimeKAN': 'models.TimeKAN',
    'MSGNet': 'models.MSGNet',
    'DLSTM': 'models.DLSTM',
    'GCT': 'models.GCT',
    'SOFTS': 'models.SOFTS',
    'FEDformer': 'models.FEDformer',
    'STDTAEm': 'models.STDTAEm',
    'GraphSAGE_IMATCN': 'models.GraphSAGE_IMATCN',
    'FASConvAELSTM': 'models.FASConvAELSTM',
    'TSLambdaGRU': 'models.TSLambdaGRU',
}


class Exp_basic(object):
    def __init__(self,args):

        self.args = args
        self.device = self._acquire_device()
        
        self.model_dict = MODEL_REGISTRY
        self._model_module_cache: Dict[str, ModuleType] = {}
    

        self.model = self._build_model().to(self.device)


    def _build_model(self):

        raise  NotImplementedError

        return None

    def _get_model_module(self, model_name):
        try:
            module_path = self.model_dict[model_name]
        except KeyError:
            supported_models = ', '.join(sorted(self.model_dict.keys()))
            raise ValueError(
                f"Unsupported model: {model_name}. Supported models: {supported_models}"
            )

        if model_name not in self._model_module_cache:
            self._model_module_cache[model_name] = importlib.import_module(module_path)

        model_module = self._model_module_cache[model_name]

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
