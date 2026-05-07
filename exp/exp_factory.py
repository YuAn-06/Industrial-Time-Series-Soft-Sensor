# Copyright (C) 2021
# @Time    : 2025/12/16 11:00
# @Author  : 
# @Email   : 
# @File    : exp_factory.py
# @Software: PyCharm



from exp import Exp_Short_Term_Forecasting, Exp_Soft_Sensor
from exp import Exp_basic



def get_exp_by_model_and_task(args) -> Exp_basic:
    """
    Choose the appropriate Exp class based on the model name and task type
    """
    task = getattr(args, 'task', None)
    model = getattr(args, 'model', None)
    
    if not task or not model:
        raise ValueError("args must contain both 'task' and 'model' attributes")
    

    if task == 'soft_sensor':
        return Exp_Soft_Sensor(args)
    elif task == 'short_term_forecasting':    
        return Exp_Short_Term_Forecasting(args)
    
    raise ValueError(f"Unsupported task: {task}")
