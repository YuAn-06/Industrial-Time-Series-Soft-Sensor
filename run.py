"""
Copyright (C) 2025
@ Name: RNN_run_with_yaml.py
@ Time: 2025/1/17 16:03
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""




import yaml
import argparse
import numpy as np
import os

from exp import get_exp_by_model_and_task

from utils import Parse_arguments, setup_seed, Logger,print_args

if __name__ == '__main__':

    
    configs = Parse_arguments()
    print_args(configs)

    #
    setting = "{}_{}_dm{}_sl{}__bt{}_lr{}_ep{}".format(
        configs.data_name,
        configs.model,
        configs.d_model,
        configs.seq_len,
        configs.batch_size,
        configs.learning_rate,
        configs.epoch,
    )
    print(setting)
    
    setup_seed(configs.seed)
    print_args(configs)
    
    logger = Logger(configs.save_dir)

    exp = get_exp_by_model_and_task(configs)
    
    print(configs.freq)
    
    logger.info("Start training...")
    exp.train(logger)

    logger.info("Start testing...")
    exp.test(logger)

    logger.remove_handles()