
"""
Copyright (C) 2025
@ Name: RNN_run.py
@ Time: 2025/1/17 16:03
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""




import yaml
import argparse
import numpy as np
import os

from exp.exp_factory import get_exp_by_model_and_task
from utils.print_configs import print_args
from utils.tools import *
from utils.print_configs import print_args
from utils.logger import Logger

# from utils.configs import args

from utils.configs import Parse_arguments

if __name__ == '__main__':

    yaml_name = "SS_task/Ironmaking_scripts/iTransformer.yaml"
    yaml_path = f"./scripts/{yaml_name}"
    

    args = Parse_arguments(yaml_path)
    
    logger = Logger(args.save_dir)


    print(f"====using {yaml_name}====")
    print('Args in experiment after YAML update:')
    print_args(args)

    logger.info(f"using configs: {args.setting}")

    setup_seed(args.seed)

    
    print_args(args)
    
    # choose exp
    exp = get_exp_by_model_and_task(args)
    

    logger.info("Start training...")
    exp.train(logger)

    logger.info("Start testing...")
    exp.test(logger)

    logger.remove_handles()
