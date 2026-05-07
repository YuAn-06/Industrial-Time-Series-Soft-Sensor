"""
Copyright (C) 2024
@ Name: print_configs.py
@ Time: 2024/9/6 16:11
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""


def print_args(args, indent: int = 4):
    """ print all arguments, grouped by category
    
    Args:
        args: object containing the arguments
        indent: number of spaces for indentation, default is 4
    """
    # group parameters by category
    basic_config = {
        'Model': getattr(args, 'model', None),
        'Data Name': getattr(args, 'data_name', None),
        'Data Path': getattr(args, 'data_path', None),
        'Target': getattr(args, 'target', None),
        'Features': getattr(args, 'features', None),
        'Use AMP': getattr(args, 'use_amp', None),
        'Num Workers': getattr(args, 'num_workers', None),
        'Data Dim': getattr(args, 'data_dim', None),
        'Data Aug': getattr(args, 'data_aug', None),
        'Time Encoding': getattr(args, 'timeenc', None),
    }
    
    model_config = {
        'Encoder Input': getattr(args, 'enc_in', None),
        'Decoder Input': getattr(args, 'dec_in', None),
        'Input Channels': getattr(args, 'C_in', None),
        'Output Channels': getattr(args, 'C_out', None),
        'Sequence Length': getattr(args, 'seq_len', None),
        'Label Length': getattr(args, 'label_len', None),
        'Prediction Length': getattr(args, 'pred_len', None),
        'Components': getattr(args, 'n_components', None),
        'Model Dimension': getattr(args, 'd_model', None),
        'Heads': getattr(args, 'n_heads', None),
        'Encoder Layers': getattr(args, 'e_layers', None),
        'Decoder Layers': getattr(args, 'd_layers', None),
        'Feed Forward Dim': getattr(args, 'd_ff', None),
        'Dropout': getattr(args, 'dropout', None),
        'Activation': getattr(args, 'activation', None),
    }
    
    train_config = {
        'Batch Size': getattr(args, 'batch_size', None),
        'Learning Rate': getattr(args, 'learning_rate', None),
        'Epochs': getattr(args, 'epoch', None),
        'Validation': getattr(args, 'if_valid', None),
        'Patience': getattr(args, 'patience', None),
        'LR Adjustment': getattr(args, 'lradj', None),
    }
    
    test_config = {
        'Inverse': getattr(args, 'inverse', None),
    }
    
    gpu_config = {
        'Use CUDA': getattr(args, 'use_cuda', None),
        'Device': getattr(args, 'device', None),
        'GPU ID': getattr(args, 'gpu', None),
        'Seed': getattr(args, 'seed', None),
        'Multi GPU': getattr(args, 'use_multi_gpu', None),
        'Device IDs': getattr(args, 'device_ids', None),
    }
    
    # 打印各个配置部分
    _print_section("Basic Config", basic_config, indent)
    _print_section("Model Config", model_config, indent)
    _print_section("Train Config", train_config, indent)
    _print_section("Test Config", test_config, indent)
    _print_section("GPU Config", gpu_config, indent)


def _print_section(title: str, config_dict: dict, indent: int):
    """打印配置的一个部分"""
    print(f"\033[1m{title}:\033[0m")
    for key, value in config_dict.items():
        if value is not None:
            print(f"{' ' * indent}{key:<20}: {value}")
    print()  # 空行分隔 