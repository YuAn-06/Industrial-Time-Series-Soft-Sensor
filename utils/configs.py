import argparse
from utils.ExpConfigs import ExpConfigs
import yaml
from dataclasses import asdict
import os

def Init_parser():

    parser = argparse.ArgumentParser(description='Configuration for All Models')
    # Data Configs
    parser.add_argument('--model', type=str, default='ARDNN',help='Model name to use')
    parser.add_argument('--task', type=str, default='short_term_forecasting',help="Task type: ['short_term_forecasting', 'soft_sensor']")
    parser.add_argument('--data_path', type=str, default="",help="Dataset path")         
    parser.add_argument('--data_name', type=str, default="DC",help="Dataset name: [ 'DEB', 'SRU']")
    parser.add_argument('--target', type=str, default="",help="Target variable name")                
    parser.add_argument('--data_aug',  action='store_true',help='If use data augmentation for Deb and SRU dataset')                    
    parser.add_argument('--use_amp',  action='store_true',help='If use automatic mixed precision training')                   
    parser.add_argument('--num_workers', type=int, default=1,help='DataLoader the number of workers')                   
    parser.add_argument('--if_missing',  action='store_true',help='If exists Missing Data')                  
    parser.add_argument('--missing_rate', type=float, default=0.0,help='Missing data rate [0:0.5]')                
    parser.add_argument('--use_condition_label',  action='store_true',help='If use mode variable for multi_mode dataset')
    
    # Model Config
    parser.add_argument('--enc_in', type=int, default=16,help='Dimension of encoder Input')               
    parser.add_argument('--dec_in', type=int, default=16,help='Dimension of decoder Input')               
    parser.add_argument('--C_in', type=int, default=16,help='Dimension of Channel Input')                 
    parser.add_argument('--C_out', type=int, default=4,help='Dimension of Channel Output')
    parser.add_argument('--seq_len', type=int, default=32,help='Sequence length')
    parser.add_argument('--label_len', type=int, default=10, help='Label length')                 
    parser.add_argument('--patch_len', type=int, default=8,help='Patch length')      
    parser.add_argument('--pred_len', type=int, default=6, help='Prediction Length,re')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--embed', type=str, default='TimeF', help='Embedding type')            
    parser.add_argument('--freq', type=str, default='s', help='Time embedding frequency')
    parser.add_argument('--factor', type=int, default=1, help='Factor of Attention')    
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=1,help='Encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,help='Decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024,help='Feed forward dimension')
    parser.add_argument('--dropout', type=float, default=0.05,help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu',help='Activation function')



    # Train Config
    parser.add_argument('--collate_fn', type=str, default='collate_fn',   help='Collate function to use')           
    parser.add_argument('--batch_size', type=int, default=64,  help='Batch size for training')           
    parser.add_argument('--learning_rate', type=float, default=0.001,  help='Learning rate for optimizer')              
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')                
    parser.add_argument('--patience', type=int, default=10,help='Patience for early stopping')                  
    parser.add_argument('--lradj', type=str, default='cosine',help="Learning rate adjustment strategy: ['type1', 'type2', 'cosine']")
    parser.add_argument('--weight_decay', type=float, default=0.0,help='L2 regularization weight')         

    # Test Config
    parser.add_argument('--inverse', action='store_true', help='If the data is scaled, inverse the data to the original scale')

    # GPU config
    parser.add_argument('--use_cuda', default=False, help='Use CUDA for training')                
    parser.add_argument('--device', type=str, default="cuda",help='Device to use')                  
    parser.add_argument('--gpu', type=int, default=0,help='GPU ID to use')               
    parser.add_argument('--seed', type=int, default=2021,help='Random seed')                
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0],help='List of GPU device IDs')                 
    parser.add_argument('--use_multi_gpu', default=False, help='Use multiple GPUs')

    # Nystromformer config
    parser.add_argument('--num_landmarks', type=int, default=10,help='Number of landmarks')

    # TCVAE config
    parser.add_argument('--n_components', type=int, default=3,help='TCAVE type')

    # HSAM_dGRUs CVAESMC VA-LSTM
    parser.add_argument('--hidden_dim', type=int, default=10,help='hidden dimension for each distributed GRU unit')

    # Autoformer
    parser.add_argument('--moving_avg', type=int, default=25,help='moving average for Autoformer')

    # DLinear
    parser.add_argument('--individual', action='store_true',help='If use individual linear layer for each forecast')

    # Informer
    parser.add_argument('--distil', action='store_true',help='whether to use distilling in encoder, using this argument means not using distilling')
    
    # MSACNN
    parser.add_argument('--reduction_ratio', type=float, default=16,help='Reduction ratio for MSACNN')

    # Save config
    parser.add_argument('--save_dir', type=str, default='logs',help='Directory to save logs and models')

    # CVAESMC and DMVAER config
    parser.add_argument('--num_samples', type=int, default=10,help='Number of samples for CVAESMC')
    parser.add_argument('--z_dim', type=int, default=10,help='Latent dimension for CVAESMC and DMVAER')
    parser.add_argument('--output_type', type=str, default='mean',help='Output type for CVAESMC', choices=['mean', 'median', 'sample'])
    parser.add_argument('--z_global_dim', type=int, default=16,help='Global latent dimension for DMVAER')
    parser.add_argument('--z_local_dim', type=int, default=16,help='Local latent dimension for DMVAER')
    parser.add_argument('--DMVAER_loss_weight', type=list, default=[0.1, 1, 1, 1, 0.01],help='0: x reconstruction, 1: y reconstruction, 2: KL_zt, 3: KL_zs, 4: KL_C')


    # Nonstationary Transformer
    parser.add_argument('--p_hidden_dims', type=list, default=[128, 128],help='Hidden dimensions for projection network')
    parser.add_argument('--p_hidden_layers', type=int, default=2,help='Number of hidden layers for projection network')

    # EnvFormer & TCN
    parser.add_argument('--kernel_size', type=int, default=4,help='Kernel size for EnvFormer & TCN')

    # TimesMixer & SOFTS
    parser.add_argument('--down_sampling_window', type=int, default=4,help='Down sampling window for TimesMixer and TimeKAN ')
    parser.add_argument('--channel_independence', action='store_true',help='if channel independence for TimesNet')
    parser.add_argument('--top_k', type=int, default=5,help='Top k for TimesNet')
    parser.add_argument('--num_kernels', type=int, default=5,help='Number of kernels for TimesMixer')
    parser.add_argument('--decomp_method', type=str, default='none',help='[moving_avg, dft_decomp]')
    parser.add_argument('--down_sampling_layers', type=int, default=2,help='Number of down sampling layers for TimesMixer')
    parser.add_argument('--use_norm', action='store_true',help='If use normalization for TimesMixer')
    parser.add_argument('--down_sampling_method', type=str, default='max_pooling',help='[max, avg, conv')

    # GTFTS
    parser.add_argument('--latent_dim', type=int, default=10,help='Latent dimension for GTFTS')
    parser.add_argument('--n_fft', type=int, default=8,help='nfft dimension in STFT for GTFTS')
    # SparseTSF
    parser.add_argument('--model_type', type=str, default='linear',help='[linear, mlp]')
    parser.add_argument('--period_len', type=int, default=10,help='Period length for SparseTSF')
    
    # TCN
    parser.add_argument('--num_channels', type=list, default=[16, 32, 64],help='Number of channels for TCN')


    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # STALSTM
    parser.add_argument('--SA_dim', type=int, default=10, help='Spatial Attention dimension for TimeFilter')
    parser.add_argument('--TA_dim', type=int, default=10, help='Temporal Attention dimension for TimeFilter')


    # TimeKAN
    parser.add_argument('--begin_order', action='store_true',help='If use future temporal feature for TimeKAN')

    
    # GCN
    parser.add_argument('--conv_channel', type=int, default=32,help='Convolution channel for GCN')
    parser.add_argument('--skip_channel', type=int, default=32,help='Skip channel for GCN')
    parser.add_argument('--gcn_depth', type=int, default=2,help='GCN depth')
    parser.add_argument('--node_dim', type=int, default=10,help='Node dimension for GCN')
    parser.add_argument('--propalpha', type=float, default=0.1,help='Propagation alpha for GCN')

    # SOFTS
    parser.add_argument('--d_core', type=int, default=10, help='Core dimension for SOFTS')

    # FEDformer
    parser.add_argument('--version', type=str, default='fourier', help='Version of FEDformer: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random', help='Mode selection method for FEDformer: [random, low]')
    parser.add_argument('--modes', type=int, default=32, help='Number of modes to be selected for FEDformer')

    # Setting
    parser.add_argument('--setting', type=str, default='',help='Setting to use')




    args = ExpConfigs(**vars(parser.parse_args()))

    return args


def Parse_arguments(yaml_path=None):
    args = Init_parser()


    if yaml_path is not None:
        if os.path.exists(yaml_path):
            print('YAML exists')
        else:
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r", encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Modify the global args object directly
        for key, value in config["params"].items():
            setattr(args, key, value)

    

    if args.model == 'HSAM_dGRUs':
        setting = "{}_{}_{}_sl{}_hd{}_bt{}_lr{}_pat{}".format(
        args.data_name,
        args.model,
        args.task,
        args.seq_len,
        args.hidden_dim,
        args.batch_size,
        args.learning_rate,
        args.patience
    )

    else:
        setting = "{}_{}_{}_sl{}_dm{}_bt{}_lr{}_pat{}".format(
        args.data_name,
        args.model,
        args.task,
        args.seq_len,
        args.d_model,
        args.batch_size,
        args.learning_rate,
        args.patience
    )

    print(setting)
    folder_path = f"./results/{args.model}/" + setting +"/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    args.save_dir = folder_path

    config_dict = asdict(args)
    # Log setting
    
    selected_keys = ["data_name", "model", "task", "seq_len", "label_len", "pred_len", "dropout", "activation", "batch_size", "learning_rate", "epoch","d_model","d_ff","hidden_dim", "patience"]
    
    args.setting = ", ".join(f"{k}: {v}" for k, v in config_dict.items() if k in selected_keys)
    


    return args

