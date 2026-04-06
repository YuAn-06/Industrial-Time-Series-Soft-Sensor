from dataclasses import dataclass
from typing import List

@dataclass
class ExpConfigs:
    model: str
    task: str
    data_name: str
    data_path: str
    data_aug: bool
    target: str
    use_amp: bool
    num_workers: int
    if_missing: bool
    missing_rate: float
    use_condition_label: bool
    if_data_aug: bool
    
    # Model Config
    enc_in: int
    dec_in: int 
    C_in: int 
    C_out: int 
    seq_len: int 
    label_len: int 
    patch_len: int
    stride: int
    pred_len: int 
    embed: str 
    freq: str 
    factor: int 
    d_model: int 
    n_heads: int 
    e_layers: int 
    d_layers: int 
    d_ff: int 
    dropout: float 
    activation: str 
    

    # Train Config
    collate_fn: str 
    batch_size: int 
    learning_rate: float 
    epoch: int
    if_valid: bool 
    patience: int 
    lradj: str 
    weight_decay: float
    save_dir: str 

    # Test Config
    inverse: bool 

    # GPU config
    use_cuda: bool 
    device: str 
    gpu: int 
    seed: int 
    device_ids: List[int]   # type: ignore
    use_multi_gpu: bool


    # Nystroformer
    num_landmarks: int

    # TCVAE
    n_components: int

    # HSAM_dGRUs
    hidden_dim: int

    # Autoformer
    moving_avg: int

    # DLinear
    individual: bool

    # MSACNN
    reduction_ratio: float

    # CVAESMC and DMVAER
    num_samples: int
    z_dim: int
    output_type: str
    z_global_dim: int
    z_local_dim: int

    # Nonstationary Transformer
    p_hidden_dims: list
    p_hidden_layers: int

    # EnvFormer
    kernel_size: int

    # Setting
    setting: str



