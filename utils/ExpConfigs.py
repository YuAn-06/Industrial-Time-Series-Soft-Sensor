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
    # if_data_aug: bool
    
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

    # HSAM_dGRUs & VALSTM
    hidden_dim: int

    # Autoformer & TimesNet
    moving_avg: int

    # Informer
    distil: bool

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
    DMVAER_loss_weight: list

    # Nonstationary Transformer
    p_hidden_dims: list
    p_hidden_layers: int

    # EnvFormer
    kernel_size: int

    # TimesNet
    down_sampling_window: int
    channel_independence: bool
    top_k: int
    num_kernels: int
    decomp_method: str
    down_sampling_layers: int
    use_norm: bool
    down_sampling_method: str

    

    # GTFTS
    latent_dim: int
    n_fft: int

    # TimeFilter
    alpha: float
    top_p: float
    pos: int

    # TCN
    num_channels: list

    # SparseTSF
    model_type: str
    period_len: int

    # STALSTM
    SA_dim: int
    TA_dim: int

    # TimeKAN
    begin_order: bool

    # Setting
    setting: str



