from __future__ import annotations

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
    use_tensorboard: bool
    
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
    patience: int 
    lradj: str 
    weight_decay: float
    model_stage: str
    pretrained_ckpt: str
    pretrain_epoch: int
    finetune_epoch: int
    pretrain_learning_rate: float
    finetune_learning_rate: float
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
    use_true_y_in_train: bool

    # FASConvAELSTM
    fa_lags: list[int]
    fa_kernel_sizes: list[int]
    fa_channels: list[int]
    fa_pretrain_learning_rates: list[float]

    # TSLambdaGRU
    lambda1: float
    lambda2: float

    # Autoformer & TimesNet
    moving_avg: int

    # Informer
    distil: bool

    # DLinear
    individual: bool

    # MSACNN
    reduction_ratio: int
    out_per_channel: List[int]

    # CVAESMC and DMVAER
    num_samples: int
    z_dim: int
    output_type: str
    z_global_dim: int
    z_local_dim: int
    DMVAER_loss_weight: list[float]

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
    std_window: int
    tae_beta: float
    triplet_margin: float
    tae_noise_std: float
    tae_mask_ratio: float
    tae_backbone: str
    tae_hidden_dims: list[int]
    freeze_tae: bool

    # TimeFilter
    alpha: float
    top_p: float
    pos: int

    # TCN
    num_channels: list
    graph_build_method: str
    graph_threshold: float
    graph_sample_size: int

    # SparseTSF
    model_type: str
    period_len: int

    # STALSTM
    SA_dim: int
    TA_dim: int

    # TimeKAN
    begin_order: bool

    # GCN
    conv_channel: int
    skip_channel: int
    gcn_depth: int
    node_dim: int
    propalpha: float

    # SOFTS
    d_core: int

    # FEDformer
    version: str
    mode_select: str
    modes: int

    # VRNN
    x_embed_dim: int
    z_embed_dim: int

    # Setting
    setting: str



