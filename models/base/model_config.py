"""Common experiment configuration inherited by every model."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass
class BaseExpConfig:
    """Data, training, runtime, and evaluation parameters shared by models."""


    # General parameters
    model: str = field(default='ARDNN', metadata={'help': 'Model name to use'})
    task: str = field(default='short_term_forecasting', metadata={'help': "Task type: ['short_term_forecasting', 'soft_sensor']"})
    data_name: str = field(default='DC', metadata={'help': "Dataset name: [ 'DEB', 'SRU']"})
    data_path: str = field(default='', metadata={'help': 'Dataset path'})
    data_aug: bool = field(default=False, metadata={'help': 'If use data augmentation for Deb and SRU dataset'})
    target: str = field(default='', metadata={'help': 'Target variable name'})
    use_amp: bool = field(default=False, metadata={'help': 'If use automatic mixed precision training'})
    num_workers: int = field(default=1, metadata={'help': 'DataLoader the number of workers'})
    use_condition_label: bool = field(default=False, metadata={'help': 'If use mode variable for multi_mode dataset'})
    use_tensorboard: bool = field(default=False, metadata={'help': 'If use tensorboard for visualization'})
   
    # Model parameters
    enc_in: int = field(default=16, metadata={'help': 'Dimension of encoder Input'})
    dec_in: int = field(default=16, metadata={'help': 'Dimension of decoder Input'})
    C_in: int = field(default=16, metadata={'help': 'Dimension of Channel Input'})
    C_out: int = field(default=4, metadata={'help': 'Dimension of Channel Output'})
    seq_len: int = field(default=32, metadata={'help': 'Sequence length', 'prefix': 'sl', 'order': 0})
    label_len: int = field(default=10, metadata={'help': 'Label length', 'prefix': 'll', 'order': 1})
    pred_len: int = field(default=6, metadata={'help': 'Prediction Length,re', 'prefix': 'pl', 'order': 2})
    embed: str = field(default='TimeF', metadata={'help': 'Embedding type'})
    freq: str = field(default='s', metadata={'help': 'Time embedding frequency'})
    collate_fn: str = field(default='collate_fn', metadata={'help': 'Collate function to use'})

    # Training parameters
    batch_size: int = field(default=64, metadata={'help': 'Batch size for training', 'prefix': 'bt', 'order': 3})
    learning_rate: float = field(default=0.001, metadata={'help': 'Learning rate for optimizer', 'prefix': 'lr', 'order': 4})
    epoch: int = field(default=200, metadata={'help': 'Number of training epochs', 'prefix': 'ep', 'order': 5})
    patience: int = field(default=10, metadata={'help': 'Patience for early stopping', 'prefix': 'pat', 'order': 6})
    lradj: str = field(default='cosine', metadata={'help': "Learning rate adjustment strategy: ['type1', 'type2', 'cosine']"})
    weight_decay: float = field(default=0.0, metadata={'help': 'L2 regularization weight'})
    model_stage: str = field(default='finetune', metadata={'help': 'Model stage for models with pretraining'})
    pretrained_ckpt: str = field(default='', metadata={'help': 'Checkpoint path for loading pretrained weights'})
    pretrain_epoch: int = field(default=-1, metadata={'help': 'Optional epoch override for pretraining'})
    finetune_epoch: int = field(default=-1, metadata={'help': 'Optional epoch override for finetuning'})
    pretrain_learning_rate: float = field(default=-1.0, metadata={'help': 'Optional learning-rate override for pretraining'})
    finetune_learning_rate: float = field(default=-1.0, metadata={'help': 'Optional learning-rate override for finetuning'})
    
    # Test parameters
    save_dir: str = field(default='logs', metadata={'help': 'Directory to save logs and models'})
    inverse: bool = field(default=False, metadata={'help': 'If the data is scaled, inverse the data to the original scale'})
    use_cuda: bool = field(default=False, metadata={'help': 'Use CUDA for training'})
    device: str = field(default='cuda', metadata={'help': 'Device to use'})
    gpu: int = field(default=0, metadata={'help': 'GPU ID to use'})
    seed: int = field(default=2021, metadata={'help': 'Random seed'})
    device_ids: list[int] = field(default_factory=lambda: [0], metadata={'help': 'List of GPU device IDs'})
    use_multi_gpu: bool = field(default=False, metadata={'help': 'Use multiple GPUs'})
    setting: str = field(default='', metadata={'help': 'Setting to use'})

    @classmethod
    def field_names(cls) -> set[str]:
        return {config_field.name for config_field in fields(cls)}

    @classmethod
    def help_messages(cls) -> dict[str, str]:
        return {config_field.name: config_field.metadata.get("help", "") for config_field in fields(cls)}

    @classmethod
    def from_params(cls, params: dict[str, Any]):
        unknown = set(params) - cls.field_names()
        if unknown:
            raise ValueError(f"Unknown configuration fields for {cls.__name__}: {sorted(unknown)}")
        config = cls(**params)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.task == "short_term_forecasting" and self.pred_len <= 0:
            raise ValueError("pred_len must be positive for forecasting.")
        if self.batch_size <= 0 or self.epoch <= 0:
            raise ValueError("batch_size and epoch must be positive.")
