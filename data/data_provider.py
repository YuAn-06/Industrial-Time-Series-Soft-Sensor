"""Dataset and DataLoader selection for InduTS-SS experiments.

Dataset classes are selected from two independent properties: the prediction
task and the input representation required by the model. Dataset names identify
the source data only; they are not combined into synthetic registry keys such
as ``DC_MultiMode_Soft_Sensor``.
"""

from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from data import (
    Dataset_Custom,
    Dataset_Custom_4_Soft_Sensor,
    Dataset_LaggedMatrix_4_Soft_Sensor,
    Dataset_MultiMode,
    Dataset_MultiMode_4_Soft_Sensor,
)
from models.registry import get_model_spec


DATASET_ALIASES = {
    "PPGAS2012": "PPGAS",
}

SUPPORTED_DATASETS = frozenset({"DC", "SRU", "PPGAS", "Ironmaking", "MP"})

# The concrete Dataset implementation depends on the task and representation,
# not on the physical dataset name. DC, SRU, PPGAS, Ironmaking, and MP all use
# these shared implementations with their own args.data_path and args.data_name.
DATASET_CLASS_REGISTRY = {
    ("short_term_forecasting", "standard"): Dataset_Custom,
    ("soft_sensor", "standard"): Dataset_Custom_4_Soft_Sensor,
    ("short_term_forecasting", "multimode"): Dataset_MultiMode,
    ("soft_sensor", "multimode"): Dataset_MultiMode_4_Soft_Sensor,
    ("soft_sensor", "lagged_matrix"): Dataset_LaggedMatrix_4_Soft_Sensor,
}


def normalize_data_name(data_name: str) -> str:
    """Return the canonical dataset name for known aliases."""
    return DATASET_ALIASES.get(data_name, data_name)


def resolve_data_representation(args) -> str:
    """Resolve the input representation requested by the config and model."""
    if args.use_condition_label:
        return "multimode"
    return get_model_spec(args.model).dataset_type


def resolve_dataset_class(args):
    """Select a Dataset class from task and representation declarations."""
    data_name = normalize_data_name(args.data_name)
    if data_name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Unknown dataset '{args.data_name}'. Supported datasets: {supported}."
        )

    representation = resolve_data_representation(args)
    registry_key = (args.task, representation)
    try:
        return DATASET_CLASS_REGISTRY[registry_key]
    except KeyError as exc:
        raise ValueError(
            "No dataset implementation for "
            f"task='{args.task}', representation='{representation}'."
        ) from exc


def data_provider(args, flag: str) -> Tuple[Dataset, DataLoader]:
    """Build one dataset split and its DataLoader without changing split policy."""
    Data = resolve_dataset_class(args)

    timeenc = 0 if args.embed != "timeF" else 1
    if flag in ["valid", "test"]:
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = False
        drop_last = True

    dataset = Data(args, flag, timeenc=timeenc)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )
    return dataset, data_loader
