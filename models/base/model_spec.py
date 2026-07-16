"""Common schema for an InduTS-SS model's benchmark capability card."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Describe how one model integrates with the benchmark framework.

    Architecture hyperparameters do not belong here. They remain in the YAML
    and model Config dataclass. This card only records stable integration capabilities
    used by registries, data providers, experiments, losses, and runners.
    """

    # Public model identity and import location.
    name: str
    module: str
    supported_tasks: tuple[str, ...]

    # Framework integration required by this model.
    dataset_type: str = "standard"
    loss_type: str = "default"

    # Empty means the model has no pretraining stage.
    pretrain_stages: tuple[str, ...] = ()

    # Optional documentation metadata; runtime behavior must not depend on it.
    paper_title: str = ""
    paper_url: str = ""
    source_url: str = ""
