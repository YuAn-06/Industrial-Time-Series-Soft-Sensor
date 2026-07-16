# Model folder template

Copy this directory to `models/<ModelName>/`, then update:

1. `model_arch.py`: implement the architecture as a public `Model` class.
2. `model_spec.py`: set the model identity and benchmark capabilities.
3. `model_config.py`: declare model-specific parameters and defaults.
4. `__init__.py`: normally requires no changes.

The capability card describes stable framework integration, not experiment
hyperparameters. Dataset paths, dimensions, learning rates, epochs, and other
run-specific values remain in YAML and `utils/ExpConfigs.py`.

Use standard dataclass fields in `model_config.py`. For now, field metadata only
contains a `help` description; additional metadata should be added only when a
real configuration consumer needs it.

After the model registry supports folder cards, register the folder's
`MODEL_SPEC`, add a YAML for each genuinely supported task, and run the standard
CPU smoke test before launching benchmark experiments.
