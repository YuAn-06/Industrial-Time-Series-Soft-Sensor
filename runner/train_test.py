"""Standard training and testing workflow.

Use ``train_test_from_yaml`` to start directly from YAML, or ``train_test``
when prepared args are already available. Concrete Experiment classes in
``exp`` still implement training and testing; this module only orchestrates
their order, logging, and resource cleanup.
"""

from .builder import build_args_from_yaml, build_experiment, build_logger


def train_test(args, do_train=True, do_test=True):
    """Run training and/or testing with an already prepared config.

    ``args`` must already contain valid ``setting`` and ``save_dir`` values.
    The created Experiment is returned for optional post-run inspection.
    """
    if not do_train and not do_test:
        raise ValueError("At least one of do_train or do_test must be True.")

    logger = build_logger(args)
    try:
        logger.info(f"Setting: {args.setting}")
        exp = build_experiment(args)

        if do_train:
            logger.info("Start training...")
            exp.train(logger)
        if do_test:
            logger.info("Start testing...")
            exp.test(logger)
        return exp
    finally:
        logger.remove_handles()


def train_test_from_yaml(
    yaml_path,
    overrides=None,
    do_train=True,
    do_test=True,
):
    """Start a standard experiment from one YAML file.

    ``overrides`` can temporarily replace YAML values during debugging, for
    example ``{"epoch": 1}``. Overrides are applied before building the
    setting, keeping the result directory consistent with the actual config.
    """
    args = build_args_from_yaml(yaml_path, overrides=overrides)
    return train_test(args, do_train=do_train, do_test=do_test)
