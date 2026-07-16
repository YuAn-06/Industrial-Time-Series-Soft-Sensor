import argparse
from dataclasses import fields
import sys
from typing import get_args, get_origin
import yaml
import os
import warnings

from models.registry import canonical_model_name, get_model_package

def _format_setting_value(value):
    if isinstance(value, (list, tuple)):
        return "-".join(_format_setting_value(item) for item in value)
    if isinstance(value, float):
        text = f"{value:g}"
        return text.replace("-", "m").replace(".", "p")
    return str(value)

def build_setting(args):
    setting_parts = [args.data_name, args.model, args.task]
    setting_fields = [
        config_field
        for config_field in fields(args)
        if config_field.metadata.get("prefix")
    ]
    setting_fields.sort(key=lambda item: item.metadata.get("order", 1000))
    for config_field in setting_fields:
        prefix = config_field.metadata["prefix"]
        value = getattr(args, config_field.name)
        setting_parts.append(f"{prefix}{_format_setting_value(value)}")
    return "_".join(str(part) for part in setting_parts)

def read_yaml_params(yaml_path: str) -> dict:
    """Read and validate the top-level params mapping from one YAML file."""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as file:
        document = yaml.safe_load(file)

    if not isinstance(document, dict):
        raise ValueError("YAML root must be a mapping.")
    params = document.get("params")
    if not isinstance(params, dict):
        raise ValueError("YAML must contain a 'params' mapping.")
    return dict(params)


def _build_model_config(params: dict):
    """Build and validate the selected model-owned config."""
    if "model" not in params:
        raise ValueError("Configuration must define the 'model' field.")

    original_model_name = params["model"]
    model_name = canonical_model_name(original_model_name)
    if model_name != original_model_name:
        warnings.warn(
            f"Model name '{original_model_name}' is deprecated; use "
            f"'{model_name}'.",
            UserWarning,
            stacklevel=3,
        )

    params = dict(params)
    params["model"] = model_name
    model_package = get_model_package(model_name)
    config_class = model_package.MODEL_CONFIG
    model_spec = model_package.MODEL_SPEC
    task = params.get("task", config_class().task)
    if task not in model_spec.supported_tasks:
        raise ValueError(
            f"Model {model_name} does not support task {task}. Supported tasks: "
            f"{model_spec.supported_tasks}"
        )
    config_fields = {item.name for item in fields(config_class)}
    unknown_fields = set(params) - config_fields
    if unknown_fields:
        raise ValueError(
            f"Unknown configuration fields for {model_name}: "
            f"{sorted(unknown_fields)}"
        )

    declared_params = {
        name: value for name, value in params.items() if name in config_fields
    }
    config = config_class(**declared_params)
    config.validate()
    return config


def _cli_value_type(config_field, default):
    """Resolve an argparse scalar type from a dataclass field and default."""
    if isinstance(default, list):
        args = get_args(config_field.type)
        return args[0] if args and args[0] in (int, float, str) else str
    if config_field.type in (int, float, str):
        return config_field.type
    if default is not None and type(default) in (int, float, str):
        return type(default)
    return str


def _add_cli_argument(parser, config_field, default, hidden=False):
    """Add one dataclass field to a dynamic parser."""
    option = f"--{config_field.name}"
    help_text = argparse.SUPPRESS if hidden else config_field.metadata.get(
        "help", ""
    )

    if isinstance(default, bool):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            option,
            dest=config_field.name,
            action="store_true",
            default=argparse.SUPPRESS,
            help=help_text,
        )
        group.add_argument(
            f"--no_{config_field.name}",
            dest=config_field.name,
            action="store_false",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        return

    kwargs = {
        "dest": config_field.name,
        "default": argparse.SUPPRESS,
        "help": help_text,
    }
    if isinstance(default, list):
        kwargs["nargs"] = "+"
    kwargs["type"] = _cli_value_type(config_field, default)
    parser.add_argument(option, **kwargs)


def build_dynamic_cli_parser(model_name):
    """Build a CLI parser from one model's Config dataclass."""
    canonical_name = canonical_model_name(model_name)
    config_class = get_model_package(canonical_name).MODEL_CONFIG
    config_defaults = config_class()
    config_fields = {item.name: item for item in fields(config_class)}

    parser = argparse.ArgumentParser(
        description=f"Configuration for {canonical_name}",
        argument_default=argparse.SUPPRESS,
    )
    for name, config_field in config_fields.items():
        _add_cli_argument(parser, config_field, getattr(config_defaults, name))

    return parser


def parse_dynamic_cli(argv=None):
    """Parse CLI arguments after selecting the requested model Config."""
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--model", default="ARDNN")
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    model_name = canonical_model_name(bootstrap_args.model)

    parser = build_dynamic_cli_parser(model_name)
    params = vars(parser.parse_args(argv))
    params["model"] = model_name
    return _build_model_config(params)


def load_config(yaml_path: str = None, argv=None):
    """Load CLI or YAML parameters through the selected model Config."""
    if yaml_path is None:
        return parse_dynamic_cli(argv)

    params = read_yaml_params(yaml_path)
    return _build_model_config(params)


def prepare_run(args):
    """Build the experiment identity and create its output directory."""
    args.setting = build_setting(args)
    args.save_dir = os.path.join(".", "results", args.model, args.setting) + os.sep
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def Parse_arguments(yaml_path: str = None):
    """Backward-compatible config loader that also prepares the run folder."""
    args = load_config(yaml_path)
    return prepare_run(args)

