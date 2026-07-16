"""Convenient code-edited YAML entry point for a full train/test run."""

from runner import train_test_from_yaml


if __name__ == "__main__":
    yaml_name = "SS_task/DC_scripts/yaml/MSACNN.yaml"
    yaml_path = f"./scripts/{yaml_name}"

    print(f"==== using {yaml_name} ====")
    train_test_from_yaml(yaml_path)
