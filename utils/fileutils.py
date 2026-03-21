"""File I/O utilities for configuration and COCO annotation files."""

import json
from pathlib import Path


def read_config(config_path):
    """Load a JSON configuration file.

    Args:
        config_path: Path to the JSON file, or None.

    Returns:
        Parsed config dict, or None if config_path is None.
    """
    if config_path is None:
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def save_coco_to_json(coco_data, output_path):
    """Save COCO-format annotation data to a JSON file.

    Creates parent directories if they don't exist.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=4)
