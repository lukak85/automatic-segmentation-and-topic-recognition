def read_json(path):
    import json

    j = None
    if path is not None:
        with open(path, "r") as file:
            j = json.load(file)

    return j


def read_config(config_path):
    return read_json(config_path)


def save_coco_to_json(coco_data, output_path):
    import json

    # Ensure the output directory exists
    from pathlib import Path

    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)
