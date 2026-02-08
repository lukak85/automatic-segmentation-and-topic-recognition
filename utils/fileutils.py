def read_config(config_path):
    import json

    config = None
    if config_path is not None:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

    return config


def save_coco_to_json(coco_data, output_path):
    import json

    with open(output_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)
