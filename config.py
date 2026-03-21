from utils.fileutils import *

# Read configurations from JSON
config = read_json("./config/astr.json")
COCO_ANNO_PATH = config["coco_annotations_path"]
WEIGHTS_PATH = config["weights_path"]