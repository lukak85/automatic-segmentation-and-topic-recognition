import argparse
from utils.displayutils import *
from pycocotools.coco import COCO

import os
import json

from utils.fileutils import save_coco_to_json


def load_coco_annotations(annotations, categories=None):
    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele["bbox"]

        if ele["score"] is None:
            continue

        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=(
                    categories.get(ele["category_id"])["name"]
                    if categories
                    else ele["category_id"]
                ),
                id=ele["id"],
                score=ele["score"] if "score" in ele else 1,
            )
        )

    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Document layout analysis helper")

    parser.add_argument(
        "-a",
        "--annotations-file",
        help="Path to the annotation",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--image-id",
        help="Image id of the annotation to visualize",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="What kind of action we want to perform. Currently supported are: 'join-annotations'",
        type=str,
        default="page",
    )
    parser.add_argument(
        "-p", "--path", help="Path to the input file/folder", type=str, required=True
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to the output file/folder",
        type=str,
    )

    args = parser.parse_args()

    if args.mode == "join-annotations":
        if not args.path:
            print("Please provide a folder path to join annotations")
            exit(1)

        # Read all annotations and join them into a single file
        coco_anns_list = []
        coco_imgs_list = []
        coco_cats = None

        annotation_id = 1
        for file in os.listdir(args.path):
            if file.endswith(".json"):
                coco = COCO(os.path.join(args.path, file))
                coco_anns = coco.loadAnns(coco.getAnnIds())
                # Seat each annotation id anew to avoid conflicts
                for ann in coco_anns:
                    ann["id"] = annotation_id
                    annotation_id += 1
                coco_imgs = coco.loadImgs(coco.getImgIds())
                coco_anns_list.extend(coco_anns)
                coco_imgs_list.extend(coco_imgs)
                if not coco_cats:
                    coco_cats = coco.cats

        coco_full = {
            "images": coco_imgs_list,
            "annotations": coco_anns_list,
            "categories": [],
        }

        for category_id in coco_cats:
            coco_full["categories"].append(coco_cats[category_id])

        save_coco_to_json(coco_full, args.output_path)
    else:
        if not args.annotations_file:
            print("Please provide an annotation file to visualize")
            exit(1)
        if not args.image_id:
            print("Please provide an image id to visualize")
            exit(1)

        print(args.annotations_file)
        coco = COCO(args.annotations_file)

        coco_anns = load_coco_annotations(
            coco.loadAnns(coco.getAnnIds([int(args.image_id)])), categories=coco.cats
        )
        display_img = cv2.imread(args.path)

        # Convert all anns to names

        draw_layout(display_img, coco_anns)  # Drawing box for detection
