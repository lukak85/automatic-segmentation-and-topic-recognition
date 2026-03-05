import argparse
from utils.displayutils import *
from pycocotools.coco import COCO

import os
import json

from utils.fileutils import save_coco_to_json

IMAGES_ROOT = "./annotation/pawls/labels/images/"


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
        "-o",
        "--output-path",
        help="Path to the output file/folder",
        type=str,
    )
    parser.add_argument(
        "-p", "--path", help="Path to the input file/folder", type=str, required=False
    )
    parser.add_argument(
        "-r",
        "--remove-duplicates",
        help="Whether to remove duplicate annotations when joining, by comparing two annotations as duplicates if they"
        "have the same category and IoU > 0.95",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--save-visualization",
        help="Where to save the visualization of the annotations. If not provided, it will be displayed but not saved",
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
    elif args.remove_duplicates:
        from pycocotools import mask as maskUtils

        if not args.annotations_file:
            print("Please provide an annotation file to remove duplicates from")
            exit(1)

        coco = COCO(args.annotations_file)
        coco_anns = coco.loadAnns(coco.getAnnIds())
        coco_cats = coco.cats

        # Create a map with image_id as key and list of annotations as value
        image_id_to_anns = {}
        for ann in coco_anns:
            image_id = ann["image_id"]
            if image_id not in image_id_to_anns:
                image_id_to_anns[image_id] = []
            image_id_to_anns[image_id].append(ann)

        # For each image, remove duplicate annotations
        unique_anns_all = []
        for image_id in image_id_to_anns:
            anns = image_id_to_anns[image_id]
            unique_anns = []
            for ann in anns:
                found_duplicate = False
                if not unique_anns:
                    unique_anns.append(ann)
                else:
                    for annCompare in unique_anns:
                        if ann == annCompare:
                            continue

                        bbox1 = [
                            ann["bbox"][0],
                            ann["bbox"][1],
                            ann["bbox"][2],
                            ann["bbox"][3],
                        ]
                        bbox2 = [
                            annCompare["bbox"][0],
                            annCompare["bbox"][1],
                            annCompare["bbox"][2],
                            annCompare["bbox"][3],
                        ]

                        ious = maskUtils.iou([bbox1], [bbox2], [False])
                        if ious[0][0] > 0.95:
                            if ann["score"] > annCompare["score"]:
                                unique_anns.remove(annCompare)
                                unique_anns.append(ann)
                            found_duplicate = True
                            break
                if ann not in unique_anns and not found_duplicate:
                    unique_anns.append(ann)
            unique_anns_all.extend(unique_anns)

        coco_full = {
            "images": [],
            "annotations": unique_anns_all,
            "categories": [],
        }

        for image_id in coco.imgs:
            coco_full["images"].append(coco.imgs[image_id])

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

        img_path = coco.loadImgs(coco.getImgIds([int(args.image_id)]))[0]["file_name"]
        img_path = os.path.join(IMAGES_ROOT, img_path)
        anns = coco.loadAnns(coco.getAnnIds([int(args.image_id)]))
        coco_anns = load_coco_annotations(anns, categories=coco.cats)
        display_img = cv2.imread(img_path)

        # Convert all anns to names

        draw_layout(
            display_img, coco_anns, save_path=args.save_visualization
        )  # Drawing box for detection
