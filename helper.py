import argparse
import json
import os

import cv2
from pycocotools.coco import COCO

from utils.displayutils import *
from utils.fileutils import save_coco_to_json, read_json

IMAGES_ROOT = "./dataset/images/"
STATUS_JSON = "annotation/pawls/skiff_files/apps/pawls/papers/status/development_user@example.com.json"

# IoU threshold above which two annotations are considered duplicates
DUPLICATE_IOU_THRESHOLD = 0.95


# ==============================================================================
# Annotation helpers
# ==============================================================================


def load_coco_annotations(annotations, categories=None):
    """Convert COCO annotation dicts to a layoutparser Layout.

    Args:
        annotations: List of COCO annotation dicts.
        categories: Optional COCO categories dict. If provided, resolves
                    category IDs to human-readable names.
    """
    layout = lp.Layout()

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=(
                    categories[ann["category_id"]]["name"]
                    if categories
                    else ann["category_id"]
                ),
                id=ann["id"],
                score=ann.get("score", 1),
            )
        )

    return layout


def join_annotations(path):
    """Read all JSON annotation files in a folder and merge them.

    Reassigns annotation IDs sequentially to avoid conflicts across files.
    """
    coco_anns_list = []
    coco_imgs_list = []
    coco_cats = None
    annotation_id = 1

    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue

        coco = COCO(os.path.join(path, filename))
        coco_anns = coco.loadAnns(coco.getAnnIds())

        # Reassign annotation IDs to avoid collisions
        for ann in coco_anns:
            ann["id"] = annotation_id
            annotation_id += 1

        coco_anns_list.extend(coco_anns)
        coco_imgs_list.extend(coco.loadImgs(coco.getImgIds()))

        if coco_cats is None:
            coco_cats = coco.cats

    return {
        "images": coco_imgs_list,
        "annotations": coco_anns_list,
        "categories": [coco_cats[cid] for cid in coco_cats],
    }


def remove_duplicates(coco, annotations_file):
    """Remove near-duplicate annotations based on IoU overlap.

    Two annotations on the same image with IoU > DUPLICATE_IOU_THRESHOLD are
    considered duplicates. When both have scores, the higher-scoring one is kept.

    Args:
        coco: A loaded COCO object.
        annotations_file: Path to the annotation file (used for validation).
    """
    from pycocotools import mask as maskUtils

    if not annotations_file:
        print("Please provide an annotation file to remove duplicates from.")
        exit(1)

    coco_anns = coco.loadAnns(coco.getAnnIds())

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_anns:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # For each image, deduplicate by IoU
    unique_anns = []
    for image_id, anns in anns_by_image.items():
        kept = []
        for ann in anns:
            is_duplicate = False
            for existing in kept:
                iou = maskUtils.iou([ann["bbox"]], [existing["bbox"]], [False])
                if iou[0][0] > DUPLICATE_IOU_THRESHOLD:
                    # Keep the annotation with the higher score
                    if ann.get("score") is not None and ann["score"] > existing.get("score", 0):
                        kept.remove(existing)
                        kept.append(ann)
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(ann)
        unique_anns.extend(kept)

    return {
        "images": [coco.imgs[img_id] for img_id in coco.imgs],
        "annotations": unique_anns,
        "categories": [coco.cats[cid] for cid in coco.cats],
    }


def visualize_annotations(coco, image_id, save_path=None):
    """Load and display annotations for a single image.

    Args:
        coco: A loaded COCO object.
        image_id: The ID of the image to visualize.
        save_path: Optional path to save the visualization.
    """
    img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
    img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
    anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
    layout = load_coco_annotations(anns, categories=coco.cats)
    display_img = cv2.imread(img_path)
    draw_layout(display_img, layout, save_path=save_path)


def visualize_all_images(coco, save_path=None, skip_hashes=None):
    """Visualize annotations for all images, optionally skipping some.

    Args:
        coco: A loaded COCO object.
        save_path: Optional path to save the visualizations.
        skip_hashes: Set of document hashes to skip.
    """
    for image_id in coco.imgs:
        img_info = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]
        doc_hash = img_info["file_name"].split("_")[0]

        if skip_hashes and doc_hash in skip_hashes:
            continue

        print(f"Processing image {img_info['file_name']} with id {image_id}")
        img_path = os.path.join(IMAGES_ROOT, img_info["file_name"])
        anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
        layout = load_coco_annotations(anns, categories=coco.cats)
        display_img = cv2.imread(img_path)
        draw_layout(display_img, layout, save_path=save_path)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document layout analysis helper")

    parser.add_argument(
        "-a", "--annotations-file",
        help="Path to the COCO annotation file",
        type=str,
    )
    parser.add_argument(
        "-i", "--image-id",
        help="Image ID to visualize",
        type=str,
    )
    parser.add_argument(
        "-m", "--mode",
        help="Action: join-annotations, prepare-annotations, order-images, "
             "remove-scores, review-annotations, or visualize (default)",
        type=str,
        default="visualize",
    )
    parser.add_argument(
        "-o", "--output-path",
        help="Output file path",
        type=str,
    )
    parser.add_argument(
        "-p", "--path",
        help="Input folder path (for join-annotations)",
        type=str,
    )
    parser.add_argument(
        "-r", "--remove-duplicates",
        help="Remove duplicate annotations (IoU > 0.95) when joining",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save-visualization",
        help="Path to save the annotation visualization",
        type=str,
    )

    args = parser.parse_args()

    # ---- Mode dispatch ----

    if args.mode == "prepare-annotations":
        if not args.annotations_file:
            print("Please provide an annotation file to prepare.")
            exit(1)

        # Join all annotations in the folder
        coco_data = join_annotations(args.annotations_file)
        output_json = os.path.join(args.annotations_file, args.output_path)
        save_coco_to_json(coco_data, output_json)

        # Remove duplicates
        coco = COCO(output_json)
        coco_data = remove_duplicates(coco, args.annotations_file)
        save_coco_to_json(coco_data, output_json)

        # Visualize for review
        coco = COCO(output_json)
        visualize_all_images(coco, save_path=args.save_visualization)

    elif args.mode == "order-images":
        if not args.annotations_file:
            print("Please provide an annotation file.")
            exit(1)

        coco = COCO(args.annotations_file)
        sorted_images = sorted(coco.dataset["images"], key=lambda x: x["id"])

        # Filter to finished documents only
        with open(STATUS_JSON, "r") as f:
            status_data = json.load(f)
        finished_images = [
            img for img in sorted_images
            if status_data[img["file_name"].split("_")[0]]["finished"]
        ]

        sorted_annotations = sorted(
            coco.dataset["annotations"], key=lambda x: x["image_id"]
        )
        save_coco_to_json(
            {
                "images": finished_images,
                "annotations": sorted_annotations,
                "categories": coco.dataset["categories"],
            },
            args.output_path,
        )

    elif args.mode == "remove-scores":
        coco = COCO(args.annotations_file)
        for ann in coco.dataset["annotations"]:
            ann.pop("score", None)
        save_coco_to_json(
            {
                "images": coco.dataset["images"],
                "annotations": coco.dataset["annotations"],
                "categories": coco.dataset["categories"],
            },
            args.output_path,
        )

    elif args.mode == "review-annotations":
        # Documents already reviewed — skip these when reviewing
        already_checked = {
            "00de9bb518f39464b6b5bb7254d6fdd6e2e2e1fa46710ffe84a6863dca4be950",
            "0166d9b3f20fa5a4f6bd9d6d001f8b81b24665a6368dd0c10ed3d8a9e30dd691",
            "04bb9872050b5a73939ae9734a7a1f6935df7b6623f03dc407f3403d52392aa6",
            "04bc67afae7e1c9113cbbd83e98df59f252ba7757ad90d2c8856f227e5cd8beb",
            "0525ec05617fc357460ca247faa0b0be9b2caedc8b2663680f852b93541831b6",
            "111eba9400e08e9e0a5a257aae5c3d36c3c63dd383005a3ca65cbb4d884d8346",
            "1c3968e8cc47ae26ed907f561ccd55dedbbad3c6645f289fe964582ba864bddd",
            "20ecf1d1b0602973c2449ed90428bc31847ab613749ffc5d7ce92c5e05788f27",
            "230edb119aff067fecd3586eb3ce857f9ce402b0867037c156efaaaa32d0ba4b",
            "2a6e4009dac571c6d4e8b58009acd58a0c0ea1d859f21ac518cf82f2d52a5eda",
            "326f6533357ab6e301abf9731667626678ccfa078497c866e12df4ff1f652e8f",
            "4289acbeebf1a459a5339c0f3ed89268ae9437541e5fcce8cd3fa1862517e19a",
            "53473f43fd47f257cab19acbf24ef1b1b7abe75b4cd643a2387cb10c6c4c44ea",
            "7612369f2c0ac02697feb81598cf9069a94ea21637329c59bf3955ab731860c2",
            "7901803e4e1f43b71379ab2657057fc8545977dc4b5f6cbda225c965c4d1c849",
            "7c43f3e9c7b8ef76798616f47f26cb7e514b7d7216e2e934e366c5eb7266339d",
            "7f2ee648660870a37590aafa87d1d5636bdddea816f5c386770961f6724fb495",
            "8b73208759cd38d30f92e167303c95774902d0554e704b7f64bcbde96ec0d00e",
            "8ec2c4adff08b0297d741164d97068f8f561c18923f28a042382e742be45996c",
            "9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21",
            "9d4eceb46db57f78273b82f5c7e2139b1386b264a1b18703f3d247b5310886dc",
            "ac30fbcf6678b2b5d3f278a37fb3785adcc1a0791cac4328acd7a86cada649ad",
            "ba0f4987395c485a886948b2d4d527e7a0cf6382feb245bde7ef39ac8cae0435",
            "c95ad1a22c65a26798da6407ccf29373b4ff999b0b4d4d4828f803bff7405529",
            "cd0c26aa8cad0a2c40e96abccad393a2f9a55742c651724081168c2425acd7a2",
            "d393c9ee0d6653bafac4c34990cffbc414f57ee1ae11a01669b0ae0b8fcdb97f",
            "ef01d9a74ff40330527608d5ff5434c22664a7a3f639949b281646ad6bfd28f5",
            "f5753bdada7c6202759859b13c320ce9830aea66fcd49e63721d2b3dca0c45bb",
        }

        coco = COCO(args.annotations_file)
        visualize_all_images(
            coco, save_path=args.save_visualization, skip_hashes=already_checked
        )

    elif args.mode == "join-annotations":
        if not args.path:
            print("Please provide a folder path to join annotations.")
            exit(1)
        coco_data = join_annotations(args.path)
        save_coco_to_json(coco_data, args.output_path)

    elif args.mode == "assign-ids":
        if not args.path:
            print("Please provide a folder path to join annotations.")
            exit(1)
        coco_data = read_json(args.path)
        for anno_id, annotation in enumerate(coco_data["annotations"]):
            annotation["id"] = anno_id
            annotation["segmentation"] = None
        save_coco_to_json(coco_data, args.output_path)

    elif args.remove_duplicates:
        coco = COCO(args.annotations_file)
        coco_data = remove_duplicates(coco, args.annotations_file)
        save_coco_to_json(coco_data, args.output_path)

    else:
        # Default: visualize a single image's annotations
        if not args.annotations_file:
            print("Please provide an annotation file to visualize.")
            exit(1)
        if not args.image_id:
            print("Please provide an image ID to visualize.")
            exit(1)

        coco = COCO(args.annotations_file)
        visualize_annotations(coco, args.image_id, save_path=args.save_visualization)
