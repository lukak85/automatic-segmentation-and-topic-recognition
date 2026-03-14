import argparse
from utils.displayutils import *
from pycocotools.coco import COCO

import os
import json

from utils.fileutils import save_coco_to_json, read_json

IMAGES_ROOT = "./dataset/images/"
STATUS_JSON = "annotation/pawls/skiff_files/apps/pawls/papers/status/development_user@example.com.json"


def load_coco_annotations(annotations, categories=None):
    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele["bbox"]

        # if ele["score"] is None:
        #    continue

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


def join_annotations(path):
    # Read all annotations and join them into a single file
    coco_anns_list = []
    coco_imgs_list = []
    coco_cats = None

    annotation_id = 1
    for file in os.listdir(path):
        if file.endswith(".json"):
            coco = COCO(os.path.join(path, file))
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

    return coco_full


def remove_duplicates(coco):
    from pycocotools import mask as maskUtils

    if not args.annotations_file:
        print("Please provide an annotation file to remove duplicates from")
        exit(1)

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
                        if (
                            ann.get("score") is not None
                            and ann["score"] > annCompare["score"]
                        ):
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

    return coco_full


def map_ids_between_datasets(
    coco,
    from_dataset=None,
    to_dataset=None,
):
    # This function is a placeholder for mapping category ids between two datasets if they have different category ids
    # for the same categories
    # For example, if dataset A has category id 1 for "Paragraph" and dataset B has category id 5 for "Paragraph", we
    # can use this function to map category id 1 to 5 when joining annotations from both datasets
    return coco_full


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
        "have IoU > 0.95",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--save-visualization",
        help="Where to save the visualization of the annotations. If not provided, it will be displayed but not saved",
        type=str,
    )

    args = parser.parse_args()

    if args.mode == "prepare-annotations":
        if not args.annotations_file:
            print("Please provide an annotation file to prepare")
            exit(1)

        coco = join_annotations(args.annotations_file)
        output_json = str(os.path.join(args.annotations_file, args.output_path))
        save_coco_to_json(coco, output_json)

        coco = COCO(output_json)
        coco = remove_duplicates(coco)
        save_coco_to_json(coco, output_json)

        coco = COCO(output_json)
        # Check and fix annotations
        for image_id in coco.imgs:
            img_path = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]["file_name"]
            print(f"Processing image {img_path} with id {image_id}")
            img_path = os.path.join(IMAGES_ROOT, img_path)
            anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
            coco_anns = load_coco_annotations(anns, categories=coco.cats)
            display_img = cv2.imread(img_path)

            draw_layout(
                display_img, coco_anns, save_path=args.save_visualization
            )  # Drawing box for detection
    elif args.mode == "order-images":
        if not args.annotations_file:
            print("Please provide an annotation file to prepare")
            exit(1)
        coco = COCO(args.annotations_file)

        sorted_images = sorted(coco.dataset["images"], key=lambda x: x["id"])
        with open(STATUS_JSON, "r") as file:
            status_data = json.load(file)
        filtered_images = unfinished_images = [
            img
            for img in sorted_images
            if status_data[img["file_name"].split("_")[0]]["finished"]
        ]

        sorted_annotations = sorted(
            coco.dataset["annotations"], key=lambda x: x["image_id"]
        )

        coco_full = {
            "images": sorted_images,
            "annotations": sorted_annotations,
            "categories": coco.dataset["categories"],
        }

        save_coco_to_json(coco_full, args.output_path)
    elif args.mode == "remove-scores":
        coco = COCO(args.annotations_file)

        for d in coco.dataset["annotations"]:
            d.pop("score", None)

        coco_full = {
            "images": coco.dataset["images"],
            "annotations": coco.dataset["annotations"],
            "categories": coco.dataset["categories"],
        }

        save_coco_to_json(coco_full, args.output_path)
    elif args.mode == "review-annotations":
        # Just for me at the moment
        already_checked = [
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
            "326f6533357ab6e301abf9731667626678ccfa078497c866e12df4ff1f652e8f",  # TODO: join the annotations
            "4289acbeebf1a459a5339c0f3ed89268ae9437541e5fcce8cd3fa1862517e19a",  # TODO: join the annotations
            "53473f43fd47f257cab19acbf24ef1b1b7abe75b4cd643a2387cb10c6c4c44ea",  # TODO: join the annotations
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
            # "f5753bdada7c6202759859b13c320ce9830aea66fcd49e63721d2b3dca0c45bb",
        ]

        coco = COCO(args.annotations_file)
        # Check and fix annotations
        for image_id in coco.imgs:
            img_path = coco.loadImgs(coco.getImgIds([int(image_id)]))[0]["file_name"]
            if img_path.split("_")[0] in already_checked:
                continue
            print(f"Processing image {img_path} with id {image_id}")
            img_path = os.path.join(IMAGES_ROOT, img_path)
            anns = coco.loadAnns(coco.getAnnIds([int(image_id)]))
            coco_anns = load_coco_annotations(anns, categories=coco.cats)
            display_img = cv2.imread(img_path)

            draw_layout(
                display_img, coco_anns, save_path=args.save_visualization
            )  # Drawing box for detection
    elif args.mode == "join-annotations":
        if not args.path:
            print("Please provide a folder path to join annotations")
            exit(1)

        coco = join_annotations(args.path)

        save_coco_to_json(coco, args.output_path)
    elif args.mode == "assign-ids":
        if not args.path:
            print("Please provide a folder path to annotations")
            exit(1)
        json_to_assign_ids = read_json(args.path)

        anno_id = 0
        for annotation in json_to_assign_ids["annotations"]:
            annotation["id"] = anno_id
            anno_id += 1
            annotation["segmentation"] = None

        save_coco_to_json(json_to_assign_ids, args.output_path)
    elif args.remove_duplicates:
        coco = COCO(args.annotations_file)
        coco_full = remove_duplicates(coco)

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
