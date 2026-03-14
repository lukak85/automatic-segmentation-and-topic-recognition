import argparse
import os
from contextlib import redirect_stdout

from pycocotools.coco import COCO

from utils.conversionutils import *
from utils.displayutils import *
from utils.evalutils import *
from utils.fileutils import *

# ======================================================================================================================
# Parameters
# ======================================================================================================================

COCO_ANNO_PATH = "dataset/annotations.json"
WEIGHTS_PATH = "/"


# ======================================================================================================================
# Main function
# ======================================================================================================================


def main(
    img,
    image_info,
    ground_truth,
    model,
    evaluation_metic,
    categories,
    visualization=False,
    display_ground=False,
    display_img=None,
    save_coco=None,
    save_image_path=None,
):
    # Preform layout analysis
    if model is not None:  # TODO: do check if it's correct class
        layout = model.detect(img)
    else:
        raise Exception("No DLA model is provided.")

    # Visualization
    if visualization:
        draw_layout(display_img, layout)  # Drawing box for detection
        if ground_truth is not None and display_ground:
            draw_layout(display_img, ground_truth)  # Drawing box for ground truth

    # Evaluation
    if evaluation_metic is not None:  # TODO: add back later
        if evaluation_metic == "f1":
            print(f1_score(layout, ground_truth))
        else:  # Default evaluation metric
            print(mean_average_precision(layout, ground_truth))

    if save_coco is not None:
        # Check if filename ends with .json, else throw an error
        if not save_coco.endswith(".json"):
            raise Exception("The save path for COCO annotations must end with .json")

        coco_annotations = layout_parser_to_coco(layout, image_info, categories),

        save_coco_to_json(
            coco_annotations, save_coco,
        )

    if save_image_path is not None:
        draw_layout(
            display_img, layout, save_path=save_image_path
        )  # Drawing box for detection


# ======================================================================================================================
# Helper functions
# ======================================================================================================================


def read_picture(path, to_rgb=True):
    img = cv2.imread(path)
    if to_rgb:
        img = img[..., ::-1]
    return img


def load_coco_annotations(annotations, coco=None):
    """
    Args:
        annotations (List):
            a list of coco annotaions for the current image
        coco (`optional`, defaults to `False`):
            COCO annotation object instance. If set, this function will
            convert the loaded annotation category ids to category names
            set in COCO.categories
    """
    layout = lp.Layout()

    for ele in annotations:

        x, y, w, h = ele["bbox"]

        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w + x, h + y),
                type=(
                    ele["category_id"]
                    if coco is None
                    else coco.cats[ele["category_id"]]["name"]
                ),
                id=ele["id"],
            )
        )

    return layout


def check_connection(address="localhost", port=8080, timeout=5):
    import socket

    try:
        # connect to the host -- tells us if the host is actually reachable
        socket.create_connection((address, port), timeout=timeout)
        return True
    except OSError:
        pass
    return False


# ======================================================================================================================
# Run
# ======================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Document layout analysis pipeline")
    parser.add_argument(
        "-c",
        "--config",
        help="JSON configuration file path for the model",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-dd",
        "--display-detection",
        help="Display image with detected bounding boxes if mode is single image",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-dg",
        "--display-ground",
        help="Display ground",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-dm",
        "--dla-method",
        help="Method for Document Layout Analysis",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--evaluation_metric",
        help="Evaluation metric for Document Layout Analysis",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-f", "--file", help="Path to the input file", type=str, required=True
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Detecting single image, whole PDF, or the whole corpus",
        type=str,
        default="page",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save the COCO annotations for the detected layout in a JSON file.",
        type=str,
    )
    parser.add_argument(
        "-si",
        "--save-image-path",
        help="Save the image with detected bounding boxes in a file.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Display more information about the procedure than you would normally",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    coco = COCO(COCO_ANNO_PATH)
    config = read_config(args.config)

    show = False

    # For single image
    image = None
    coco_anns = None
    img_info = None
    # For PDF and corpus
    image_list = []
    coco_anns_list = []
    img_info_list = []

    if args.mode == "page":
        for image_id, image_info in coco.imgs.items():
            if image_info["file_name"] == args.file.split("/")[-1]:
                coco_anns = load_coco_annotations(
                    coco.loadAnns(coco.getAnnIds([image_id]))
                )
                image_list.append(
                    "/".join(args.file.split("/")[:-1]) + "/" + image_info["file_name"]
                )
                coco_anns_list.append(coco_anns)
                img_info_list.append(image_info)
                break
        show = args.display_detection or args.display_ground
    elif args.mode == "pdf":
        for image_id, image_info in coco.imgs.items():
            if args.file.split("/")[-1] in image_info["file_name"]:
                coco_anns = load_coco_annotations(
                    coco.loadAnns(coco.getAnnIds([image_id]))
                )
                image_list.append(
                    "/".join(args.file.split("/")[:-1]) + "/" + image_info["file_name"]
                )
                coco_anns_list.append(coco_anns)
                img_info_list.append(image_info)
        show = args.display_detection or args.display_ground
    elif args.mode == "corpus":
        coco.imgs.keys()

    model = None
    # ==================================================================================================================
    # Currently considered actual models
    # ==================================================================================================================
    if args.dla_method == "layoutlmv3":
        model = (
            lp.LayoutLMv3LayoutModel(**config)
            if config is not None
            else lp.LayoutLMv3LayoutModel(WEIGHTS_PATH + "/layoutlmv3/model_final.pth")
        )
    elif args.dla_method == "dit":
        model = (
            lp.DiTLMv3LayoutModel(**config)
            if config is not None
            else lp.DiTLayoutModel(WEIGHTS_PATH + "/dit/publaynet_dit-b_cascade.pth")
        )
    elif args.dla_method == "doclayout-yolo":
        model = lp.DocLayoutYOLOLayoutModel(
            WEIGHTS_PATH + "/doclayoutyolo/doclayout_yolo_docstructbench_imgsz1024.pt"
        )
    elif args.dla_method == "dotsocr":
        if not check_connection():
            print(
                "For dots.ocr, first start the vLLM server before running this script."
            )
            exit()

        model = (
            lp.DotsOCRLayoutModel(**config)
            if config is not None
            else lp.DotsOCRLayoutModel()
        )
    elif args.dla_method == "vgt":
        # model = lp.VGTLayoutModel()
        raise Exception(f"VGT model is not yet implemented.")
    # ==================================================================================================================
    # Other models
    # ==================================================================================================================
    elif args.dla_method == "docstrum": # Mostly as proof of concept for a non-ML DLA
        model = (
            lp.DocstrumLayoutModel(**config, verbose=args.verbose)
            if config is not None
            else lp.DocstrumLayoutModel(verbose=args.verbose)
        )
        image = read_picture(args.file, to_rgb=False)
    elif args.dla_method == "nemotron": # TODO: define if it's gonna be added or not
        raise Exception(f"Nemotron model is not yet implemented.")
    elif args.dla_method == "detectron2": # Already included in layout-parser, included for thoroughness
        model = lp.Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            model_path="/home/luka/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth",
            # your local checkpoint
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )
    else:
        raise Exception(f"Unknown DLA method: {args.dla_method}")

    # Actually reading a picture
    if args.dla_method in ["detectron2"]:
        image_list = [read_picture(img) for img in image_list]

    categories = coco.cats

    if not coco_anns_list and not img_info_list:
        coco_anns_list.append(coco_anns)
        img_info_list.append(img_info)

    for image, coco_anns, img_info in zip(image_list, coco_anns_list, img_info_list):
        print(f"Processing image: {image}")
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull):
                # TODO: figure this out
                pass

        save_coco_path = (
            args.save
            if args.mode == "page"
            else args.save
            + "/"
            + image.split("/")[-1].split("_")[0]
            + "/"
            + image.split("/")[-1].split("_")[1].replace(".jpg", ".json")
        )
        main(
            image,
            img_info,
            coco_anns,
            model,
            (
                args.evaluation_metric.lower()
                if args.evaluation_metric is not None
                else None
            ),
            categories,  # TODO: keep in mind that ground and detection categories might not be the same!
            visualization=show,
            display_ground=args.display_ground,
            display_img=cv2.imread("/".join(args.file.split("/")[:-1]) + "/" + img_info["file_name"]),
            save_coco=save_coco_path,
            save_image_path=args.save_image_path,
        )

"""
Usage example:
python main.py \
    -m page \
    -i "./annotation/pawls/labels/images/ef9c7b51ec7999b1173234cdb44b04a9710ddb645881722289c6a097efed921b_3.jpg" \
    -v -dm detectron2
"""
