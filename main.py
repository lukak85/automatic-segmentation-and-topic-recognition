import argparse
import json

from pycocotools.coco import COCO

from displayutils import *
from evalutils import *

# ======================================================================================================================
# Parameters
# ======================================================================================================================

COCO_ANNO_PATH="./annotation/pawls/labels/development_user@example.com.json"


# ======================================================================================================================
# Main function
# ======================================================================================================================

def main(img, ground_truth, model, evaluation_metic, visualization=False, display_ground=False,  display_img=None):
    # Preform layout analysis
    if model is not None: # TODO: do check if it's correct class
        layout = model.detect(img)
    else:
        raise Exception("No DLA model is provided.")

    # Visualization
    if visualization:
        draw_layout(display_img, layout) # Drawing box for detection
        if ground_truth is not None and display_ground:
            draw_layout(display_img, ground_truth) # Drawing box for ground truth


    # Evaluation
    if evaluation_metic is not None: # TODO: add back later
        if evaluation_metic == "f1":
            print(f1_score(ground_truth, layout))


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

        x, y, w, h = ele['bbox']

        layout.append(
            lp.TextBlock(
                block = lp.Rectangle(x, y, w+x, h+y),
                type  = ele['category_id'] if coco is None else coco.cats[ele['category_id']]['name'],
                id = ele['id']
            )
        )

    return layout

def check_connection(address="localhost", port=8080, timeout=5):
    import socket
    try:
        # connect to the host -- tells us if the host is actually reachable
        socket.create_connection((address, port))
        return True
    except OSError:
        pass
    return False


# ======================================================================================================================
# Run
# ======================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Document layout analysis pipeline")
    parser.add_argument("-c", "--config",
                        help="JSON configuration file path for the model", type=str,
                        required=False)
    parser.add_argument("-dg", "--display-ground",
                        help="Display ground", action='store_true',
                        default=False)
    parser.add_argument("-dm", "--dla-method",
                        help="Method for Document Layout Analysis", type=str,
                        required=True)
    parser.add_argument("-e", "--evaluation_metric",
                        help="Evaluation metric for Document Layout Analysis", type=str,
                        required=False)
    parser.add_argument("-i", "--image",
                        help="Path to the input image", type=str,
                        required=True)
    parser.add_argument("-m", "--mode",
                        help="Detecting single image, whole PDF, or the whole corpus", type=str,
                        required=True)
    parser.add_argument("-s", "--show",
                        help="Show image with bounding boxes if mode is single image", action='store_true',
                        default=False)
    parser.add_argument("-v", "--verbose",
                        help="Display more information about the procedure than you would normally", action='store_true',
                        default=False)

    args = parser.parse_args()

    coco = COCO(COCO_ANNO_PATH)
    config = None
    if args.config is not None:
        with open(args.config, 'r') as file:
            config = json.load(file)

    coco_anns = None
    show = False
    image = None
    if args.mode == "page":
        for image_id, image_info in coco.imgs.items():
            if image_info['file_name'] == args.image.split('/')[-1]:
                coco_anns = load_coco_annotations(coco.loadAnns(coco.getAnnIds([image_id])))
                break
        show = args.show
    elif args.mode == "pdf":
        # TODO
        pass
    elif args.mode == "corpus":
        coco.imgs.keys()

    model = None
    if args.dla_method == "detectron2":
        model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                         model_path="/home/luka/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth",
                                         # your local checkpoint
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                         label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    elif args.dla_method == "docstrum":
        from layoutparser.src.layoutparser.models.docstrum import DocstrumLayoutModel
        if config is None:
            model = DocstrumLayoutModel(verbose=args.verbose)
        else:
            model = DocstrumLayoutModel(**config, verbose=args.verbose)
        image = read_picture(args.image, to_rgb=False)
        pass
    elif args.dla_method == "dotsocr":
        if not check_connection():
            print("For dots.ocr, first start the vLLM server before running this script.")
            exit()
        from layoutparser.src.layoutparser.models.docstrum import DotsOCRLayoutModel
        if config is None:
            model = DotsOCRLayoutModel()
        else:
            model = DotsOCRLayoutModel(**config)
    elif args.dla_method == "layoutlmv3":
        # TODO: implement LayoutLMv3 model
        #from layoutparser.src.layoutparser.models.layoutlmv3 import LayoutLMv3LayoutModel
        #if config is None:
        #    model = LayoutLMv3LayoutModel()
        #else:
        #    model = LayoutLMv3LayoutModel(**config)
        raise Exception(f"LayoutLMv3 model is not yet implemented.")
    else:
        raise Exception(f"Unknown DLA method: {args.dla_method}")

    if image is None:
        image = read_picture(args.image)

    main(image, coco_anns, model, args.evaluation_metric.lower() if args.evaluation_metric is not None else None,
         visualization=show, display_ground=args.display_ground, display_img=cv2.imread(args.image))

"""
Usage example:
python main.py \
    -m page \
    -i "./annotation/pawls/labels/images/ef9c7b51ec7999b1173234cdb44b04a9710ddb645881722289c6a097efed921b_3.jpg" \
    -v -dm detectron2
"""