# TODO: move
DOCLAYOUT_YOLO_PUBLAY_TO_OUR_LABEL_MAP = {
    "title": 0,
    "plain text": 5,
    "table": 11,
    "figure": 10,
    "table_caption": 13,
    "figure_caption": 13,
}


def layout_parser_to_coco(
    layout,
    img_info,
    categories,
    category_mapping=DOCLAYOUT_YOLO_PUBLAY_TO_OUR_LABEL_MAP,
):
    coco_full = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    annotations = []
    annotation_id = 1

    for block in layout:
        category_name = block.type
        # category_name = next(
        #     (obj["id"] for obj in categories.values() if obj["name"] == category_name),
        #     None,
        # )

        # if category_name is None:
        #     continue  # Skip if category name is not found in categories

        if category_name not in category_mapping:
            category_id = 4
        else:
            category_id = category_mapping[category_name]
        # category_id = category_name
        x_min, y_min, x_max, y_max = block.coordinates
        width = x_max - x_min
        height = y_max - y_min

        coco_annotation = {
            "id": int(annotation_id),
            "image_id": img_info["id"],
            "category_id": int(category_id),
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": False,
            "score": float(block.score) if hasattr(block, "score") else 1.0,
        }
        annotations.append(coco_annotation)
        annotation_id += 1

    coco_full["annotations"] = annotations

    image = {
        "id": img_info["id"],
        "file_name": img_info["file_name"],
        "width": img_info["width"],
        "height": img_info["height"],
    }

    coco_full["images"].append(image)
    coco_full["categories"] = []
    for category_id in categories:
        coco_full["categories"].append(categories[category_id])

    return coco_full


def pdf_to_image(filename):
    from pdf2image import convert_from_path
    pages = convert_from_path(f'./data/pdf/{filename}.pdf', 500)

    for count, page in enumerate(pages):
        page.save(f'./data/pdf/images/input/{filename}_{count}.jpg', 'JPEG')
