def layout_parser_to_coco(layout, img_info, categories):
    coco_full = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    annotations = []
    annotation_id = 1

    for block in layout:
        category_name = block.type
        if category_name not in categories:
            continue  # Skip categories that are not in the mapping

        # category_id = category_mapping[category_name]
        category_id = category_name
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
    coco_full["categories"] = categories

    return coco_full
