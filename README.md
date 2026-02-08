# Automatic segmentation and topic recognition

The methods used in this repository are built into a layoutparser pipeline that can automatically segment documents. Our
extension is topic recognition of the detected segments. An example usage:

```bash
python main.py \
    -m page \
    -i "./annotation/pawls/labels/images/ef9c7b51ec7999b1173234cdb44b04a9710ddb645881722289c6a097efed921b_3.jpg" \
    -v -dm detectron2
```

Currently supported models (apart from the already included ones) are:
- docstrum (built on top of https://github.com/chulwoopack/docstrum)
- layoutlmv3
- dots.ocr

## Pre-requisites

Run the following command in the appropriate environment to install layoutparser:

```bash
pip install -e layoutparser
```

### Model specific dependencies

#### DocLayout-YOLO

```bash
pip install -r TODO
```