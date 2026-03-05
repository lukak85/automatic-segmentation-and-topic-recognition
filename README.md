# Automatic Segmentation and Topic Recognition

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
- LayoutLMv3
- DiT
- DocLayout-YOLO
- Dots.OCR

## Pre-requisites

Run the following command in the appropriate environment to install layoutparser:

```bash
pip install -e layoutparser
```

For model specific dependencies, see [Model specific dependencies](#model-specific-dependencies) section below.

# Example usage of `main.py` file

These examples use DocLayout-YOLO as the model for segment detection, but you can replace it with any of the supported
models by changing the `-dm` argument.

- For detecting segments for a single page:
    ```bash
    python main.py -m page \
      -f 230edb119aff067fecd3586eb3ce857f9ce402b0867037c156efaaaa32d0ba4b \
      -dm doclayout-yolo \
      -s ./results/temp.jsons
    ```

- For detecting segments for a whole PDF (given you have a folder with all the pages of the PDF as images, with the 
same prefix):
    ```bash
    python main.py -m pdf \
      -f ./annotation/pawls/labels/images/9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21 \
      -dm doclayout-yolo \
      -s ./results \
      -dd
    ```

# Example usage of `helper.py` file

- For joining all annotation files into a single COCO formatted JSON file:
    ```bash
    python helper.py -m join-annotations \
        -p ./results/9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21 \
        -o ./results/9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21/output.json
    ```

- For displaying the detected segments saved as COCO formatted JSON files:
    ```bash
    python helper.py -i 2028 \
        -a ./results/9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21/output.json \
        -p ./annotation/pawls/labels/images/9057f730adf6c4b43959e687df737ed7c84618b62567853161ee45cbb688ba21_15.jpg
    ```

# Model specific dependencies

In order to run the specified models, it's recommended to create separate environments for each of them. The
dependencies for each model are listed below.

## DocLayout-YOLO

```bash
pip install -r TODO
```