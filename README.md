# Automatic Segmentation and Topic Recognition (ASTR)

A document layout analysis (DLA) pipeline built on top of
[layoutparser](https://github.com/Layout-Parser/layout-parser) with planned support for segment topic recognition on top of it. It automatically detects and
segments document elements (headings, paragraphs, figures, tables, captions, etc.) from page images or PDFs (or by
extension a corpus). The pipeline outputs annotations in COCO format.

## Supported Models

| Model              | Type                        | Description                                                                | Repository                                                                       |
|--------------------|-----------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **Docstrum**       | Bottom-up                   | Line-based document structure analysis                                     | **[chulwoopack/docstrum](https://github.com/chulwoopack/docstrum)**              |
| **LayoutLMv3**     | Multimodal                  | Jointly encodes text, layout, and image patches for document understanding | **[microsoft/unilm](https://github.com/microsoft/unilm/tree/master/layoutlmv3)** |
| **DiT**            | Document Image Transformers | Document image transformer pre-trained via masked image modeling           | **[microsoft/unilm](https://github.com/microsoft/unilm/tree/master/dit)**        |
| **DocLayout-YOLO** | Object detection            | YOLOv10-based model for document structure                                 | **[opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)**  |
| **DotsOCR**        | Vision-language             | Layout detection + OCR via vLLM server                                     | **[rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)**          |

<details>
<summary><b>Work in Progress and Planned Models</b></summary>

### Work in Progress Models

| Model        | Type            | Description | Repository                                                                                                                                        |
|--------------|-----------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Nemotron** | Vision-language | TODO        | **[nvidia/nemotron-page-elements-v3](https://huggingface.co/nvidia/nemotron-page-elements-v3)**                                                   |
| **VGT**      | Multimodal      | TODO        | **[AlibabaResearch/AdvancedLiterateMachinery](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT)** |

### Planned Models

| Model               | Type              | Description | Repository                                                        |
|---------------------|-------------------|-------------|-------------------------------------------------------------------|
| **Faster R-CNN**    | CNN-Based         | TODO        | TODO                                                              |
| **Mask R-CNN**      | CNN-Based         | TODO        | TODO                                                              |
| **MinerU**          | TODO              | TODO        | **[opendatalab/mineru](https://github.com/opendatalab/mineru)**   |
| **DLAFormer**       | TODO              | TODO        | No public repository available                                    |
| **Paragraph2Graph** | TODO              | TODO        | **[NormXU/Layout2Graph](https://github.com/NormXU/Layout2Graph)** |
| **DINO**            | Transformer-Based | TODO        | **[IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)**   |
</details>

## Installation

Install layoutparser in editable mode (from the project root):

```bash
pip install -e layout-parser
```

For model-specific dependencies, see [Model-Specific Setup](#model-specific-setup) below. It is recommended to create a separate Conda
environment for each of the models.

## Usage

### Running Layout Detection (`main.py`)

```bash
python main.py -dm <model> -f <image_path> [options]
```

**Required arguments:**
- `-dm`, `--dla-method` — DLA model: `detectron2`, `doclayout-yolo`, `docstrum`, `dotsocr`
- `-f`, `--file` — Path to the input image or PDF image folder

**Optional arguments:**
- `-m`, `--mode` — Processing mode: `page` (single image, default), `pdf` (all pages), or `corpus`
- `-c`, `--config` — JSON configuration file for the model
- `-s`, `--save` — Save detections as COCO JSON to this path
- `-dd`, `--display-detection` — Display detected bounding boxes
- `-dg`, `--display-ground` — Display ground-truth bounding boxes
- `-e`, `--evaluation-metric` — Evaluate detections: `f1` or `map`
- `-v`, `--verbose` — Enable verbose output
- `-si`, `--save-image` — Save the visualization image

**Examples:**

Detect layout on a single page with DocLayout-YOLO and save COCO annotations:
```bash
python main.py -m page \
  -f ./dataset/images/example_page.jpg \
  -dm doclayout-yolo \
  -s ./results/output.json
```

Detect layout for all pages of a PDF and display results:
```bash
python main.py -m pdf \
  -f ./dataset/images/document_hash \
  -dm doclayout-yolo \
  -s ./results \
  -dd
```

Run with Docstrum and a custom configuration:
```bash
python main.py -m page \
  -f ./dataset/images/example_page.jpg \
  -dm docstrum \
  -c ./configs/docstrum_config.json \
  -v
```

### Managing Annotations (`helper.py`)

```bash
python helper.py -m <mode> [options]
```

**Modes:**
- `join-annotations` — Merge multiple COCO JSON files into one
- `prepare-annotations` — Join + deduplicate + visualize
- `order-images` — Sort images and annotations by ID
- `remove-scores` — Strip confidence scores from annotations
- `review-annotations` — Visualize all annotations for review

**Examples:**

Join annotation files from a folder:
```bash
python helper.py -m join-annotations \
  -p ./results/document_hash/ \
  -o ./results/document_hash/merged.json
```

Remove duplicate detections (IoU > 0.95):
```bash
python helper.py -r \
  -a ./results/merged.json \
  -o ./results/deduplicated.json
```

Visualize annotations for a specific image:
```bash
python helper.py \
  -a ./results/merged.json \
  -i 2028
```

## Project Structure

```
automatic-segmentation-and-topic-recognition/
├── main.py                  # Main DLA pipeline entry point
├── helper.py                # Annotation management utilities
├── utils/
│   ├── conversionutils.py   # Layout-to-COCO format conversion
│   ├── displayutils.py      # Bounding box visualization
│   ├── evalutils.py         # Evaluation metrics
│   └── fileutils.py         # JSON config and file I/O
├── layout-parser/           # Extended layoutparser (git submodule)
│   └── src/layoutparser/models/
│       ├── dit/             # DiT model
│       ├── doclayout_yolo/  # DocLayout-YOLO model
│       ├── docstrum/        # Docstrum algorithm
│       ├── dotsocr/         # DotsOCR vision-language model
│       └── layoutlmv3/      # LayoutLMv3 model
└── annotation/              # Ground-truth COCO annotations
    └── images/              # Document images
```

## Model-Specific Setup

### Docstrum

```bash
pip install opencv-python scipy shapely
```

No model weights needed — Docstrum is a classical algorithm based on nearest-neighbor clustering of character centroids.

### LayoutLMv3 and DiT

The models use detectron2 as a detection backbone. For installation, follow these instructions:
- [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3#installation)
- [DiT](https://github.com/microsoft/unilm/tree/master/layoutlmv3#installation)

### DocLayout-YOLO

```bash
pip install doclayout-yolo
```

Download the model weights from [HuggingFace](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) and place them in `./data/model/doclayoutyolo/`.

### DotsOCR

Recommended usage of vLLM server. Set up the environment by following the installation instructions on
[dots.ocr](https://github.com/rednote-hilab/dots.ocr?tab=readme-ov-file#1-installation) repository. Then start the vLLM
server before running detection:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --chat-template-content-format string \
  --served-model-name model --trust-remote-code
```

## Output Format

Detections are saved in [COCO format](https://cocodataset.org/#format-data):

```json
{
  "images": [{"id": 1, "file_name": "page.jpg", "width": 800, "height": 1200}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 10, "bbox": [50, 100, 700, 200], "area": 140000}
  ],
  "categories": [
    {"id": 5, "name": "Headline"},
    {"id": 10, "name": "Paragraph"},
    {"id": 13, "name": "Figure"},
    {"id": 14, "name": "Caption"}
  ]
}
```
