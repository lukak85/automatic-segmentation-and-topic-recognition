"""Evaluation metrics for document layout analysis."""

import re
import string
import sys
import unicodedata
from collections import Counter

# All Unicode punctuation characters plus standard ASCII punctuation
PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)


def normalize_answer(text):
    """Normalize text for comparison: lowercase, strip articles, punctuation, extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punctuation(s):
        return "".join(ch for ch in s if ch not in PUNCT)

    def collapse_whitespace(s):
        return " ".join(token for token in s.split() if token.strip())

    return collapse_whitespace(remove_articles(remove_punctuation(text.lower())))


def f1_score(prediction, ground_truth):
    """Compute token-level F1 score between two text strings.

    Based on the evaluation approach from evaluate_mlqa.py (UniLM).
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def mean_average_precision(predictions, ground_truths):
    """Compute mean average precision (mAP) for bounding box detections.

    Args:
        predictions: List of layout elements with .block attributes (x_1, y_1, width, height).
        ground_truths: List of ground-truth layout elements with .block attributes.

    Returns:
        Dict of mAP metrics from torchmetrics.
    """
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    pred_boxes = [
        [p.block.x_1, p.block.y_1, p.block.width, p.block.height]
        for p in predictions
    ]
    target_boxes = [
        [t.block.x_1, t.block.y_1, t.block.width, t.block.height]
        for t in ground_truths
    ]

    # Wrap in single-image format expected by torchmetrics
    preds = [{
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
        "scores": torch.ones(len(pred_boxes)),
        "labels": torch.ones(len(pred_boxes), dtype=torch.long),
    }]
    targets = [{
        "boxes": torch.tensor(target_boxes, dtype=torch.float32),
        "labels": torch.ones(len(target_boxes), dtype=torch.long),
    }]

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    return metric.compute()
