from collections import Counter
import string
import re
import sys
import unicodedata

PUNCT = {
    chr(i)
    for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith("P")
}.union(string.punctuation)


def whitespace_tokenize(text):
    return text.split()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        tokens = whitespace_tokenize(text)
        return " ".join([t for t in tokens if t.strip() != ""])

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(
    prediction, ground_truth
):  # Taken largely from evaluate_mlqa.py in unilm folder
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def mean_average_precision(predictions, ground_truths):
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    predictions_coco = []
    targets_coco = []

    for prediction in predictions:
        x, y, w, h = (
            prediction.block.x_1,
            prediction.block.y_1,
            prediction.block.width,
            prediction.block.height,
        )

        predictions_coco.append([x, y, w, h])

    predictions = [
        {
            "boxes": torch.tensor(predictions_coco, dtype=torch.float32),
            "scores": torch.tensor(
                [1.0] * len(predictions_coco)
            ),  # Assuming a confidence score of 1.0 for all predictions
            "labels": torch.tensor(
                [1] * len(predictions_coco)
            ),  # Assuming a single class label for all predictions
        }
    ]

    for target in ground_truths:
        x, y, w, h = (
            target.block.x_1,
            target.block.y_1,
            target.block.width,
            target.block.height,
        )

        targets_coco.append([x, y, w, h])

    targets = [
        {
            "boxes": torch.tensor(targets_coco, dtype=torch.float32),
            "labels": torch.tensor(
                [1] * len(targets_coco)
            ),  # Assuming a single class label for all targets
        }
    ]

    metric = MeanAveragePrecision(iou_type="bbox")

    metric.update(predictions, targets)
    result = metric.compute()
    return result
