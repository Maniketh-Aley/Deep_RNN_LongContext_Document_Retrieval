from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def compute_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    correct = (predictions == labels).float()
    accuracy = correct.mean().item()
    return {
        "accuracy": accuracy,
        "exact_match_accuracy": accuracy,
        "top1_retrieval_accuracy": accuracy,
        "failure_rate": 1.0 - accuracy,
    }


def bucketed_accuracy(rows: Iterable[Dict], key: str) -> Dict[str, float]:
    bucket_to_values: Dict[str, List[float]] = {}
    for row in rows:
        bucket = str(row[key])
        bucket_to_values.setdefault(bucket, []).append(float(row["correct"]))
    return {bucket: sum(values) / len(values) for bucket, values in bucket_to_values.items()}

