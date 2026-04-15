from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import bucketed_accuracy, compute_classification_metrics
from src.utils.io import ensure_dir, write_csv, write_json


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    idx_to_answer: List[str],
    device: torch.device,
    output_dir: str | Path,
    run_name: str,
) -> Dict:
    model.eval()
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    all_predictions = []
    all_labels = []
    per_example_rows = []

    for batch in tqdm(data_loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

        for row_idx in range(predictions.size(0)):
            pred_idx = predictions[row_idx].item()
            label_idx = labels[row_idx].item()
            correct = int(pred_idx == label_idx)
            seq_len = int(batch["sequence_lengths"][row_idx].item())
            needle_position = int(batch["needle_positions"][row_idx].item())
            per_example_rows.append(
                {
                    "id": batch["ids"][row_idx],
                    "sequence_length": seq_len,
                    "needle_position": needle_position,
                    "needle_position_bucket": batch["needle_position_buckets"][row_idx],
                    "prediction": idx_to_answer[pred_idx],
                    "answer": idx_to_answer[label_idx],
                    "correct": correct,
                }
            )

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(predictions=predictions, labels=labels)
    length_accuracy = bucketed_accuracy(per_example_rows, key="sequence_length")
    position_accuracy = bucketed_accuracy(per_example_rows, key="needle_position_bucket")

    metrics_payload = {
        **metrics,
        "accuracy_by_sequence_length": length_accuracy,
        "accuracy_by_needle_position_bucket": position_accuracy,
    }
    write_json(output_dir / f"{run_name}_metrics.json", metrics_payload)
    write_csv(output_dir / f"{run_name}_predictions.csv", per_example_rows)
    return metrics_payload

