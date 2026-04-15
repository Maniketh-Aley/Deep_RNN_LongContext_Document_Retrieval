from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_dataset import generate_dataset_from_config
from src.evaluation.evaluator import evaluate_model
from src.training.trainer import train_model
from src.utils.config import load_config, save_config
from src.utils.runtime import build_dataloaders, build_run_artifacts, make_model
from src.utils.seed import resolve_device, set_seed


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaling-law experiments.")
    parser.add_argument("--config", type=str, default="configs/research.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--models", type=str, nargs="+", default=["gru", "memory_gru", "transformer"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    max_seq_len = base_config["data"].get("max_seq_len", 8000)
    base_config["data"]["lengths"] = [
        sequence_length
        for sequence_length in base_config["data"]["lengths"]
        if sequence_length <= max_seq_len
    ]
    output_root = build_run_artifacts(base_config)
    detailed_path = output_root / "logs" / "scaling_results_detailed.csv"
    summary_path = output_root / "logs" / "scaling_results_summary.csv"
    results_rows = []

    for model_type in args.models:
        for seed in args.seeds:
            config = copy.deepcopy(base_config)
            config["model"]["model_type"] = model_type
            config["training"]["seed"] = seed
            config["data"]["seed"] = seed
            config["experiment_name"] = f"{model_type}_seed{seed}"
            config["data"]["dataset_dir"] = str(Path(base_config["data"]["dataset_dir"]) / f"seed_{seed}")
            set_seed(seed)
            generate_dataset_from_config(config)
            vocabulary, loaders = build_dataloaders(config)
            model = make_model(config, vocabulary)
            device = resolve_device(config["training"]["device"])
            model.to(device)

            artifacts = train_model(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                config=config,
                output_dir=output_root,
                device=device,
            )
            checkpoint = torch.load(artifacts["best_checkpoint"], map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            metrics = evaluate_model(
                model=model,
                data_loader=loaders["test"],
                idx_to_answer=vocabulary.idx_to_answer,
                device=device,
                output_dir=output_root / "logs",
                run_name=config["experiment_name"],
            )

            for sequence_length, accuracy in metrics["accuracy_by_sequence_length"].items():
                results_rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "sequence_length": int(sequence_length),
                        "accuracy": round(accuracy, 6),
                        "failure_rate": round(1.0 - accuracy, 6),
                    }
                )

            config_copy_path = output_root / "logs" / f"{config['experiment_name']}_config.yaml"
            save_config(config, config_copy_path)

    merged_rows = load_existing_rows(detailed_path)
    merged_by_key = {
        (row["model_type"], int(row["seed"]), int(row["sequence_length"])): {
            "model_type": row["model_type"],
            "seed": int(row["seed"]),
            "sequence_length": int(row["sequence_length"]),
            "accuracy": float(row["accuracy"]),
            "failure_rate": float(row["failure_rate"]),
        }
        for row in merged_rows
    }
    for row in results_rows:
        merged_by_key[(row["model_type"], row["seed"], row["sequence_length"])] = row
    results_rows = [merged_by_key[key] for key in sorted(merged_by_key)]

    per_group = {}
    for row in results_rows:
        key = (row["model_type"], row["sequence_length"])
        per_group.setdefault(key, []).append(row["accuracy"])

    summary_rows = []
    for (model_type, sequence_length), accuracies in sorted(per_group.items()):
        acc_mean = mean(accuracies)
        acc_std = pstdev(accuracies) if len(accuracies) > 1 else 0.0
        summary_rows.append(
            {
                "model_type": model_type,
                "sequence_length": sequence_length,
                "accuracy_mean": round(acc_mean, 6),
                "accuracy_std": round(acc_std, 6),
                "failure_rate_mean": round(1.0 - acc_mean, 6),
            }
        )

    with detailed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote detailed results to {detailed_path}")
    print(f"Wrote summary results to {summary_path}")


if __name__ == "__main__":
    main()
