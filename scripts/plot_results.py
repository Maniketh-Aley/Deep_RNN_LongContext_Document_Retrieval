from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def read_csv(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_scaling(summary_rows, output_dir: Path) -> None:
    ensure_dir(output_dir)
    curves = defaultdict(lambda: {"x": [], "accuracy": [], "failure": []})
    for row in summary_rows:
        curves[row["model_type"]]["x"].append(int(row["sequence_length"]))
        curves[row["model_type"]]["accuracy"].append(float(row["accuracy_mean"]))
        curves[row["model_type"]]["failure"].append(float(row["failure_rate_mean"]))

    plt.figure(figsize=(8, 5))
    for model_type, payload in curves.items():
        plt.plot(payload["x"], payload["accuracy"], marker="o", label=model_type)
    plt.xscale("log", base=2)
    plt.xlabel("Sequence length")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sequence Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_sequence_length.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for model_type, payload in curves.items():
        plt.plot(payload["x"], payload["failure"], marker="o", label=model_type)
    plt.xscale("log", base=2)
    plt.xlabel("Sequence length")
    plt.ylabel("Failure rate")
    plt.title("Failure Rate vs Sequence Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "failure_rate_vs_sequence_length.png", dpi=200)
    plt.close()


def plot_position_sensitivity(log_dir: Path, output_dir: Path) -> None:
    metrics_files = sorted(log_dir.glob("*_metrics.json"))
    position_curves = defaultdict(lambda: defaultdict(list))
    for metrics_path in metrics_files:
        model_type = metrics_path.stem.replace("_metrics", "").split("_seed")[0]
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for bucket, accuracy in payload["accuracy_by_needle_position_bucket"].items():
            position_curves[model_type][bucket].append(float(accuracy))

    ordered_buckets = ["early", "middle", "late"]
    plt.figure(figsize=(8, 5))
    for model_type, bucket_map in position_curves.items():
        y = []
        for bucket in ordered_buckets:
            values = bucket_map.get(bucket, [0.0])
            y.append(sum(values) / len(values))
        plt.plot(ordered_buckets, y, marker="o", label=model_type)
    plt.xlabel("Needle position bucket")
    plt.ylabel("Accuracy")
    plt.title("Needle Position Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "needle_position_sensitivity.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NIAH benchmark results.")
    parser.add_argument("--summary-csv", type=str, default="outputs/logs/scaling_results_summary.csv")
    parser.add_argument("--log-dir", type=str, default="outputs/logs")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = read_csv(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    plot_scaling(summary_rows=summary_rows, output_dir=output_dir)
    plot_position_sensitivity(log_dir=Path(args.log_dir), output_dir=output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

