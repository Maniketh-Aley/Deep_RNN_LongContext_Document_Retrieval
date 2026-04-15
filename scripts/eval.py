from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_dataset import generate_dataset_from_config
from src.evaluation.evaluator import evaluate_model
from src.utils.config import load_config
from src.utils.runtime import add_common_args, apply_overrides, build_dataloaders, build_run_artifacts, make_model
from src.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NIAH benchmark model.")
    parser = add_common_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skip-data-gen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(config["training"]["seed"])
    if not args.skip_data_gen:
        generate_dataset_from_config(config)

    output_root = build_run_artifacts(config)
    vocabulary, loaders = build_dataloaders(config)
    model = make_model(config, vocabulary)
    device = resolve_device(config["training"]["device"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
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
    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()

