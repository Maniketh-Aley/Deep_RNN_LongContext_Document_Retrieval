from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generate_dataset import generate_dataset_from_config
from src.training.trainer import train_model
from src.utils.config import load_config, save_config
from src.utils.runtime import add_common_args, apply_overrides, build_dataloaders, build_run_artifacts, make_model
from src.utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NIAH benchmark model.")
    parser = add_common_args(parser)
    parser.add_argument("--skip-data-gen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    set_seed(config["training"]["seed"])

    if not args.skip_data_gen:
        generate_dataset_from_config(config)

    output_root = build_run_artifacts(config)
    save_config(config, output_root / "logs" / f"{config['experiment_name']}_config.yaml")
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
    print("Training complete.")
    print(f"Best checkpoint: {artifacts['best_checkpoint']}")
    print(f"Best val accuracy: {artifacts['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()

