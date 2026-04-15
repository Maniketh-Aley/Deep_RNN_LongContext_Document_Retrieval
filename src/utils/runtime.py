from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader

from src.data.dataset import NIAHDataset, build_vocabulary, collate_batch
from src.models.build import build_model
from src.utils.io import ensure_dir


def build_dataloaders(config: Dict) -> Tuple[object, Dict[str, DataLoader]]:
    dataset_dir = config["data"]["dataset_dir"]
    vocabulary = build_vocabulary(dataset_dir)
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    collate_fn = partial(collate_batch, pad_idx=vocabulary.pad_idx)

    loaders = {}
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        dataset = NIAHDataset(dataset_dir=dataset_dir, split=split, vocabulary=vocabulary)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    return vocabulary, loaders


def build_run_artifacts(config: Dict) -> Path:
    output_root = Path(config["output_root"])
    ensure_dir(output_root)
    for child in ["logs", "plots", "checkpoints"]:
        ensure_dir(output_root / child)
    return output_root


def make_model(config: Dict, vocabulary) -> object:
    return build_model(
        config=config,
        vocab_size=vocabulary.vocab_size,
        num_answers=vocabulary.num_answers,
        pad_idx=vocabulary.pad_idx,
    )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model-type", type=str, default=None, choices=["gru", "memory_gru", "transformer"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser


def apply_overrides(config: Dict, args: argparse.Namespace) -> Dict:
    if args.model_type is not None:
        config["model"]["model_type"] = args.model_type
    if args.seed is not None:
        config["training"]["seed"] = args.seed
        config["data"]["seed"] = args.seed
    if args.dataset_dir is not None:
        config["data"]["dataset_dir"] = args.dataset_dir
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    return config

