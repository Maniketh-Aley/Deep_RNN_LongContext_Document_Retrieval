from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed

MAX_SEQUENCE_LENGTH = 8000


def _make_noise_vocab(vocab_size: int) -> List[str]:
    base_words = [f"WORD_{idx:04d}" for idx in range(vocab_size)]
    noise_words = [f"NOISE_{idx:04d}" for idx in range(max(32, vocab_size // 8))]
    return base_words + noise_words


def _make_passkeys(num_passkeys: int, rng: random.Random) -> List[str]:
    passkeys = []
    for _ in range(num_passkeys):
        left = rng.randint(1000, 9999)
        right = rng.randint(1000, 9999)
        passkeys.append(f"PASSKEY-{left}-{right}")
    return sorted(set(passkeys))


def _position_bucket(position: int, sequence_length: int) -> str:
    ratio = position / max(sequence_length - 1, 1)
    if ratio < 1 / 3:
        return "early"
    if ratio < 2 / 3:
        return "middle"
    return "late"


def generate_split(
    split: str,
    lengths: List[int],
    samples_per_length: int,
    noise_vocab: List[str],
    passkeys: List[str],
    num_needles: int,
    base_seed: int,
) -> List[Dict]:
    samples: List[Dict] = []
    split_offsets = {"train": 11, "val": 29, "test": 47}
    for length in lengths:
        if length > MAX_SEQUENCE_LENGTH:
            raise ValueError("Sequence length exceeds allowed limit (8000)")
        split_rng = random.Random(base_seed + split_offsets[split] * 100_000 + length)
        for sample_idx in range(samples_per_length):
            sample_rng = random.Random(split_rng.randint(0, 10**9) + sample_idx)
            tokens = [sample_rng.choice(noise_vocab) for _ in range(length)]
            answer = sample_rng.choice(passkeys)
            positions = sorted(sample_rng.sample(range(length), k=num_needles))
            for pos in positions:
                tokens[pos] = answer
            primary_pos = positions[0]
            sample_id = f"{split}_L{length}_{sample_idx:06d}"
            samples.append(
                {
                    "id": sample_id,
                    "tokens": tokens,
                    "sequence_length": length,
                    "needle_position": primary_pos,
                    "needle_positions": positions,
                    "needle_position_bucket": _position_bucket(primary_pos, length),
                    "passkey": answer,
                    "question": "What is the passkey?",
                    "answer": answer,
                    "num_needles": num_needles,
                }
            )
    return samples


def generate_dataset_from_config(config: Dict) -> Dict[str, Path]:
    data_cfg = config["data"]
    max_seq_len = min(data_cfg.get("max_seq_len", MAX_SEQUENCE_LENGTH), MAX_SEQUENCE_LENGTH)
    lengths = data_cfg["lengths"]
    if any(length > max_seq_len for length in lengths):
        raise ValueError("Sequence length exceeds allowed limit (8000)")
    dataset_dir = Path(data_cfg["dataset_dir"])
    ensure_dir(dataset_dir)
    set_seed(data_cfg["seed"])

    rng = random.Random(data_cfg["seed"])
    noise_vocab = _make_noise_vocab(data_cfg["vocab_size"])
    passkeys = _make_passkeys(data_cfg["num_passkeys"], rng)

    metadata = {
        "seed": data_cfg["seed"],
        "lengths": lengths,
        "max_seq_len": max_seq_len,
        "vocab_size": data_cfg["vocab_size"],
        "num_passkeys": len(passkeys),
        "num_needles": data_cfg["num_needles"],
        "noise_vocab_size": len(noise_vocab),
        "noise_vocab": noise_vocab,
        "passkeys": passkeys,
    }
    write_json(dataset_dir / "metadata.json", metadata)

    output_paths: Dict[str, Path] = {}
    split_to_count = {
        "train": data_cfg["train_samples_per_length"],
        "val": data_cfg["val_samples_per_length"],
        "test": data_cfg["test_samples_per_length"],
    }
    for split, samples_per_length in split_to_count.items():
        samples = generate_split(
            split=split,
            lengths=lengths,
            samples_per_length=samples_per_length,
            noise_vocab=noise_vocab,
            passkeys=passkeys,
            num_needles=data_cfg["num_needles"],
            base_seed=data_cfg["seed"],
        )
        out_path = dataset_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample) + "\n")
        output_paths[split] = out_path
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic NIAH benchmark data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_paths = generate_dataset_from_config(config)
    print("Generated dataset:")
    for split, path in output_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
