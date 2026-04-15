from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]"]


@dataclass
class Vocabulary:
    token_to_idx: Dict[str, int]
    idx_to_token: List[str]
    answer_to_idx: Dict[str, int]
    idx_to_answer: List[str]

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx["[PAD]"]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx["[UNK]"]

    @property
    def cls_idx(self) -> int:
        return self.token_to_idx["[CLS]"]

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_token)

    @property
    def num_answers(self) -> int:
        return len(self.idx_to_answer)


def _read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def build_vocabulary(dataset_dir: str | Path) -> Vocabulary:
    dataset_dir = Path(dataset_dir)
    tokens = set(SPECIAL_TOKENS)
    answers = set()
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        answers.update(metadata.get("passkeys", []))
        tokens.update(metadata.get("noise_vocab", []))
    for split in ["train", "val", "test"]:
        for record in _read_jsonl(dataset_dir / f"{split}.jsonl"):
            tokens.update(record["tokens"])
            answers.add(record["answer"])
    tokens.update(answers)
    idx_to_token = sorted(tokens)
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
    idx_to_answer = sorted(answers)
    answer_to_idx = {answer: idx for idx, answer in enumerate(idx_to_answer)}
    return Vocabulary(
        token_to_idx=token_to_idx,
        idx_to_token=idx_to_token,
        answer_to_idx=answer_to_idx,
        idx_to_answer=idx_to_answer,
    )


class NIAHDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str, vocabulary: Vocabulary):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.vocabulary = vocabulary
        self.records = _read_jsonl(self.dataset_dir / f"{split}.jsonl")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        token_ids = [
            self.vocabulary.token_to_idx.get(token, self.vocabulary.unk_idx)
            for token in record["tokens"]
        ]
        return {
            "id": record["id"],
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": len(token_ids),
            "sequence_length": record["sequence_length"],
            "needle_position": record["needle_position"],
            "needle_position_bucket": record["needle_position_bucket"],
            "answer_idx": self.vocabulary.answer_to_idx[record["answer"]],
            "answer_text": record["answer"],
        }


def collate_batch(batch: List[Dict], pad_idx: int) -> Dict[str, torch.Tensor | List]:
    batch_size = len(batch)
    max_length = max(sample["length"] for sample in batch)
    input_ids = torch.full((batch_size, max_length), pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = []
    labels = []
    ids = []
    answers = []
    seq_lengths = []
    needle_positions = []
    position_buckets = []

    for row_idx, sample in enumerate(batch):
        length = sample["length"]
        input_ids[row_idx, :length] = sample["input_ids"]
        attention_mask[row_idx, :length] = True
        lengths.append(length)
        labels.append(sample["answer_idx"])
        ids.append(sample["id"])
        answers.append(sample["answer_text"])
        seq_lengths.append(sample["sequence_length"])
        needle_positions.append(sample["needle_position"])
        position_buckets.append(sample["needle_position_bucket"])

    return {
        "ids": ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "answers": answers,
        "sequence_lengths": torch.tensor(seq_lengths, dtype=torch.long),
        "needle_positions": torch.tensor(needle_positions, dtype=torch.long),
        "needle_position_buckets": position_buckets,
    }
