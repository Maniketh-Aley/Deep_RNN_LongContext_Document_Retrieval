from __future__ import annotations

from typing import Dict

from src.models.gru_baseline import GRUBaseline
from src.models.memory_gru import MemoryAugmentedGRU
from src.models.transformer_encoder import TransformerEncoderClassifier


def build_model(config: Dict, vocab_size: int, num_answers: int, pad_idx: int):
    model_cfg = config["model"]
    model_type = model_cfg["model_type"]
    if model_type == "gru":
        return GRUBaseline(
            vocab_size=vocab_size,
            num_answers=num_answers,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            pad_idx=pad_idx,
        )
    if model_type == "memory_gru":
        return MemoryAugmentedGRU(
            vocab_size=vocab_size,
            num_answers=num_answers,
            embedding_dim=model_cfg["embedding_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            pad_idx=pad_idx,
            memory_stride=model_cfg["memory_stride"],
            memory_top_k=model_cfg["memory_top_k"],
        )
    if model_type == "transformer":
        return TransformerEncoderClassifier(
            vocab_size=vocab_size,
            num_answers=num_answers,
            d_model=model_cfg["d_model"],
            num_heads=model_cfg["transformer_heads"],
            num_layers=model_cfg["transformer_layers"],
            ff_dim=model_cfg["transformer_ff_dim"],
            dropout=model_cfg["dropout"],
            pad_idx=pad_idx,
        )
    raise ValueError(f"Unsupported model type: {model_type}")

