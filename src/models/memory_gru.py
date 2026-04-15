from __future__ import annotations

import torch
from torch import nn

from src.memory.retrieval import HiddenStateMemory


class MemoryAugmentedGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        memory_stride: int,
        memory_top_k: int,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.memory = HiddenStateMemory(stride=memory_stride, top_k=memory_top_k)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_answers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        embeddings = self.embedding(input_ids)
        outputs, _ = self.encoder(embeddings)
        lengths = attention_mask.long().sum(dim=1) - 1
        final_hidden = outputs[torch.arange(outputs.size(0), device=outputs.device), lengths]
        retrieved, memory_indices = self.memory(outputs, attention_mask, final_hidden)
        fused = self.fusion(torch.cat([final_hidden, retrieved], dim=-1))
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "sequence_representation": fused,
            "memory_indices": memory_indices,
        }

