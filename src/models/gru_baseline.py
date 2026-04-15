from __future__ import annotations

import torch
from torch import nn


class GRUBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_answers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        embeddings = self.embedding(input_ids)
        outputs, _ = self.encoder(embeddings)
        lengths = attention_mask.long().sum(dim=1) - 1
        final_hidden = outputs[torch.arange(outputs.size(0), device=outputs.device), lengths]
        logits = self.classifier(self.dropout(final_hidden))
        return {"logits": logits, "sequence_representation": final_hidden}

