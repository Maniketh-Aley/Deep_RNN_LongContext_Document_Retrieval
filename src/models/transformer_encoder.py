from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        pad_idx: int,
        max_len: int = 20001,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_answers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        embeddings = self.embedding(input_ids)
        batch_size = embeddings.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, embeddings], dim=1)
        x = self.positional_encoding(x)

        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=~full_mask)
        cls_rep = encoded[:, 0]
        logits = self.classifier(self.dropout(cls_rep))
        return {"logits": logits, "sequence_representation": cls_rep}

