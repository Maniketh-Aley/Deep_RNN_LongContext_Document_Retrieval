from __future__ import annotations

import torch
import torch.nn.functional as F


class HiddenStateMemory(torch.nn.Module):
    def __init__(self, stride: int = 32, top_k: int = 4):
        super().__init__()
        self.stride = stride
        self.top_k = top_k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        memory_states = hidden_states[:, :: self.stride, :]
        memory_mask = attention_mask[:, :: self.stride]

        normalized_query = F.normalize(query, dim=-1)
        normalized_memory = F.normalize(memory_states, dim=-1)
        scores = torch.einsum("bd,bmd->bm", normalized_query, normalized_memory)
        scores = scores.masked_fill(~memory_mask, float("-inf"))

        top_k = min(self.top_k, memory_states.size(1))
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        gathered_memory = memory_states.gather(
            1,
            top_indices.unsqueeze(-1).expand(batch_size, top_k, hidden_dim),
        )
        weights = torch.softmax(top_scores, dim=-1)
        retrieved = torch.sum(gathered_memory * weights.unsqueeze(-1), dim=1)
        return retrieved, top_indices

