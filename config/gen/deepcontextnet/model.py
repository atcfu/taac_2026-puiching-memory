from __future__ import annotations

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


class SequentialTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.pre_attention_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.pre_ffn_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.pre_attention_norm(hidden_states)
        batch_size, token_count, _ = normalized.shape
        qkv = self.qkv(normalized).view(batch_size, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if token_mask is not None:
            attention_logits = attention_logits.masked_fill(~token_mask.unsqueeze(1).unsqueeze(2), -1.0e4)

        attention_weights = torch.softmax(attention_logits, dim=-1)
        attended = torch.matmul(attention_weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, token_count, self.hidden_dim)

        hidden_states = hidden_states + self.dropout(self.output_projection(attended))
        hidden_states = hidden_states + self.dropout(self.feed_forward(self.pre_ffn_norm(hidden_states)))
        return hidden_states


class DeepContextNetModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.recent_seq_len = max(0, model_config.recent_seq_len)

        self.token_embedding = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            embedding_dim=model_config.embedding_dim,
            padding_idx=0,
        )
        self.token_projection = (
            nn.Identity()
            if model_config.embedding_dim == model_config.hidden_dim
            else nn.Linear(model_config.embedding_dim, model_config.hidden_dim)
        )

        self.time_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.group_embedding = nn.Embedding(len(data_config.sequence_names) + 1, model_config.hidden_dim, padding_idx=0)
        self.global_token = nn.Parameter(torch.randn(1, 1, model_config.hidden_dim) * 0.02)

        self.user_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.item_projection = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.bottleneck = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)
        self.classifier = nn.Linear(model_config.hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [
                SequentialTemporalBlock(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
                    dropout=model_config.dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )

    def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError(f"Batch is missing required tensor: {name}")
        return tensor

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _slice_recent(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.recent_seq_len <= 0 or tensor.shape[1] <= self.recent_seq_len:
            return tensor
        return tensor[:, -self.recent_seq_len :]

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        user_tokens = self._require(batch.user_tokens, "user_tokens")
        user_mask = self._require(batch.user_mask, "user_mask")
        candidate_post_tokens = self._require(batch.candidate_post_tokens, "candidate_post_tokens")
        candidate_post_mask = self._require(batch.candidate_post_mask, "candidate_post_mask")
        candidate_author_tokens = self._require(batch.candidate_author_tokens, "candidate_author_tokens")
        candidate_author_mask = self._require(batch.candidate_author_mask, "candidate_author_mask")
        history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
        history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")
        history_time_gap = self._require(batch.history_time_gap, "history_time_gap")
        history_group_ids = self._require(batch.history_group_ids, "history_group_ids")

        history_mask = self._slice_recent(batch.history_mask)
        history_post_tokens = self._slice_recent(history_post_tokens)
        history_action_tokens = self._slice_recent(history_action_tokens)
        history_time_gap = self._slice_recent(history_time_gap)
        history_group_ids = self._slice_recent(history_group_ids)

        user_summary = masked_mean(self._embed_tokens(user_tokens), user_mask)
        context_summary = masked_mean(self._embed_tokens(batch.context_tokens), batch.context_mask)
        dense_summary = self.dense_projection(batch.dense_features)
        user_node = self.user_projection(torch.cat([user_summary, context_summary, dense_summary], dim=-1))

        candidate_post_summary = masked_mean(self._embed_tokens(candidate_post_tokens), candidate_post_mask)
        candidate_author_summary = masked_mean(self._embed_tokens(candidate_author_tokens), candidate_author_mask)
        candidate_summary = masked_mean(self._embed_tokens(batch.candidate_tokens), batch.candidate_mask)
        item_node = self.item_projection(
            torch.cat(
                [candidate_post_summary, candidate_author_summary, candidate_summary],
                dim=-1,
            )
        )

        sequence_tokens = (
            self._embed_tokens(history_post_tokens)
            + self._embed_tokens(history_action_tokens)
            + self.time_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
            + self.group_embedding(history_group_ids.clamp(min=0, max=self.group_embedding.num_embeddings - 1))
        )
        sequence_tokens = sequence_tokens * history_mask.unsqueeze(-1).float()

        batch_size = batch.batch_size
        cls_token = self.global_token.expand(batch_size, -1, -1)
        combined_tokens = torch.cat(
            [
                cls_token,
                user_node.unsqueeze(1),
                item_node.unsqueeze(1),
                sequence_tokens,
            ],
            dim=1,
        )
        token_mask = torch.cat(
            [
                torch.ones(batch_size, 3, dtype=torch.bool, device=combined_tokens.device),
                history_mask,
            ],
            dim=1,
        )

        for block in self.blocks:
            combined_tokens = block(combined_tokens, token_mask)

        latent = combined_tokens[:, 0, :]
        logits = self.classifier(torch.relu(self.bottleneck(latent))).squeeze(-1)
        return logits


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> DeepContextNetModel:
    return DeepContextNetModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)