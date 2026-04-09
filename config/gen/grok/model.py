from __future__ import annotations

import math

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


def make_grok_attention_mask(
    user_tokens: int,
    history_tokens: int,
    candidate_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    total_tokens = user_tokens + history_tokens + candidate_tokens
    attention_mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
    prefix_tokens = user_tokens + history_tokens
    if prefix_tokens > 0:
        attention_mask[:prefix_tokens, :prefix_tokens] = torch.tril(
            torch.ones(prefix_tokens, prefix_tokens, dtype=torch.bool, device=device)
        )
    candidate_start = prefix_tokens
    for query_index in range(candidate_start, total_tokens):
        attention_mask[query_index, :prefix_tokens] = True
        attention_mask[query_index, query_index] = True
    return attention_mask


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return self.scale * normalized


def _ffn_size(hidden_dim: int, widening_factor: float) -> int:
    width = int(widening_factor * hidden_dim) * 2 // 3
    return width + (8 - width) % 8


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(hidden_states, 2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)


def _apply_rotary_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    _, sequence_length, _, head_dim = hidden_states.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even to apply rotary embeddings")
    positions = torch.arange(sequence_length, device=hidden_states.device, dtype=torch.float32)
    inverse_frequencies = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device=hidden_states.device, dtype=torch.float32) / head_dim)
    )
    phase = torch.outer(positions, inverse_frequencies)
    cosine = torch.repeat_interleave(torch.cos(phase), 2, dim=-1).unsqueeze(0).unsqueeze(2)
    sine = torch.repeat_interleave(torch.sin(phase), 2, dim=-1).unsqueeze(0).unsqueeze(2)
    cosine = cosine.to(dtype=hidden_states.dtype)
    sine = sine.to(dtype=hidden_states.dtype)
    return hidden_states * cosine + _rotate_half(hidden_states) * sine


class GrokSelfAttention(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if model_config.hidden_dim % model_config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = model_config.hidden_dim
        self.num_heads = model_config.num_heads
        self.head_dim = model_config.hidden_dim // model_config.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("per-head hidden dimension must be even")
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attention_dropout = nn.Dropout(model_config.attention_dropout)
        self.output_dropout = nn.Dropout(model_config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_attention: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.query_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = self.key_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = self.value_projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = _apply_rotary_embedding(query)
        key = _apply_rotary_embedding(key)

        attention_logits = torch.einsum("bthd,bshd->bhts", query, key).float() * self.scale
        attention_logits = 30.0 * torch.tanh(attention_logits / 30.0)
        attention_logits = attention_logits.masked_fill(
            ~allowed_attention.unsqueeze(0).unsqueeze(0),
            -1.0e4,
        )
        attention_logits = attention_logits.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(2),
            -1.0e4,
        )

        attention_weights = torch.softmax(attention_logits, dim=-1).to(dtype=query.dtype)
        attention_weights = self.attention_dropout(attention_weights)
        attended = torch.einsum("bhts,bshd->bthd", attention_weights, value)
        attended = attended.reshape(batch_size, sequence_length, self.hidden_dim)
        attended = self.output_projection(self.output_dropout(attended))
        return attended.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GrokFeedForward(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        feedforward_dim = _ffn_size(model_config.hidden_dim, model_config.ffn_multiplier)
        self.value_projection = nn.Linear(model_config.hidden_dim, feedforward_dim, bias=False)
        self.gate_projection = nn.Linear(model_config.hidden_dim, feedforward_dim, bias=False)
        self.output_projection = nn.Linear(feedforward_dim, model_config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.gelu(self.gate_projection(hidden_states)) * self.value_projection(hidden_states)
        return self.output_projection(self.dropout(gated))


class GrokBlock(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.pre_attention_norm = RMSNorm(model_config.hidden_dim)
        self.post_attention_norm = RMSNorm(model_config.hidden_dim)
        self.pre_ffn_norm = RMSNorm(model_config.hidden_dim)
        self.post_ffn_norm = RMSNorm(model_config.hidden_dim)
        self.attention = GrokSelfAttention(model_config)
        self.feed_forward = GrokFeedForward(model_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_attention: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(self.pre_attention_norm(hidden_states), allowed_attention, padding_mask)
        hidden_states = hidden_states + self.post_attention_norm(attention_output)
        feed_forward_output = self.feed_forward(self.pre_ffn_norm(hidden_states))
        hidden_states = hidden_states + self.post_ffn_norm(feed_forward_output)
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GrokBaselineModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.history_capacity = len(data_config.sequence_names) * data_config.max_seq_len
        self.hidden_dim = model_config.hidden_dim

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
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.history_group_embedding = nn.Embedding(len(data_config.sequence_names) + 1, model_config.hidden_dim, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, model_config.hidden_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.user_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.history_reduce = nn.Sequential(
            nn.Linear(model_config.hidden_dim * 6, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([GrokBlock(model_config) for _ in range(model_config.num_layers)])
        self.final_norm = RMSNorm(model_config.hidden_dim)

        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
        self.readout_scorer = nn.Sequential(
            nn.LayerNorm(model_config.hidden_dim * 4),
            nn.Linear(model_config.hidden_dim * 4, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_config.hidden_dim * 7),
            nn.Linear(model_config.hidden_dim * 7, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise RuntimeError(f"Batch is missing required tensor: {name}")
        return tensor

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def _target_aware_readout(
        self,
        candidate_query: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_query = candidate_query.unsqueeze(1).expand_as(memory_tokens)
        scores = self.readout_scorer(
            torch.cat(
                [
                    expanded_query,
                    memory_tokens,
                    expanded_query * memory_tokens,
                    torch.abs(expanded_query - memory_tokens),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        scores = scores.masked_fill(~memory_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * memory_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.bmm(weights.unsqueeze(1), memory_tokens).squeeze(1)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        user_tokens = self._require(batch.user_tokens, "user_tokens")
        user_mask = self._require(batch.user_mask, "user_mask")
        candidate_post_tokens = self._require(batch.candidate_post_tokens, "candidate_post_tokens")
        candidate_post_mask = self._require(batch.candidate_post_mask, "candidate_post_mask")
        candidate_author_tokens = self._require(batch.candidate_author_tokens, "candidate_author_tokens")
        candidate_author_mask = self._require(batch.candidate_author_mask, "candidate_author_mask")
        history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
        history_author_tokens = self._require(batch.history_author_tokens, "history_author_tokens")
        history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")
        history_time_gap = self._require(batch.history_time_gap, "history_time_gap")
        history_group_ids = self._require(batch.history_group_ids, "history_group_ids")

        user_summary = masked_mean(self._embed_tokens(user_tokens), user_mask)
        context_summary = masked_mean(self._embed_tokens(batch.context_tokens), batch.context_mask)
        dense_summary = self.dense_projection(batch.dense_features)
        user_representation = self.user_reduce(
            torch.cat(
                [
                    user_summary,
                    context_summary,
                    dense_summary,
                    user_summary * context_summary,
                ],
                dim=-1,
            )
        )

        candidate_post_summary = masked_mean(self._embed_tokens(candidate_post_tokens), candidate_post_mask)
        candidate_author_summary = masked_mean(self._embed_tokens(candidate_author_tokens), candidate_author_mask)
        candidate_legacy_summary = masked_mean(self._embed_tokens(batch.candidate_tokens), batch.candidate_mask)
        candidate_representation = self.candidate_reduce(
            torch.cat(
                [
                    candidate_post_summary,
                    candidate_author_summary,
                    candidate_legacy_summary,
                    candidate_post_summary * candidate_author_summary,
                ],
                dim=-1,
            )
        )

        history_representation = self.history_reduce(
            torch.cat(
                [
                    self._embed_tokens(history_post_tokens),
                    self._embed_tokens(history_author_tokens),
                    self._embed_tokens(history_action_tokens),
                    self.time_gap_embedding(history_time_gap.clamp_max(TIME_GAP_BUCKET_COUNT)),
                    self._embed_tokens(batch.history_tokens),
                    self.history_group_embedding(history_group_ids.clamp_max(self.history_group_embedding.num_embeddings - 1)),
                ],
                dim=-1,
            )
        )
        history_representation = history_representation * batch.history_mask.unsqueeze(-1).float()

        batch_size = batch.labels.shape[0]
        user_valid = user_mask.any(dim=1, keepdim=True)
        candidate_valid = candidate_post_mask.any(dim=1, keepdim=True)
        user_segment = self.segment_embedding(
            torch.zeros((batch_size, 1), dtype=torch.long, device=batch.labels.device)
        )
        history_segment = self.segment_embedding(
            torch.ones((batch_size, history_representation.shape[1]), dtype=torch.long, device=batch.labels.device)
        )
        candidate_segment = self.segment_embedding(
            torch.full((batch_size, 1), 2, dtype=torch.long, device=batch.labels.device)
        )

        hidden_states = torch.cat(
            [
                user_representation.unsqueeze(1) + user_segment,
                history_representation + history_segment,
                candidate_representation.unsqueeze(1) + candidate_segment,
            ],
            dim=1,
        )
        padding_mask = torch.cat([~user_valid, ~batch.history_mask, ~candidate_valid], dim=1)
        attention_mask = make_grok_attention_mask(
            user_tokens=1,
            history_tokens=history_representation.shape[1],
            candidate_tokens=1,
            device=hidden_states.device,
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, padding_mask)
        hidden_states = self.final_norm(hidden_states)

        user_output = hidden_states[:, 0]
        history_output = hidden_states[:, 1 : 1 + history_representation.shape[1]]
        candidate_output = hidden_states[:, -1]
        target_context = self._target_aware_readout(
            candidate_output,
            torch.cat([user_output.unsqueeze(1), history_output], dim=1),
            torch.cat([user_valid, batch.history_mask], dim=1),
        )
        history_summary = masked_mean(history_output, batch.history_mask)
        static_context = 0.5 * (context_summary + dense_summary)

        fused = torch.cat(
            [
                candidate_output,
                target_context,
                user_output,
                history_summary,
                static_context,
                candidate_output * target_context,
                torch.abs(candidate_output - user_output),
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> GrokBaselineModel:
    return GrokBaselineModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)