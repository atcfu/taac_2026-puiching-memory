from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_base: float = 10000.0, rope_fraction: float = 1.0) -> None:
        super().__init__()
        rotary_dim = int(head_dim * float(rope_fraction))
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("s,f->sf", positions, self.inv_freq.to(device=device, dtype=dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, -1).unsqueeze(0).unsqueeze(0)
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, -1).unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        first_half = hidden_states[..., ::2]
        second_half = hidden_states[..., 1::2]
        rotated = torch.stack((-second_half, first_half), dim=-1)
        return rotated.flatten(-2)

    def apply_rotary(self, query_states: torch.Tensor, key_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, seq_len, head_dim = query_states.shape
        if head_dim != self.head_dim or self.rotary_dim == 0:
            return query_states, key_states
        cos, sin = self._build_cos_sin(seq_len, query_states.device, query_states.dtype)

        def apply_one(hidden_states: torch.Tensor) -> torch.Tensor:
            rotary_part = hidden_states[..., : self.rotary_dim]
            pass_part = hidden_states[..., self.rotary_dim :]
            rotated = rotary_part * cos + self._rotate_half(rotary_part) * sin
            return torch.cat([rotated, pass_part], dim=-1)

        return apply_one(query_states), apply_one(key_states)


class RelativeTimeBias(nn.Module):
    def __init__(self, num_buckets: int) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_weights = nn.Parameter(torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02))

    def bucketize(self, deltas: torch.Tensor) -> torch.Tensor:
        bucket_ids = (torch.log2(deltas.clamp(min=1.0)).floor() + 1).long()
        return bucket_ids.clamp(min=1, max=self.num_buckets)

    def forward(self, timestamps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        seq_len = timestamps.shape[1]
        time_i = timestamps.unsqueeze(2)
        time_j = timestamps.unsqueeze(1)
        deltas = torch.clamp(time_i - time_j, min=0.0)
        bucket_ids = self.bucketize(deltas)

        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        tril_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=timestamps.device))
        diag_mask = torch.eye(seq_len, dtype=torch.bool, device=timestamps.device)
        pair_valid = pair_valid & tril_mask.unsqueeze(0) & (~diag_mask.unsqueeze(0))
        bucket_ids = torch.where(pair_valid, bucket_ids, torch.zeros_like(bucket_ids))
        return self.bucket_weights[bucket_ids]


class FourierTimeEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_frequencies: int = 8,
        min_period: float = 1.0,
        max_period: float = 256.0,
    ) -> None:
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), steps=num_frequencies)
        self.register_buffer("periods", periods, persistent=False)
        self.register_buffer("two_pi", torch.tensor(2.0 * math.pi), persistent=False)
        self.projection = nn.Linear(2 * num_frequencies, hidden_dim, bias=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        angles = (timestamps.unsqueeze(-1) / self.periods) * self.two_pi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(features)


class HSTU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        rope_fraction: float = 1.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkvu_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 6, bias=False),
            nn.SiLU(),
        )
        self.out_linear = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.rms_norm = RMSNorm(hidden_dim)
        self.rope = RotaryEmbedding(self.head_dim, rope_base=rope_base, rope_fraction=rope_fraction)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rel_ts_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        fused = self.qkvu_linear(hidden_states)
        gated_states, value_states, query_states, key_states = torch.split(
            fused,
            [self.hidden_dim * 3, self.hidden_dim, self.hidden_dim, self.hidden_dim],
            dim=-1,
        )

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_rope, key_rope = self.rope.apply_rotary(query_states, key_states)

        qk_attn = torch.matmul(query_states, key_states.transpose(-2, -1))
        qk_attn = F.relu(qk_attn) / max(seq_len, 1)

        qk_attn_rope = torch.matmul(query_rope, key_rope.transpose(-2, -1))
        qk_attn_rope = F.relu(qk_attn_rope) / max(seq_len, 1)

        if rel_ts_bias is None:
            rel_ts_bias = torch.zeros(batch_size, seq_len, seq_len, device=hidden_states.device, dtype=value_states.dtype)

        if attention_mask is not None:
            masked_positions = attention_mask.logical_not()
            qk_attn = qk_attn.masked_fill(masked_positions.unsqueeze(1), 0.0)
            qk_attn_rope = qk_attn_rope.masked_fill(masked_positions.unsqueeze(1), 0.0)
            rel_ts_bias = rel_ts_bias.masked_fill(masked_positions, 0.0)

        ts_output = torch.einsum("bnm,bhmd->bnhd", rel_ts_bias, value_states)
        rope_output = torch.einsum("bhnm,bhmd->bnhd", qk_attn_rope, value_states)
        plain_output = torch.einsum("bhnm,bhmd->bnhd", qk_attn, value_states)
        combined_output = torch.cat([rope_output, ts_output, plain_output], dim=-1).contiguous()
        combined_output = combined_output.view(batch_size, seq_len, self.hidden_dim * 3)
        next_hidden_states = self.out_linear(combined_output * gated_states)
        next_hidden_states = self.output_dropout(next_hidden_states)
        return self.rms_norm(next_hidden_states + hidden_states)


class FeatureInteractionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float, expansion_factor: int = 4) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.expansion_factor = expansion_factor
        self.down_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(output_dim, output_dim * expansion_factor, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = hidden_states.shape
        hidden_states = self.down_projection(hidden_states)
        expanded = self.projection(hidden_states).view(-1, seq_len, self.expansion_factor, self.output_dim)
        return torch.einsum("bsd,bsrd->bsd", hidden_states, expanded) + hidden_states


class OOModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = model_config.hidden_dim
        self.sequence_count = len(data_config.sequence_names)

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
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
        )
        self.sequence_group_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
        self.sequence_encoder = FeatureInteractionEncoder(model_config.hidden_dim * 9, model_config.hidden_dim, model_config.dropout)
        self.candidate_encoder = FeatureInteractionEncoder(model_config.hidden_dim * 3, model_config.hidden_dim, model_config.dropout)
        self.time_abs_encoding = FourierTimeEncoding(
            hidden_dim=model_config.hidden_dim,
            num_frequencies=8,
            min_period=1.0,
            max_period=float(TIME_GAP_BUCKET_COUNT + data_config.max_seq_len + 1),
        )
        self.rel_time_bias = RelativeTimeBias(num_buckets=64)
        self.attention_layers = nn.ModuleList(
            [
                HSTU(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        self.score_head = nn.Sequential(
            nn.LayerNorm(model_config.hidden_dim * 3),
            nn.Linear(model_config.hidden_dim * 3, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def build_sequence_inputs(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.user_tokens is None or batch.user_mask is None:
            raise RuntimeError("Batch is missing user token fields")
        if batch.history_post_tokens is None or batch.history_author_tokens is None:
            raise RuntimeError("Batch is missing history entity token fields")
        if batch.history_action_tokens is None or batch.history_time_gap is None or batch.history_group_ids is None:
            raise RuntimeError("Batch is missing history metadata fields")

        user_summary = masked_mean(self.embed_tokens(batch.user_tokens), batch.user_mask)
        context_summary = masked_mean(self.embed_tokens(batch.context_tokens), batch.context_mask)
        dense_summary = self.dense_projection(batch.dense_features)

        history_hidden = self.embed_tokens(batch.history_tokens)
        post_hidden = self.embed_tokens(batch.history_post_tokens)
        author_hidden = self.embed_tokens(batch.history_author_tokens)
        action_hidden = self.embed_tokens(batch.history_action_tokens)
        group_hidden = self.sequence_group_embedding(batch.history_group_ids)
        gap_hidden = self.time_gap_embedding(batch.history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))

        seq_len = batch.history_tokens.shape[1]
        user_states = user_summary.unsqueeze(1).expand(-1, seq_len, -1)
        context_states = context_summary.unsqueeze(1).expand(-1, seq_len, -1)
        dense_states = dense_summary.unsqueeze(1).expand(-1, seq_len, -1)
        sequence_inputs = torch.cat(
            [
                user_states,
                context_states,
                dense_states,
                history_hidden,
                post_hidden,
                author_hidden,
                action_hidden,
                gap_hidden,
                group_hidden,
            ],
            dim=-1,
        )
        sequence_states = self.sequence_encoder(sequence_inputs) * batch.history_mask.unsqueeze(-1).float()

        sequence_positions = torch.arange(seq_len, device=sequence_states.device, dtype=torch.float32).unsqueeze(0)
        recency = (
            float(TIME_GAP_BUCKET_COUNT + 1) - batch.history_time_gap.float().clamp(min=0.0, max=float(TIME_GAP_BUCKET_COUNT + 1))
        ) / float(TIME_GAP_BUCKET_COUNT + 1)
        pseudo_timestamps = (sequence_positions + recency) * batch.history_mask.float()
        return sequence_states, batch.history_mask, pseudo_timestamps

    def encode_sequence(self, batch: BatchTensors) -> torch.Tensor:
        sequence_states, sequence_mask, pseudo_timestamps = self.build_sequence_inputs(batch)
        sequence_states = sequence_states * math.sqrt(self.hidden_dim)
        time_states = self.time_abs_encoding(pseudo_timestamps) * sequence_mask.unsqueeze(-1).float()
        sequence_states = sequence_states + time_states

        seq_len = sequence_states.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequence_states.device))
        attention_mask = causal_mask.unsqueeze(0) & sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
        rel_ts_bias = self.rel_time_bias(pseudo_timestamps, sequence_mask)
        for layer in self.attention_layers:
            sequence_states = layer(sequence_states, attention_mask=attention_mask, rel_ts_bias=rel_ts_bias)

        sequence_states = F.normalize(sequence_states, dim=-1)
        last_indices = sequence_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(batch.batch_size, device=sequence_states.device)
        return sequence_states[batch_indices, last_indices]

    def encode_candidate(self, batch: BatchTensors) -> torch.Tensor:
        candidate_summary = masked_mean(self.embed_tokens(batch.candidate_tokens), batch.candidate_mask)
        if batch.candidate_post_tokens is not None and batch.candidate_post_mask is not None:
            candidate_post_summary = masked_mean(self.embed_tokens(batch.candidate_post_tokens), batch.candidate_post_mask)
        else:
            candidate_post_summary = candidate_summary
        if batch.candidate_author_tokens is not None and batch.candidate_author_mask is not None:
            candidate_author_summary = masked_mean(self.embed_tokens(batch.candidate_author_tokens), batch.candidate_author_mask)
        else:
            candidate_author_summary = candidate_summary

        candidate_inputs = torch.cat(
            [candidate_summary, candidate_post_summary, candidate_author_summary],
            dim=-1,
        ).unsqueeze(1)
        candidate_states = self.candidate_encoder(candidate_inputs).squeeze(1)
        return F.normalize(candidate_states, dim=-1)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_repr = self.encode_sequence(batch)
        candidate_repr = self.encode_candidate(batch)
        dot_logits = (sequence_repr * candidate_repr).sum(dim=-1)
        match_features = torch.cat([sequence_repr, candidate_repr, sequence_repr * candidate_repr], dim=-1)
        mlp_logits = self.score_head(match_features).squeeze(-1)
        return dot_logits + mlp_logits


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> OOModel:
    return OOModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)