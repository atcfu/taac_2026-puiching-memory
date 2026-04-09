from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    summed = (tokens * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def apply_token_specific_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = torch.einsum("bnd,ndo->bno", hidden_states, weight)
    if bias is not None:
        projected = projected + bias.unsqueeze(0)
    return projected


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


class AutoSplitNSTokenizer(nn.Module):
    def __init__(self, dense_dim: int, hidden_dim: int, ns_token_count: int, dropout: float) -> None:
        super().__init__()
        self.ns_token_count = ns_token_count
        self.hidden_dim = hidden_dim
        self.group_position_embedding = nn.Embedding(ns_token_count, hidden_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        input_dim = hidden_dim * 6
        self.auto_split = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, ns_token_count * hidden_dim),
        )

    def forward(self, batch: BatchTensors, token_embedding: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.user_tokens is None or batch.user_mask is None:
            raise RuntimeError("Batch is missing user token fields")
        if batch.candidate_post_tokens is None or batch.candidate_post_mask is None:
            raise RuntimeError("Batch is missing candidate post token fields")
        if batch.candidate_author_tokens is None or batch.candidate_author_mask is None:
            raise RuntimeError("Batch is missing candidate author token fields")

        user_summary = masked_mean(token_embedding(batch.user_tokens), batch.user_mask)
        context_summary = masked_mean(token_embedding(batch.context_tokens), batch.context_mask)
        candidate_post_summary = masked_mean(token_embedding(batch.candidate_post_tokens), batch.candidate_post_mask)
        candidate_author_summary = masked_mean(token_embedding(batch.candidate_author_tokens), batch.candidate_author_mask)
        candidate_summary = masked_mean(token_embedding(batch.candidate_tokens), batch.candidate_mask)
        dense_summary = self.dense_projection(batch.dense_features)

        fused = torch.cat(
            [
                dense_summary,
                user_summary,
                context_summary,
                candidate_post_summary,
                candidate_author_summary,
                candidate_summary,
            ],
            dim=-1,
        )
        tokens = self.auto_split(fused).view(batch.batch_size, self.ns_token_count, self.hidden_dim)
        positions = torch.arange(self.ns_token_count, device=tokens.device)
        tokens = tokens + self.group_position_embedding(positions).unsqueeze(0)
        mask = torch.ones(batch.batch_size, self.ns_token_count, dtype=torch.bool, device=tokens.device)
        return tokens, mask


class UnifiedSequentialTokenizer(nn.Module):
    def __init__(self, max_sequence_tokens: int, hidden_dim: int, sequence_group_count: int, dropout: float) -> None:
        super().__init__()
        self.max_sequence_tokens = max_sequence_tokens
        self.hidden_dim = hidden_dim
        self.event_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 6),
            nn.Linear(hidden_dim * 6, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, hidden_dim, padding_idx=0)
        self.sequence_group_embedding = nn.Embedding(sequence_group_count + 1, hidden_dim, padding_idx=0)
        self.sequence_position_embedding = nn.Embedding(max_sequence_tokens, hidden_dim)
        self.sep_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, batch: BatchTensors, token_embedding: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.history_post_tokens is None or batch.history_author_tokens is None:
            raise RuntimeError("Batch is missing history entity token fields")
        if batch.history_action_tokens is None or batch.history_time_gap is None or batch.history_group_ids is None:
            raise RuntimeError("Batch is missing history metadata fields")

        history_hidden = token_embedding(batch.history_tokens)
        post_hidden = token_embedding(batch.history_post_tokens)
        author_hidden = token_embedding(batch.history_author_tokens)
        action_hidden = token_embedding(batch.history_action_tokens)
        time_hidden = self.time_gap_embedding(batch.history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
        group_hidden = self.sequence_group_embedding(batch.history_group_ids)
        event_inputs = torch.cat(
            [
                history_hidden,
                post_hidden,
                author_hidden,
                action_hidden,
                time_hidden,
                group_hidden,
            ],
            dim=-1,
        )
        event_tokens = self.event_projection(event_inputs)
        return self.merge_with_sep(event_tokens, batch.history_group_ids, batch.history_mask)

    def merge_with_sep(
        self,
        event_tokens: torch.Tensor,
        history_group_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = event_tokens.shape[0]
        device = event_tokens.device
        merged_tokens = event_tokens.new_zeros(batch_size, self.max_sequence_tokens, self.hidden_dim)
        merged_mask = torch.zeros(batch_size, self.max_sequence_tokens, dtype=torch.bool, device=device)

        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(history_mask[batch_index], as_tuple=False).squeeze(-1).tolist()
            if not valid_positions:
                continue
            pieces: list[torch.Tensor] = []
            for offset, position in enumerate(valid_positions):
                pieces.append(event_tokens[batch_index, position])
                if offset + 1 >= len(valid_positions):
                    continue
                next_position = valid_positions[offset + 1]
                if int(history_group_ids[batch_index, position]) != int(history_group_ids[batch_index, next_position]):
                    pieces.append(self.sep_token)

            merged_length = min(len(pieces), self.max_sequence_tokens)
            start = self.max_sequence_tokens - merged_length
            merged_tokens[batch_index, start:] = torch.stack(pieces[-merged_length:], dim=0)
            merged_mask[batch_index, start:] = True

        positions = torch.arange(self.max_sequence_tokens, device=device)
        merged_tokens = merged_tokens + self.sequence_position_embedding(positions).unsqueeze(0)
        merged_tokens = merged_tokens * merged_mask.unsqueeze(-1).float()
        return merged_tokens, merged_mask


class MixedCausalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ns_token_count: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ns_token_count = ns_token_count
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.seq_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.seq_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.seq_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ns_query_weight = nn.Parameter(torch.empty(ns_token_count, hidden_dim, hidden_dim))
        self.ns_key_weight = nn.Parameter(torch.empty(ns_token_count, hidden_dim, hidden_dim))
        self.ns_value_weight = nn.Parameter(torch.empty(ns_token_count, hidden_dim, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.ns_query_weight)
        nn.init.xavier_uniform_(self.ns_key_weight)
        nn.init.xavier_uniform_(self.ns_value_weight)

    def reshape_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = hidden_states.shape
        return hidden_states.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
        next_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_queries = sequence_tokens[:, -next_sequence_length:]
        sequence_query_mask = sequence_mask[:, -next_sequence_length:]

        query_states = torch.cat(
            [
                self.seq_query(sequence_queries),
                apply_token_specific_linear(ns_tokens, self.ns_query_weight),
            ],
            dim=1,
        )
        key_states = torch.cat(
            [
                self.seq_key(sequence_tokens),
                apply_token_specific_linear(ns_tokens, self.ns_key_weight),
            ],
            dim=1,
        )
        value_states = torch.cat(
            [
                self.seq_value(sequence_tokens),
                apply_token_specific_linear(ns_tokens, self.ns_value_weight),
            ],
            dim=1,
        )

        query_states = self.reshape_heads(query_states)
        key_states = self.reshape_heads(key_states)
        value_states = self.reshape_heads(value_states)

        sequence_length = sequence_tokens.shape[1]
        total_key_length = sequence_length + self.ns_token_count
        query_positions = torch.cat(
            [
                torch.arange(sequence_length - next_sequence_length, sequence_length, device=sequence_tokens.device),
                torch.arange(sequence_length, total_key_length, device=sequence_tokens.device),
            ]
        )
        key_positions = torch.arange(total_key_length, device=sequence_tokens.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        full_key_mask = torch.cat([sequence_mask, ns_mask], dim=1)
        attention_logits = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) * self.scale
        attention_logits = attention_logits.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1.0e4)
        attention_logits = attention_logits.masked_fill(~full_key_mask.unsqueeze(1).unsqueeze(1), -1.0e4)

        attention_weights = torch.softmax(attention_logits, dim=-1).to(dtype=query_states.dtype)
        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, value_states)
        attended = attended.transpose(1, 2).contiguous().view(
            sequence_tokens.shape[0],
            next_sequence_length + self.ns_token_count,
            self.hidden_dim,
        )
        attended = self.output_projection(self.dropout(attended))

        sequence_output = attended[:, :next_sequence_length] * sequence_query_mask.unsqueeze(-1).float()
        ns_output = attended[:, next_sequence_length:] * ns_mask.unsqueeze(-1).float()
        return sequence_output, sequence_query_mask, ns_output


class MixedFFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, ns_token_count: int, dropout: float) -> None:
        super().__init__()
        self.seq_up = nn.Linear(hidden_dim, ffn_dim)
        self.seq_down = nn.Linear(ffn_dim, hidden_dim)
        self.ns_up_weight = nn.Parameter(torch.empty(ns_token_count, hidden_dim, ffn_dim))
        self.ns_up_bias = nn.Parameter(torch.zeros(ns_token_count, ffn_dim))
        self.ns_down_weight = nn.Parameter(torch.empty(ns_token_count, ffn_dim, hidden_dim))
        self.ns_down_bias = nn.Parameter(torch.zeros(ns_token_count, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.ns_up_weight)
        nn.init.xavier_uniform_(self.ns_down_weight)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_hidden = self.seq_down(self.dropout(F.silu(self.seq_up(sequence_tokens))))
        sequence_hidden = self.dropout(sequence_hidden) * sequence_mask.unsqueeze(-1).float()

        ns_hidden = apply_token_specific_linear(ns_tokens, self.ns_up_weight, self.ns_up_bias)
        ns_hidden = F.silu(ns_hidden)
        ns_hidden = self.dropout(ns_hidden)
        ns_hidden = apply_token_specific_linear(ns_hidden, self.ns_down_weight, self.ns_down_bias)
        ns_hidden = self.dropout(ns_hidden) * ns_mask.unsqueeze(-1).float()
        return sequence_hidden, ns_hidden


class OneTransBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        ns_token_count: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)
        self.attention = MixedCausalAttention(hidden_dim, num_heads, ns_token_count, attention_dropout)
        self.ffn = MixedFFN(hidden_dim, ffn_dim, ns_token_count, dropout)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        sequence_mask: torch.Tensor,
        ns_tokens: torch.Tensor,
        ns_mask: torch.Tensor,
        next_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_sequence = self.attention_norm(sequence_tokens)
        normalized_ns = self.attention_norm(ns_tokens)
        sequence_attention, next_sequence_mask, ns_attention = self.attention(
            normalized_sequence,
            sequence_mask,
            normalized_ns,
            ns_mask,
            next_sequence_length,
        )
        sequence_tokens = sequence_tokens[:, -next_sequence_length:] + sequence_attention
        sequence_mask = next_sequence_mask
        ns_tokens = ns_tokens + ns_attention

        normalized_sequence = self.ffn_norm(sequence_tokens)
        normalized_ns = self.ffn_norm(ns_tokens)
        sequence_ffn, ns_ffn = self.ffn(normalized_sequence, sequence_mask, normalized_ns, ns_mask)
        sequence_tokens = (sequence_tokens + sequence_ffn) * sequence_mask.unsqueeze(-1).float()
        ns_tokens = (ns_tokens + ns_ffn) * ns_mask.unsqueeze(-1).float()
        return sequence_tokens, sequence_mask, ns_tokens, ns_mask


class OneTransModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.sequence_group_count = len(data_config.sequence_names)
        self.history_capacity = self.sequence_group_count * data_config.max_seq_len
        self.max_sequence_tokens = max(1, self.history_capacity * 2 - 1)
        self.ns_token_count = max(1, model_config.segment_count)
        self.hidden_dim = model_config.hidden_dim
        self.ffn_dim = int(model_config.hidden_dim * model_config.ffn_multiplier)

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
        self.sequential_tokenizer = UnifiedSequentialTokenizer(
            max_sequence_tokens=self.max_sequence_tokens,
            hidden_dim=model_config.hidden_dim,
            sequence_group_count=self.sequence_group_count,
            dropout=model_config.dropout,
        )
        self.ns_tokenizer = AutoSplitNSTokenizer(
            dense_dim=dense_dim,
            hidden_dim=model_config.hidden_dim,
            ns_token_count=self.ns_token_count,
            dropout=model_config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                OneTransBlock(
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    ffn_dim=self.ffn_dim,
                    ns_token_count=self.ns_token_count,
                    dropout=model_config.dropout,
                    attention_dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        self.output_head = nn.Sequential(
            nn.LayerNorm((self.ns_token_count * 2) * model_config.hidden_dim),
            nn.Linear((self.ns_token_count * 2) * model_config.hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, model_config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim * 2, 1),
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def make_pyramid_schedule(self, layer_count: int) -> list[int]:
        if layer_count <= 0:
            return []
        schedule: list[int] = []
        for layer_index in range(1, layer_count + 1):
            ratio = layer_index / layer_count
            keep = round(self.max_sequence_tokens + ratio * (self.ns_token_count - self.max_sequence_tokens))
            keep = max(self.ns_token_count, min(self.max_sequence_tokens, keep))
            if schedule:
                keep = min(keep, schedule[-1])
            schedule.append(keep)
        schedule[-1] = self.ns_token_count
        return schedule

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        sequence_tokens, sequence_mask = self.sequential_tokenizer(batch, self.embed_tokens)
        ns_tokens, ns_mask = self.ns_tokenizer(batch, self.embed_tokens)
        pyramid_schedule = self.make_pyramid_schedule(len(self.blocks))

        for block, next_sequence_length in zip(self.blocks, pyramid_schedule, strict=True):
            sequence_tokens, sequence_mask, ns_tokens, ns_mask = block(
                sequence_tokens,
                sequence_mask,
                ns_tokens,
                ns_mask,
                next_sequence_length=next_sequence_length,
            )

        fused = torch.cat([sequence_tokens, ns_tokens], dim=1).reshape(batch.batch_size, -1)
        logits = self.output_head(fused)
        return logits.squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> OneTransModel:
    return OneTransModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)