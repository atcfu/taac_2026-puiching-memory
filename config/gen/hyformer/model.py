from __future__ import annotations

import math

import torch
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


def masked_sequence_mean(sequence_states: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
    weights = sequence_mask.unsqueeze(-1).float()
    summed = (sequence_states * weights).sum(dim=2)
    counts = weights.sum(dim=2).clamp_min(1.0)
    return summed / counts


class SemanticGroupedNSTokenizer(nn.Module):
    def __init__(self, dense_dim: int, hidden_dim: int, ns_token_count: int, dropout: float) -> None:
        super().__init__()
        if ns_token_count < 5:
            raise ValueError("HyFormer requires at least 5 semantic non-sequential tokens")
        self.hidden_dim = hidden_dim
        self.ns_token_count = ns_token_count
        self.base_token_count = 5
        self.extra_dense_token_count = ns_token_count - self.base_token_count
        self.token_position_embedding = nn.Embedding(ns_token_count, hidden_dim)
        self.dense_projection = (
            None
            if self.extra_dense_token_count == 0
            else nn.Sequential(
                nn.Linear(dense_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, self.extra_dense_token_count * hidden_dim),
            )
        )

    def forward(self, batch: BatchTensors, embed_tokens) -> torch.Tensor:
        if batch.user_tokens is None or batch.user_mask is None:
            raise RuntimeError("Batch is missing user token fields")
        if batch.candidate_post_tokens is None or batch.candidate_post_mask is None:
            raise RuntimeError("Batch is missing candidate post token fields")
        if batch.candidate_author_tokens is None or batch.candidate_author_mask is None:
            raise RuntimeError("Batch is missing candidate author token fields")

        base_tokens = torch.stack(
            [
                masked_mean(embed_tokens(batch.user_tokens), batch.user_mask),
                masked_mean(embed_tokens(batch.context_tokens), batch.context_mask),
                masked_mean(embed_tokens(batch.candidate_tokens), batch.candidate_mask),
                masked_mean(embed_tokens(batch.candidate_post_tokens), batch.candidate_post_mask),
                masked_mean(embed_tokens(batch.candidate_author_tokens), batch.candidate_author_mask),
            ],
            dim=1,
        )
        if self.extra_dense_token_count > 0 and self.dense_projection is not None:
            dense_tokens = self.dense_projection(batch.dense_features).view(
                batch.batch_size,
                self.extra_dense_token_count,
                self.hidden_dim,
            )
            ns_tokens = torch.cat([base_tokens, dense_tokens], dim=1)
        else:
            ns_tokens = base_tokens

        positions = torch.arange(self.ns_token_count, device=ns_tokens.device)
        ns_tokens = ns_tokens + self.token_position_embedding(positions).unsqueeze(0)
        return ns_tokens


class MultiSequenceEventTokenizer(nn.Module):
    def __init__(self, max_seq_len: int, hidden_dim: int, sequence_count: int, dropout: float) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.sequence_count = sequence_count
        self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, hidden_dim, padding_idx=0)
        self.sequence_id_embedding = nn.Embedding(sequence_count + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.event_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5),
            nn.Linear(hidden_dim * 5, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.empty_sequence_tokens = nn.Parameter(torch.randn(sequence_count, hidden_dim) * 0.02)

    def forward(self, batch: BatchTensors, embed_tokens) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.history_post_tokens is None or batch.history_author_tokens is None:
            raise RuntimeError("Batch is missing history entity token fields")
        if batch.history_action_tokens is None or batch.history_time_gap is None or batch.history_group_ids is None:
            raise RuntimeError("Batch is missing history metadata fields")

        history_hidden = embed_tokens(batch.history_tokens)
        post_hidden = embed_tokens(batch.history_post_tokens)
        author_hidden = embed_tokens(batch.history_author_tokens)
        action_hidden = embed_tokens(batch.history_action_tokens)
        gap_hidden = self.time_gap_embedding(batch.history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
        event_inputs = torch.cat(
            [history_hidden, post_hidden, author_hidden, action_hidden, gap_hidden],
            dim=-1,
        )
        event_tokens = self.event_projection(event_inputs)

        batch_size = batch.batch_size
        device = event_tokens.device
        sequence_states = event_tokens.new_zeros(batch_size, self.sequence_count, self.max_seq_len, self.hidden_dim)
        sequence_mask = torch.zeros(batch_size, self.sequence_count, self.max_seq_len, dtype=torch.bool, device=device)

        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(batch.history_mask[batch_index], as_tuple=False).squeeze(-1)
            for sequence_index in range(self.sequence_count):
                if valid_positions.numel() > 0:
                    group_positions = valid_positions[
                        batch.history_group_ids[batch_index, valid_positions] == (sequence_index + 1)
                    ]
                else:
                    group_positions = valid_positions

                selected_positions = group_positions[-self.max_seq_len :]
                selected_count = int(selected_positions.numel())
                if selected_count == 0:
                    sequence_states[batch_index, sequence_index, 0] = self.empty_sequence_tokens[sequence_index]
                    sequence_mask[batch_index, sequence_index, 0] = True
                    continue

                sequence_states[batch_index, sequence_index, :selected_count] = event_tokens[
                    batch_index,
                    selected_positions,
                ]
                sequence_mask[batch_index, sequence_index, :selected_count] = True

        position_states = self.position_embedding(torch.arange(self.max_seq_len, device=device)).view(
            1,
            1,
            self.max_seq_len,
            self.hidden_dim,
        )
        sequence_id_states = self.sequence_id_embedding(
            torch.arange(1, self.sequence_count + 1, device=device)
        ).view(1, self.sequence_count, 1, self.hidden_dim)
        sequence_states = sequence_states + position_states + sequence_id_states
        sequence_states = sequence_states * sequence_mask.unsqueeze(-1).float()
        return sequence_states, sequence_mask


class SequenceEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_states: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        normalized_states = self.attention_norm(sequence_states)
        attention_output, _ = self.self_attention(
            normalized_states,
            normalized_states,
            normalized_states,
            key_padding_mask=~sequence_mask,
            need_weights=False,
        )
        sequence_states = (sequence_states + self.dropout(attention_output)) * sequence_mask.unsqueeze(-1).float()
        ffn_output = self.ffn(self.ffn_norm(sequence_states))
        sequence_states = (sequence_states + self.dropout(ffn_output)) * sequence_mask.unsqueeze(-1).float()
        return sequence_states


class QueryGenerator(nn.Module):
    def __init__(
        self,
        sequence_count: int,
        queries_per_sequence: int,
        ns_token_count: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_count = sequence_count
        self.queries_per_sequence = queries_per_sequence
        self.hidden_dim = hidden_dim
        global_info_dim = (sequence_count + ns_token_count) * hidden_dim
        self.query_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(global_info_dim),
                    nn.Linear(global_info_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(sequence_count * queries_per_sequence)
            ]
        )
        self.query_embeddings = nn.Parameter(torch.randn(sequence_count, queries_per_sequence, hidden_dim) * 0.02)

    def forward(self, ns_tokens: torch.Tensor, sequence_pools: torch.Tensor) -> torch.Tensor:
        batch_size = ns_tokens.shape[0]
        global_info = torch.cat(
            [ns_tokens.reshape(batch_size, -1), sequence_pools.reshape(batch_size, -1)],
            dim=-1,
        )
        query_states = ns_tokens.new_zeros(
            batch_size,
            self.sequence_count,
            self.queries_per_sequence,
            self.hidden_dim,
        )
        for sequence_index in range(self.sequence_count):
            for query_index in range(self.queries_per_sequence):
                generator_index = sequence_index * self.queries_per_sequence + query_index
                query_states[:, sequence_index, query_index] = (
                    self.query_generators[generator_index](global_info)
                    + self.query_embeddings[sequence_index, query_index]
                    + sequence_pools[:, sequence_index]
                )
        return query_states


class QueryDecodingBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_states: torch.Tensor,
        sequence_states: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(query_states)
        normalized_sequence = self.sequence_norm(sequence_states)
        attention_output, _ = self.cross_attention(
            normalized_queries,
            normalized_sequence,
            normalized_sequence,
            key_padding_mask=~sequence_mask,
            need_weights=False,
        )
        decoded_queries = query_states + self.dropout(attention_output)
        decoded_queries = decoded_queries + self.dropout(self.ffn(self.ffn_norm(decoded_queries)))
        return decoded_queries


class QueryBoosting(nn.Module):
    def __init__(self, token_count: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.token_count = token_count
        self.mix_dim = math.ceil(hidden_dim / token_count) * token_count
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.pre_projection = (
            nn.Identity()
            if self.mix_dim == hidden_dim
            else nn.Linear(hidden_dim, self.mix_dim)
        )
        self.post_projection = (
            nn.Identity()
            if self.mix_dim == hidden_dim
            else nn.Linear(self.mix_dim, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.per_token_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != self.token_count:
            raise RuntimeError(f"Expected {self.token_count} boosting tokens, got {tokens.shape[1]}")
        normalized_tokens = self.pre_norm(tokens)
        projected_tokens = self.pre_projection(normalized_tokens)
        subspace_dim = self.mix_dim // self.token_count
        mixed_tokens = projected_tokens.view(
            tokens.shape[0],
            self.token_count,
            self.token_count,
            subspace_dim,
        ).transpose(1, 2).reshape(tokens.shape[0], self.token_count, self.mix_dim)
        mixed_tokens = self.post_projection(mixed_tokens)
        boosted_tokens = tokens + self.dropout(mixed_tokens)
        boosted_tokens = boosted_tokens + self.dropout(self.per_token_ffn(self.ffn_norm(boosted_tokens)))
        return boosted_tokens


class HyFormerLayer(nn.Module):
    def __init__(
        self,
        sequence_count: int,
        queries_per_sequence: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        total_token_count: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_count = sequence_count
        self.queries_per_sequence = queries_per_sequence
        self.query_count = sequence_count * queries_per_sequence
        self.hidden_dim = hidden_dim
        self.sequence_encoder = SequenceEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.query_decoder = QueryDecodingBlock(hidden_dim, num_heads, ffn_dim, dropout, attention_dropout)
        self.query_boosting = QueryBoosting(total_token_count, hidden_dim, dropout)

    def forward(
        self,
        sequence_states: torch.Tensor,
        sequence_mask: torch.Tensor,
        query_states: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        updated_sequence_states = []
        decoded_query_states = []
        for sequence_index in range(self.sequence_count):
            sequence_hidden = self.sequence_encoder(sequence_states[:, sequence_index], sequence_mask[:, sequence_index])
            updated_sequence_states.append(sequence_hidden)
            decoded_query_states.append(
                self.query_decoder(query_states[:, sequence_index], sequence_hidden, sequence_mask[:, sequence_index])
            )

        sequence_states = torch.stack(updated_sequence_states, dim=1)
        decoded_queries = torch.stack(decoded_query_states, dim=1)
        boost_input = torch.cat(
            [decoded_queries.reshape(decoded_queries.shape[0], self.query_count, self.hidden_dim), ns_tokens],
            dim=1,
        )
        boosted_tokens = self.query_boosting(boost_input)
        next_queries = boosted_tokens[:, : self.query_count].reshape(
            boosted_tokens.shape[0],
            self.sequence_count,
            self.queries_per_sequence,
            self.hidden_dim,
        )
        next_ns_tokens = boosted_tokens[:, self.query_count :]
        return sequence_states, next_queries, next_ns_tokens


class HyFormerModel(nn.Module):
    def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
        super().__init__()
        self.sequence_count = len(data_config.sequence_names)
        self.queries_per_sequence = max(1, model_config.num_queries)
        self.ns_token_count = max(5, model_config.segment_count)
        self.query_count = self.sequence_count * self.queries_per_sequence
        self.total_token_count = self.query_count + self.ns_token_count
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
        self.ns_tokenizer = SemanticGroupedNSTokenizer(
            dense_dim=dense_dim,
            hidden_dim=model_config.hidden_dim,
            ns_token_count=self.ns_token_count,
            dropout=model_config.dropout,
        )
        self.sequence_tokenizer = MultiSequenceEventTokenizer(
            max_seq_len=data_config.max_seq_len,
            hidden_dim=model_config.hidden_dim,
            sequence_count=self.sequence_count,
            dropout=model_config.dropout,
        )
        self.query_generator = QueryGenerator(
            sequence_count=self.sequence_count,
            queries_per_sequence=self.queries_per_sequence,
            ns_token_count=self.ns_token_count,
            hidden_dim=model_config.hidden_dim,
            dropout=model_config.dropout,
        )
        self.layers = nn.ModuleList(
            [
                HyFormerLayer(
                    sequence_count=self.sequence_count,
                    queries_per_sequence=self.queries_per_sequence,
                    hidden_dim=model_config.hidden_dim,
                    num_heads=model_config.num_heads,
                    ffn_dim=self.ffn_dim,
                    total_token_count=self.total_token_count,
                    dropout=model_config.dropout,
                    attention_dropout=model_config.attention_dropout,
                )
                for _ in range(model_config.num_layers)
            ]
        )
        head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 4
        head_input_dim = self.total_token_count * model_config.hidden_dim
        self.output_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(head_hidden_dim, model_config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.hidden_dim * 2, 1),
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_projection(self.token_embedding(tokens))

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        ns_tokens = self.ns_tokenizer(batch, self.embed_tokens)
        sequence_states, sequence_mask = self.sequence_tokenizer(batch, self.embed_tokens)
        sequence_pools = masked_sequence_mean(sequence_states, sequence_mask)
        query_states = self.query_generator(ns_tokens, sequence_pools)

        for layer in self.layers:
            sequence_states, query_states, ns_tokens = layer(sequence_states, sequence_mask, query_states, ns_tokens)

        top_tokens = torch.cat(
            [query_states.reshape(batch.batch_size, self.query_count, self.hidden_dim), ns_tokens],
            dim=1,
        )
        logits = self.output_head(top_tokens.reshape(batch.batch_size, -1))
        return logits.squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> HyFormerModel:
    return HyFormerModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)